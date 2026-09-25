"""Lightweight retrieval orchestration and geometry regressions."""
from concurrent.futures import Future
from copy import deepcopy
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from retrieval import pipeline
from retrieval.assemble import assemble_object
from retrieval.database import object_json_path
from retrieval.postprocess import align_predictions


def prediction():
    return dict(axis_dir=[0.0, 0.0, 2.0], axis_ori=[0.0, 0.0, 0.1], range=[0.25], type=0,
                aabb_min=[-0.4, -0.4, -0.2], aabb_max=[0.4, 0.4, 0.2],
                aabb_base_min=[-1.0, -1.0, -0.5], aabb_base_max=[1.0, 1.0, 0.5])


def graph(count=1):
    return {"meta": {"obj_cat": "Dishwasher"}, "diffuse_tree": [
        {"id": 0, "parent": -1, "name": "base", "children": list(range(1, count+1))},
        *[{"id": i, "parent": 0, "name": "door", "children": []} for i in range(1, count+1)]]}


def bbox(index=1):
    return {"id": index, "base_bbox_cxcywh_norm": [0.5, 0.5, 1.0, 1.0],
            "joint_bbox_cxcywh_norm": [0.25 if index == 1 else 0.75, 0.5, 0.4, 0.6]}


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.pred = self.root / "predictions"
        self.data = self.root / "data"
        self.database = self.root / "database"
        self.output = self.root / "results"
        (self.database / "Dishwasher").mkdir(parents=True)
        pipeline.write_json(self.pred / "123/joint_1/geom_pred.json", {"pred": prediction()})
        self.view = self.data / "test/Dishwasher/123/waiting_use/view_id_04"
        pipeline.write_json(self.view / "process/graph_renum.json", graph())
        pipeline.write_json(self.view / "joint_1/mask_renum_bbox.json", bbox())

    def arguments(self, *extra):
        return ["--pred_root", str(self.pred), "--data_root", str(self.data),
                "--database_root", str(self.database), "--out_dir", str(self.output), *extra]

    def test_assembly_preserves_geometry_and_does_not_mutate(self):
        source, pred = graph(), prediction()
        original = deepcopy((source, pred))
        obj = assemble_object(source, [pred])
        door = obj["diffuse_tree"][1]
        self.assertEqual(door["joint"]["range"], [0.0, 90.0])
        self.assertEqual(door["joint"]["axis"]["direction"], [0.0, 0.0, 1.0])
        self.assertEqual(obj["diffuse_tree"][0]["aabb"]["size"], [2.0, 2.0, 1.0])
        self.assertEqual((source, pred), original)

    def test_invalid_graphs_and_independent_handles_fail(self):
        with self.assertRaisesRegex(ValueError, "count mismatch"):
            assemble_object(graph(2), [prediction()])
        source = graph()
        source["diffuse_tree"][1]["name"] = "handle"
        with self.assertRaisesRegex(ValueError, "merged"):
            assemble_object(source, [prediction()])
        source = graph()
        source["diffuse_tree"][1]["children"] = [0]
        with self.assertRaises(ValueError):
            assemble_object(source, [prediction()])

    def test_alignment_moves_axis_with_box_and_preserves_depth(self):
        preds = [prediction(), prediction()]
        original = deepcopy(preds)
        aligned = align_predictions(preds, [bbox(1), bbox(2)], ["id=1", "id=2"])
        self.assertNotEqual(aligned, preds)
        for old, new in zip(preds, aligned):
            for field in ("aabb_min", "aabb_max", "aabb_base_min", "aabb_base_max", "axis_ori"):
                self.assertEqual(old[field][2], new[field][2])
            center = (np.array(new["aabb_min"]) + new["aabb_max"]) / 2
            np.testing.assert_allclose(np.array(new["axis_ori"])[:2], center[:2])
        self.assertEqual(preds, original)

    def test_alignment_skip_gates(self):
        pred = prediction()
        self.assertEqual(align_predictions([pred], [bbox()], ["id=1"]), [pred])
        left, right = prediction(), prediction()
        left.update(aabb_min=[-0.9, -0.2, -0.2], aabb_max=[-0.5, 0.2, 0.2])
        right.update(aabb_min=[0.5, -0.2, -0.2], aabb_max=[0.9, 0.2, 0.2])
        self.assertEqual(align_predictions([left, right], [bbox(1), bbox(2)], ["a", "b"]), [left, right])

    def test_default_postprocess_and_disable(self):
        args = pipeline.parser().parse_args(self.arguments())
        self.assertFalse(args.no_postprocess)
        job = pipeline.prepare_job(self.pred / "123", args)
        self.assertTrue(job["postprocess"])
        self.assertFalse(job["evaluate"])
        args.no_postprocess = True
        (self.view / "joint_1/mask_renum_bbox.json").unlink()
        self.assertFalse(pipeline.prepare_job(self.pred / "123", args)["postprocess"])

    def test_ambiguous_view_requires_selection(self):
        other = self.view.parent / "view_id_05"
        pipeline.write_json(other / "process/graph_renum.json", graph())
        args = pipeline.parser().parse_args(self.arguments())
        with self.assertRaisesRegex(ValueError, "specify --view_id"):
            pipeline.prepare_job(self.pred / "123", args)
        args.view_id = "4"
        self.assertEqual(pipeline.prepare_job(self.pred / "123", args)["object_id"], "123")

    def test_dry_run_and_external_working_directory(self):
        result = subprocess.run(["bash", str(ROOT / "scripts/retrieve.sh"), *self.arguments("--dry-run")],
                                cwd=self.root, env=dict(os.environ, RETRIEVAL_PYTHON=sys.executable),
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("postprocess=True", result.stdout)
        self.assertFalse(self.output.exists())

    def test_workers_and_failure_status(self):
        sizes = []
        class Executor:
            def __init__(self, max_workers, **kwargs):
                sizes.append(max_workers)
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def submit(self, function, job):
                future = Future()
                future.set_exception(RuntimeError("Missing part mesh"))
                return future
        with patch.object(pipeline, "ProcessPoolExecutor", Executor), contextlib.redirect_stdout(io.StringIO()):
            status = pipeline.main(self.arguments("--workers", "3"))
        self.assertEqual(sizes, [3])
        self.assertEqual(status, 1)
        report = pipeline.read_json(self.output / "retrieval_summary.json")
        self.assertIn("Missing part mesh", report[0]["error"])
        self.assertFalse((self.output / "0@Dishwasher@123/0/object_pwm.json").exists())

    def test_evaluation_requires_explicit_ground_truth(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(pipeline.main(self.arguments("--evaluate", "--dry-run")), 1)
        self.assertFalse(self.output.exists())

    def test_database_prefers_pwm_and_accepts_legacy(self):
        directory = self.database / "Dishwasher/123"
        pipeline.write_json(directory / "object_uwm.json", graph())
        self.assertEqual(object_json_path(directory).name, "object_uwm.json")
        pipeline.write_json(directory / "object_pwm.json", graph())
        self.assertEqual(object_json_path(directory).name, "object_pwm.json")


if __name__ == "__main__":
    unittest.main()
