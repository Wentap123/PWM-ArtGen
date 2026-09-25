"""Regression checks without model downloads or GPU use."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def entrypoint(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


train = entrypoint("pwm_train", "train.py")
infer = entrypoint("pwm_infer", "infer.py")


class Observations(Dataset):
    low_dim_dim, ha, action_dim = 8, 1, 20

    def __init__(self, **kwargs):
        pass

    def __len__(self):
        return 5

    def __getitem__(self, index):
        image = torch.full((1, 2, 2, 3), index, dtype=torch.uint8)
        return {"obs": {"rgb_00": image, "rgb_01": image.clone(),
                        "low_dim": torch.full((1, 8), float(index))},
                "meta": {"oid": str(index), "jid": torch.tensor(index + 10),
                         "views": list(range(index + 1)),
                         "jdir_sample": {index: f"joint_{index}"}}}


class Encoder:
    decode_calls = 0

    def __init__(self, **kwargs):
        pass

    def to(self, device):
        return self

    def apply_vae(self, latents, inverse=False):
        assert inverse
        type(self).decode_calls += 1
        latents.add_(0.25)  # Like the real decoder, this mutates its input.
        return latents[:, :, :3]


class InferenceModel:
    def __init__(self, obs_encoder, **kwargs):
        self.obs_encoder = obs_encoder

    def to(self, device):
        return self

    def register_type_codes(self, codes):
        pass

    def load_state_dict(self, state, strict=False):
        pass

    def eval(self):
        return self

    def sample_geom(self, obs, steps):
        ids = obs["low_dim"][:, 0, 0]
        return {"axis_dir": ids[:, None].repeat(1, 3),
                "type": torch.zeros(len(ids), dtype=torch.long),
                "type_sim": torch.ones(len(ids), 2),
                "action_raw": ids[:, None, None].repeat(1, 1, 20)}

    def sample_marginal_next_obs(self, obs):
        return torch.zeros(len(obs["low_dim"]), 1, 4, 2, 2, 2)


class Accelerator:
    is_main_process = True
    device = torch.device("cpu")

    def gather_for_metrics(self, value):
        return value

    def unwrap_model(self, model):
        return model

    def wait_for_everyone(self):
        pass


class ValidationModel:
    def __init__(self):
        self.training, self.calls = True, 0

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return torch.tensor(1.0), {"loss": torch.tensor(1.0)}


class ReleaseTests(unittest.TestCase):
    def test_pretrained_blocks_are_loaded_only_when_explicit(self):
        from models.pwm.pwm import PartWorldModel

        encoder = Mock()
        encoder.latent_img_shape.return_value = (1, 4, 2, 4, 4)
        encoder.feat_dim.return_value = 8
        kwargs = dict(action_len=1, action_dim=20, obs_encoder=encoder,
                      embed_dim=12, timestep_embed_dim=12, depth=1, num_heads=3,
                      latent_patch_shape=(2, 2, 2))
        with patch("torch.load") as loader:
            source = PartWorldModel(**kwargs)
            loader.assert_not_called()
        block_state = {"noise_pred_net.blocks." + key: torch.full_like(value, 0.25)
                       for key, value in source.noise_pred_net.blocks.state_dict().items()}
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            checkpoint = Path(tmp) / "blocks.pt"
            torch.save({"model": block_state}, checkpoint)
            restored = PartWorldModel(**kwargs, pretrained_blocks_path=str(checkpoint))
        for key, value in restored.noise_pred_net.blocks.state_dict().items():
            torch.testing.assert_close(value, block_state["noise_pred_net.blocks." + key], rtol=0, atol=0)

    def test_launchers_do_not_enable_pretrained_blocks(self):
        env = dict(os.environ, PYTHON_BIN="/bin/echo", NUM_PROCESSES="1")
        for script in ["train.sh", "infer.sh"]:
            result = subprocess.run(["bash", str(ROOT / "scripts" / script), "--data_root", "example"],
                                    cwd="/tmp", env=env, check=True, capture_output=True, text=True)
            self.assertNotIn("--pretrained_blocks_path", result.stdout)
            self.assertIn(str(ROOT / script.replace(".sh", ".py")), result.stdout)
            self.assertTrue(result.stdout.strip().endswith("--data_root example"))

    def test_default_training_has_no_pretrained_blocks(self):
        with patch.object(sys, "argv", ["train.py", "--data_root", "unused", "--vae_path", "unused"]), \
             patch.object(train, "Accelerator", side_effect=RuntimeError("stop after parsing")):
            with self.assertRaisesRegex(RuntimeError, "stop after parsing"):
                train.main()
        self.assertIsNone(train.args.pretrained_blocks_path)
        self.assertEqual(train.args.eval_batches, 0)

    def test_inference_limits_and_batching(self):
        for limit, batch_size, expected in [(0, 2, 5), (3, 2, 3), (20, 2, 5), (2, 1, 2)]:
            with self.subTest(limit=limit, batch_size=batch_size), tempfile.TemporaryDirectory() as tmp:
                Encoder.decode_calls = 0
                ckpt, out = Path(tmp) / "checkpoint.pt", Path(tmp) / "predictions"
                torch.save({"model": {}}, ckpt)
                argv = ["infer.py", "--data_root", tmp, "--vae_path", tmp,
                        "--ckpt", str(ckpt), "--out_dir", str(out), "--workers", "0",
                        "--batch_size", str(batch_size), "--num_samples", str(limit)]
                with patch.object(sys, "argv", argv), \
                     patch.object(infer, "PartNetMobilityInferenceDataset", Observations), \
                     patch.object(infer, "PWMObservationEncoder", Encoder), \
                     patch.object(infer, "PartWorldModel", InferenceModel), \
                     patch.object(torch.cuda, "is_available", return_value=False), \
                     contextlib.redirect_stdout(io.StringIO()):
                    infer.main()
                self.assertEqual(len(list(out.rglob("geom_pred.json"))), expected)
                self.assertEqual(Encoder.decode_calls, (expected + batch_size - 1) // batch_size)
                for index in range(expected):
                    directory = out / str(index) / f"joint_{index + 10}"
                    pred = json.loads((directory / "geom_pred.json").read_text())["pred"]
                    self.assertEqual(pred["axis_dir"], [float(index)] * 3)
                    self.assertEqual(pred["action_raw"], [[float(index)] * 20])
                    self.assertTrue((directory / "next_pred_rgb_view_00_t01.png").is_file())

    def test_validation_limit_and_logging_opt_in(self):
        loader = DataLoader([{"obs": {}, "next": {}, "actions": torch.zeros(1, 20)}
                             for _ in range(5)], batch_size=1)
        for limit, expected, logging in [(0, 5, False), (2, 2, False), (1, 1, True)]:
            with self.subTest(limit=limit, logging=logging):
                model, tracker = ValidationModel(), Mock()
                with patch.object(train, "wandb", tracker), contextlib.redirect_stdout(io.StringIO()):
                    train.maybe_evaluate(1, 1, model, loader, Accelerator(), False,
                                         eval_batches=limit, use_wandb=logging)
                self.assertEqual(model.calls, expected)
                self.assertTrue(model.training)
                self.assertEqual(tracker.log.call_count, int(logging))

    def test_checkpoint_roundtrip(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters())
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        scheduler.step()
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            train.maybe_save_checkpoint(1, 1, 10, tmp + "/", model, optimizer,
                                        scheduler, Accelerator())
            saved = next(Path(tmp).glob("*.step1.pt"))
            restored = torch.nn.Linear(2, 1)
            restored_optimizer = torch.optim.AdamW(restored.parameters())
            restored_scheduler = torch.optim.lr_scheduler.LambdaLR(restored_optimizer, lambda _: 1)
            step = train.maybe_resume_checkpoint(str(saved), restored, restored_optimizer,
                                                restored_scheduler, Accelerator())
            self.assertEqual(step, 1)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)
            self.assertTrue(restored_optimizer.state)
            self.assertEqual(scheduler.state_dict(), restored_scheduler.state_dict())


if __name__ == "__main__":
    unittest.main()
