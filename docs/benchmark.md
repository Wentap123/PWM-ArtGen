# PM and ACD benchmarks

Commands run from the repository root. See [setup](setup.md) and [data preparation](data.md).


These use the prepared dataset observations and GT part structure, not the
SAM/SAM3 photo pipeline. All inference entrypoints default to
`checkpoints/pwm_final.pt`; historical experiment checkpoints are not used.

```bash
bash scripts/infer_pm.sh --data_root data/benchmarks/pm --vae_path /path/to/sdxl_vae
bash scripts/infer_acd.sh --data_root data/benchmarks/acd --vae_path /path/to/sdxl_vae
```

The public inference subset uses **two views per object**, evaluated separately.
They are stored as `view_0` and `view_1`; `views.json` contains the sample records
needed for loading. `--view_ids 0` selects only the first view.

Both datasets have the same portable layout:

```text
pm/ or acd/
  dataset.json, test_ids.json, checksums.json
  <object_key>/
    graph.json, views.json
    view_0/  # RGB and label masks (PM: per-joint subdirectories)
    view_1/
```

Object keys are `<category>/<id>` for PM and `<source>/<category>/<id>` for ACD.
Only observation images, masks, graph structure and bboxes are copied; meshes,
other views and PM motion target frames are excluded. Inference does not read
target frames. Relative image paths allow moving the dataset, and SHA-256
checksums record every copied image and annotation. The original data is unchanged.
Ground-truth meshes and the retrieval database remain external.

Maintainer utility only: to export already rendered/prepared sources into a fresh directory (end users should download the HF archives):

```bash
python scripts/export_test_data.py --dataset pm --data_root /path/to/rendered_pm --out_dir /path/to/pm
python scripts/export_test_data.py --dataset acd --data_root /path/to/prepared_acd --out_dir /path/to/acd
```

Use `--dry-run` to preflight the copy. Test IDs default to the exported dataset's
`test_ids.json`, or can be overridden with `--test_ids`.

Legacy source layouts remain supported. PM reads the rendered
`test/<category>/<id>` layout (frames 0/2, views 00/01 by default).
The Singapo mesh database is **not** the PM rendered input directory. Legacy ACD uses
`<source>/<category>/<id>/{object_pwm.json,info/10_bbox.json,rgbs/10.png,masks/10.png}`,
where source is `abo-data` or `hssd-data`. Legacy `object_uwm.json` is accepted.

Fixed manifests are bundled in `splits/pm_test.json` (77 objects from the local
rendered test split) and `splits/acd_test.json` (135 objects in the prepared two-view dataset). These are path-based manifests, not a claim that all annotations have
been validated. Missing/unusable selected objects raise errors. Override with
`--test_ids /path/to/ids.json`; `--num_samples 1` is useful for a smoke test.
To prepare a manifest for another dataset copy without loading images:

```bash
python scripts/prepare_test_ids.py --dataset pm --data_root /path/to/rendered_data --output /path/to/pm_ids.json
python scripts/prepare_test_ids.py --dataset acd --data_root /path/to/prepared_acd --output /path/to/acd_ids.json
```

Portable-dataset predictions use `<object_key>/view_0|view_1/joint_<id>/`.
Legacy PM predictions use `<id>/joint_<id>/view_idx_XX/`, and legacy ACD uses
`<source>/<id>/joint_<id>/`. The following entrypoint assembles those
predictions, applies postprocessing, retrieves meshes, and computes Singapo
AS/RS IoU, center distance, Chamfer distance, and AOR in the `pwm` environment:

The retained Singapo keys `AS-IoU` and `RS-IoU` contain **1 − GIoU errors**, not
ordinary IoUs (the local implementation calls `sampling_giou`). Lower is better
for all reported metrics; undefined AOR is excluded
from its mean.

```bash
bash scripts/evaluate.sh --dataset pm --data_root data/benchmarks/pm \
  --gt_root data/raw/pm --database_root data/raw/pm --hashbook data/raw/pm/pwm_hash_filtered.json \
  --pred_root outputs/pm/predictions --out_dir outputs/pm/evaluation
bash scripts/evaluate.sh --dataset acd --data_root data/benchmarks/acd \
  --gt_root data/raw/acd --database_root data/raw/pm --hashbook data/raw/pm/pwm_hash_filtered.json \
  --pred_root outputs/acd/predictions --out_dir outputs/acd/evaluation
```

Use matching `--test_ids` and PM `--view_ids` in inference and evaluation.
GT annotations must use the merged-handle PWM representation. Postprocessing
is enabled by default (`--no-postprocess` disables it), and frontal images
generally work best. Use `--dry-run` to validate assembly without retrieval or
metric computation, and a fresh output directory for real evaluation.
`metrics.json` reports failures and averages views per object, then objects;
partial results are explicitly marked incomplete and return a nonzero status.
`--workers 4` evaluates independent objects concurrently with stable per-object
seeds. After a failed run, `--resume` validates and reuses completed object
results, backs up the previous summary, and retries failed objects. Only use it
when inputs, predictions, retrieval database and settings are unchanged. Partial
mesh outputs from a failed object must be moved aside before retrying that object.
Missing parts fall back to other indexed categories if the source category has
none (for example, drawers in ACD ovens).
After both benchmarks finish, audit coverage and save an overall/per-view
summary including the checkpoint SHA-256:

```bash
python scripts/summarize_results.py --out_root outputs --datasets_root data/benchmarks \
  --ckpt checkpoints/pwm_final.pt --hashbook data/raw/pm/pwm_hash_filtered.json
```

