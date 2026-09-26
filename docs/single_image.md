# Single-image demo

Commands run from the repository root. See [setup](setup.md) and [data preparation](data.md).


Start from one photo and run preprocessing, PWM inference, and mesh retrieval.
SAM3 is the default; `--seg_backend sam` selects the original SAM ViT-H path.
Frontal views with a clearly visible object generally work best.

First follow the official environment and checkpoint setup for
[SAM3](https://github.com/facebookresearch/sam3#installation) or
[SAM](https://github.com/facebookresearch/segment-anything#installation), then
install the optional photo preprocessing dependencies in `requirements.txt` in the selected segmentation environment. See [preprocessing setup and attribution](../preprocess/NOTICE.md)
for RMBG background removal and model requirements.

```bash
conda activate pwm
export SAM3_PYTHON=/path/to/sam3-env/bin/python
export OPENAI_API_KEY=your_api_key

bash demo/infer_image.sh --image /path/to/photo.png --out_dir /path/to/result \
  --seg_checkpoint /path/to/sam3.pt --rmbg_model /path/to/RMBG-2.0 \
  --vae_path /path/to/sdxl_vae --database_root /path/to/mesh_database
```

For SAM, set `SAM_PYTHON` instead and pass `--seg_backend sam` with its ViT-H
checkpoint. Images with a non-opaque alpha channel can omit `--rmbg_model`.
Weights remain external, except the local PWM default `checkpoints/pwm_final.pt`.

Graph prediction sends the processed image to the configured API and may incur
charges. SAM also sends a labeled segmentation image for region assignment.
Use `OPENAI_BASE_URL` only for an explicitly chosen compatible endpoint;
`--graph_model` defaults to the original `gpt-5-2025-08-07`.
`--graph_json` and (SAM only) `--assignments_json` accept cached responses.

Results are grouped into `preprocess/`, `predictions/`, and `retrieval/`.
The pipeline writes prepared inputs directly under
`preprocess/data/test/<category>/input/view_id_00/`, with `joint_<id>/`
observations and a `process/` directory for the graph and mask. These files are
generated automatically; users only supply the photo and model paths.

Postprocessing is enabled unless `--no-postprocess` is supplied. Use a fresh
output directory for a different image or configuration. Completed stages and
API responses are cached; continue with `bash demo/infer_image.sh --out_dir /path/to/result`
or select `--stage preprocess|inference|retrieval`. `--dry-run` prints commands
without model execution or API calls. Failed retrieval outputs must be moved
aside before retrying; original images are never modified.

Supported categories are StorageFurniture, Table, Refrigerator, Oven, Microwave,
WashingMachine, and Dishwasher. Use `--category` if automatic recognition is
ambiguous. Handles are part of their parent masks, not separate nodes.


## Mesh retrieval

First download the [Singapo preprocessed PartNet-Mobility data](https://github.com/3dlg-hcvc/singapo#download-data)
used for part retrieval, then convert its object annotations in the `pwm` environment:

```bash
python scripts/prepare_database.py \
  --database_root /path/to/mesh_database
```

This adds `object_pwm.json` beside each valid `object.json` and generates
`pwm_hash_filtered.json` plus `pwm_database_report.json` at the database root.
Original annotations and meshes are preserved. Handles are merged into parent
mesh references; prismatic parts named `handle` are retained as drawers.
Hashes are recomputed **after** conversion; the original Singapo index is used
only as a candidate allowlist. Conversion and retrieval require `networkx==3.4.2`
from `requirements.txt` to keep hashes compatible.

Use `--dry-run` to inspect the conversion, `--overwrite` to replace different
generated annotations/indexes, and `--reference_hashbook /path/to/index.json`
to supply an alternative category/hash/object-ID allowlist. Identical outputs
are skipped. Invalid objects are reported and excluded; any failure or an empty
candidate index returns a nonzero exit status. The conversion report is refreshed
on each non-dry run. Only the seven PWM categories are processed.

Run retrieval after inference in the same `pwm` environment:

```bash
bash scripts/retrieve.sh \
  --pred_root /path/to/predictions \
  --data_root /path/to/data \
  --database_root /path/to/mesh_database \
  --out_dir /path/to/retrieval_results
```

Postprocessing is enabled by default; use `--no-postprocess` to disable it.
Frontal images generally give better postprocessing and retrieval results.
The source view must include `process/graph_renum.json` and the per-joint bbox
files described above. Use `--view_id` if multiple source views are available.

Outputs contain `object_pwm.json`, `object.ply`, and part meshes under
`0@<category>@<object_id>/0/`. Use a fresh output directory. Original predictions
are preserved. Existing databases with the legacy object filename are accepted.
Add `--evaluate --gt_root /path/to/ground_truth` for optional evaluation, or
`--dry-run` to check inputs without writing files. `--workers` controls object
parallelism; `--seed` enables reproducible sampling. `RETRIEVAL_PYTHON` can select
a specific interpreter instead of activating the environment.

Retrieval automatically uses the database's generated `pwm_hash_filtered.json`.
Both `scripts/retrieve.sh` and `demo/infer_image.sh` accept `--hashbook /path/to/index.json`
to override it. Without a database index, the bundled legacy PWM index remains
available for existing prepared databases. Object ranking and missing-part
search both stay within the selected index's candidates.

Retrieval dependencies are included in `requirements.txt`. Evaluation also
requires PyTorch3D in the same `pwm` environment; see [setup](setup.md).

Mesh retrieval is adapted from [Singapo](https://github.com/3dlg-hcvc/singapo).
In our representation, handles are merged into their parent parts and are not
retrieved separately. See [source attribution and license](../retrieval/NOTICE.md).

