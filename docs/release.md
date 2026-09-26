# Maintainer release procedure

The intended GitHub repository is https://github.com/Wentap123/PWM-ArtGen.
Confirm the project license, authors/paper/citation, and release metadata
before publishing. Preserve third-party license files and NOTICE documents.
The commands below describe asset preparation and updates. The initial HF
assets have been uploaded and verified; GitHub has not been pushed by this task.

## Files to publish

GitHub: root README, requirements, train/infer/evaluate entrypoints; `models/`,
`datasets/`, `preprocess/`, `retrieval/` (including assets and licenses),
`evaluation/`, `splits/`, `scripts/`, `demo/`, `docs/`, and `.gitignore`.
Exclude checkpoints, raw/prepared datasets, predictions, caches, API secrets,
local environment directories and release staging. Review `git status` and the
staged file list before committing; `.gitignore` does not untrack existing files.

HF model repository: `pwm_final.pt`, `SHA256SUMS`, and the reviewed model card
from `docs/huggingface/model_card.md` renamed `README.md`.
HF dataset repository: `pwm_pm_test.zip`, `pwm_acd_test.zip`, `SHA256SUMS`, and
the reviewed dataset card renamed `README.md`. Use immutable HF revisions in
final experiment records. The public namespace is `Wentap`.

## Stage weights and inputs

Run from the repository root, using existing prepared PM/ACD inputs:

```bash
export PREPARED_ROOT=/path/to/prepared-benchmarks
mkdir -p release-staging/model release-staging/dataset release-staging/inputs
cp checkpoints/pwm_final.pt release-staging/model/
cp docs/huggingface/model_card.md release-staging/model/README.md
cp docs/huggingface/dataset_card.md release-staging/dataset/README.md
cp -a "$PREPARED_ROOT/pm" "$PREPARED_ROOT/acd" release-staging/inputs/
```

Use a fresh staging directory. Copy the checkpoint unchanged: it includes the
saved training state. Do not strip optimizer state or reserialize it for this
release. The reference file is 3,559,406,910 bytes, SHA-256:

```text
c9be8ab16b3eb1f978bd1b95a95f1f2292af575848684a0473c23af854a1ef5b
```

Sanitize only staged dataset metadata. Remove machine-local `source_root`;
replace the dataset-level `selection` description with the actual protocol.
Do not change images, masks, graphs, test IDs, or per-object `views.json`.
Existing member checksums remain unchanged; regenerate archive checksums after
packaging. Inspect other metadata for machine-specific paths before release.

```bash
"$PYTHON_BIN" - <<'PYCODE'
import json
from pathlib import Path
selection = {
    'pm': 'Source views 00 and 01 map to view_0 and view_1; frame 0 observations.',
    'acd': 'Source view 10 maps to view_0; the highest-scoring other complete frontal-like view maps to view_1 (0.25 silhouette IoU + 0.75 mean part-mask IoU).',
}
for dataset, description in selection.items():
    path = Path('release-staging/inputs') / dataset / 'dataset.json'
    data = json.loads(path.read_text())
    data.pop('source_root', None)
    data['selection'] = description
    path.write_text(json.dumps(data, indent=2) + '\n')
PYCODE
(cd release-staging/inputs && zip -qr ../dataset/pwm_pm_test.zip pm)
(cd release-staging/inputs && zip -qr ../dataset/pwm_acd_test.zip acd)
(cd release-staging/model && sha256sum pwm_final.pt > SHA256SUMS)
(cd release-staging/dataset && sha256sum pwm_pm_test.zip pwm_acd_test.zip > SHA256SUMS)
```

Verify archives extract as `pm/` and `acd/`, preserve all member checksums and
fixed manifests, and contain no meshes or training target frames. Check object
counts (77/134), view counts (154/268), and complete dataset metadata.

## Validate and publish

Run the local maintainer regression suite before preparing a release.
The `tests/` directory is kept locally and excluded from the public repository.

The local regression tests use temporary fixtures; they do not establish full benchmark
reproducibility or a clean installation. In a separate fresh checkout and output
directory, follow README downloads/conversion/inference/evaluation end to end
before declaring a public release reproduced. Record code revision, environment
versions, checkpoint/archive/index hashes, conversion reports and command logs.
Require complete evaluation with zero failures. No real benchmark was rerun as
part of the directory/documentation reorganization.

After reviewing rights, cards, staged files and actual HF repository IDs, create
the repositories under your account and upload explicitly:

```bash
export HF_MODEL_REPO='Wentap/PWM-ArtGen'
export HF_DATASET_REPO='Wentap/PWM-ArtGen-test'
hf auth login
hf upload "$HF_MODEL_REPO" release-staging/model .
hf upload "$HF_DATASET_REPO" release-staging/dataset . --repo-type dataset
```

Use the README download commands in a clean directory and verify `SHA256SUMS`
after downloading it from each repository. For local baseline comparisons,
`scripts/summarize_results.py --compare_summary /path/to/baseline/summary.json`
reads adjacent per-dataset metrics; this is optional and not a public benchmark
prerequisite. Sampling may change metric values, so bitwise metric equality is
not the reproducibility criterion.

## Published HF assets

Uploaded and verified on 2026-09-25. Remote file sizes and LFS SHA-256 values
match the staged artifacts.

- Model: [Wentap/PWM-ArtGen](https://huggingface.co/Wentap/PWM-ArtGen/commit/a5412910ff9c1443f0117b1f81f043bc2d250481), revision `a5412910ff9c1443f0117b1f81f043bc2d250481`.
- Dataset: [Wentap/PWM-ArtGen-test](https://huggingface.co/datasets/Wentap/PWM-ArtGen-test/commit/2927d3d0c2f17566dbf8970c28c4b9de28f1be81), revision `2927d3d0c2f17566dbf8970c28c4b9de28f1be81`.
