# Data sources and preparation

Run commands from the repository root. The prepared PWM observations and original
mesh packages serve different roles; downloading original meshes does not create
the rendered/masked inputs used by PWM inference.

| Resource | Source | Purpose |
| --- | --- | --- |
| PM original annotations and meshes | [Singapo pm.zip](https://aspis.cmpt.sfu.ca/projects/singapo/data/pm.zip), derived from [PartNet-Mobility](https://sapien.ucsd.edu/browse) | Retrieval library and PM GT |
| ACD original annotations and meshes | [Singapo acd_test.zip](https://aspis.cmpt.sfu.ca/projects/singapo/data/acd_test.zip), [ACD/S2O](https://huggingface.co/datasets/3dlg-hcvc/s2o) | ACD GT |
| Prepared PM/ACD observations | [Wentap/PWM-ArtGen-test](https://huggingface.co/datasets/Wentap/PWM-ArtGen-test) | Two-view inference inputs |

Follow original providers' terms and attribution. Prepared archives contain
observations, masks, bboxes, GT part graphs, view mappings and test manifests;
they do not contain GT meshes, motion target frames, or the retrieval library.
End users download these prepared inputs rather than rerendering or reexporting.

## Download original packages

```bash
mkdir -p downloads data/raw/pm data/raw/acd
curl -fL --retry 3 -C - -o downloads/pm.zip \
  https://aspis.cmpt.sfu.ca/projects/singapo/data/pm.zip
curl -fL --retry 3 -C - -o downloads/acd_test.zip \
  https://aspis.cmpt.sfu.ca/projects/singapo/data/acd_test.zip
unzip -tq downloads/pm.zip
unzip -tq downloads/acd_test.zip
unzip -q downloads/pm.zip -d data/raw/pm
unzip -q downloads/acd_test.zip -d data/raw/acd
```

The database root must directly contain PM category directories; the ACD GT root
must directly contain `abo-data/` and `hssd-data/`. Adjust roots if a downloaded
archive adds a wrapper directory. Wait for extraction to finish before conversion.

Reference archive SHA-256 values:

```text
8a9beb2ec29c67b195a31c832b53aaf351c4935bbaf72893d53c897443aca8b5  pm.zip
6f42dd99034189313eaa046d894d7f3e1ff4233cda565b9bf5c3235b69bab818  acd_test.zip
```

## Convert annotations and rebuild the PM index

```bash
"$RETRIEVAL_PYTHON" scripts/prepare_database.py --database_root data/raw/pm --dry-run
"$RETRIEVAL_PYTHON" scripts/prepare_database.py --database_root data/raw/pm
"$RETRIEVAL_PYTHON" scripts/convert_acd_pwm.py --gt_root data/raw/acd \
  --test_ids data/benchmarks/acd/test_ids.json --dry-run
"$RETRIEVAL_PYTHON" scripts/convert_acd_pwm.py --gt_root data/raw/acd \
  --test_ids data/benchmarks/acd/test_ids.json
```

Inspect `data/raw/pm/pwm_database_report.json` and
`data/raw/acd/pwm_gt_report.json`; require no failures. Conversion preserves
original `object.json` and mesh files, adding `object_pwm.json`. Handles merge
into parent mesh references; prismatic handles become drawers, and shelves
outside Dishwasher map to doors. ACD also maps category names. ACD conversion
validates all selected objects before writing and requires no segmentation files.
Identical generated files are skipped; differing files require `--overwrite`.

PM conversion produces `pwm_hash_filtered.json`, recomputing topology hashes
with NetworkX 3.4.2 after conversion. The reference candidate allowlist gives
567 converted objects and 563 indexed objects. **75 of the 77 PM test IDs also
occur in this library**: this preserves the existing protocol and is not a
held-out retrieval-library experiment. `--reference_hashbook` overrides that
allowlist. Six planar PM AABBs retain zero thickness; negative dimensions and
boxes degenerate to a line or point are rejected. Geometry is not altered.

Both PM and ACD evaluation must use `--database_root data/raw/pm --hashbook
 data/raw/pm/pwm_hash_filtered.json`. ACD is only ground truth, not a second
retrieval library. Explicitly selecting this index avoids the bundled legacy
index fallback intended for historical prepared databases.

## Prepared benchmark format

```text
pm/ or acd/
  dataset.json
  test_ids.json
  checksums.json
  <object_key>/
    graph.json
    views.json
    view_0/
    view_1/
```

PM keys are `<category>/<id>`; ACD keys are `<source>/<category>/<id>`.
PM source views 00/01 map to views 0/1. ACD source view 10 maps to view 0;
view 1 is the highest-scoring other complete view using 0.25 silhouette IoU +
0.75 mean part-mask IoU after centering and aspect-preserving scaling.
`views.json` records the mapping and scores. This is a 2D frontal similarity
measure and does not use predictions or evaluation metrics.

The fixed split contains 77 PM and 134 ACD objects.
See [benchmark details](benchmark.md) for legacy layouts and maintainer export
commands, and [release preparation](release.md) before publishing archives.
