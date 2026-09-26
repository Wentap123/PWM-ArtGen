---
language:
- en
tags:
- pwm
- articulated-objects
- benchmark
---
# PWM-ArtGen-test

Test inputs for [PWM-ArtGen: Part World Model for Articulated Object Generation](https://github.com/Wentap123/PWM-ArtGen).
Model weights: [Wentap/PWM-ArtGen](https://huggingface.co/Wentap/PWM-ArtGen).

| Archive | Extracted directory | Objects | Views |
| --- | --- | --- | --- |
| `pwm_pm_test.zip` | `pm/` | 77 | 154 |
| `pwm_acd_test.zip` | `acd/` | 135 | 270 |

Each object has two inputs, `view_0` and `view_1`, with RGB images, part masks,
bounding boxes, and a GT part graph. `views.json` contains the sample records
used by the inference loader. Each archive includes `test_ids.json`,
`dataset.json`, and per-file checksums in `checksums.json`.
Meshes and the retrieval library are downloaded separately.

## Download

```bash
hf download Wentap/PWM-ArtGen-test pwm_pm_test.zip pwm_acd_test.zip SHA256SUMS \
  --repo-type dataset --local-dir downloads
(cd downloads && sha256sum -c SHA256SUMS)
mkdir -p data/benchmarks
unzip downloads/pwm_pm_test.zip -d data/benchmarks
unzip downloads/pwm_acd_test.zip -d data/benchmarks
```

Use the extracted directories with the inference and evaluation commands in the
[code repository](https://github.com/Wentap123/PWM-ArtGen). The inference loader
reads each dataset's included `test_ids.json`.

## Data sources

The original datasets are [PartNet-Mobility](https://sapien.ucsd.edu/browse) and
[ACD/S2O](https://huggingface.co/datasets/3dlg-hcvc/s2o).
The annotation and mesh packages used for retrieval and evaluation are provided by
[SINGAPO](https://github.com/3dlg-hcvc/singapo):
[PM package](https://aspis.cmpt.sfu.ca/projects/singapo/data/pm.zip) and
[ACD test package](https://aspis.cmpt.sfu.ca/projects/singapo/data/acd_test.zip).
These mesh packages are separate from the prepared image inputs in this repository.
Original dataset terms apply.

Both test sets retrieve from the processed PM library. Evaluation uses PM ground
truth for PM and ACD ground truth for ACD. These test inputs provide GT part
structure; single-image inference uses its own segmentation and graph preparation.
The reference PM retrieval library includes 75 PM test objects.

## Citation

```bibtex
@inproceedings{zheng2026pwm,
  title={PWM-ArtGen: Part World Model for Articulated Object Generation},
  author={Zheng, Wentao and Wu, Ancong},
  booktitle={European Conference on Computer Vision},
  pages={251--267},
  year={2026},
  organization={Springer}
}
```
