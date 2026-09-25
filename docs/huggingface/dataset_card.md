---
language:
- en
tags:
- pwm
- articulated-objects
- benchmark
---
# PWM-ArtGen Benchmarks

Code: https://github.com/Wentap123/PWM-ArtGen.
Dataset repository: `Wentap/PWM-ArtGen-test`.

`pwm_pm_test.zip` extracts to `pm/` (77 objects, 154 object-views).
`pwm_acd_test.zip` extracts to `acd/` (134 objects, 268 object-views).
Archives contain observation RGB, masks, bboxes, GT part graphs, view mappings,
test IDs and member checksums. They exclude meshes, motion target frames and the
retrieval library. This benchmark supplies GT structure; it is distinct from
single-photo inference with automatically predicted masks and graphs.

PM views 00/01 map to views 0/1. ACD view 10 maps to view 0; view 1 maximizes
0.25 silhouette IoU + 0.75 mean part-mask IoU over other complete views, after
centering and aspect-preserving scaling. See `views.json` for each source mapping.


Sources: [PartNet-Mobility](https://sapien.ucsd.edu/browse),
[ACD/S2O](https://huggingface.co/datasets/3dlg-hcvc/s2o), and
[Singapo](https://github.com/3dlg-hcvc/singapo). Download original mesh packages
separately: [PM](https://aspis.cmpt.sfu.ca/projects/singapo/data/pm.zip),
[ACD](https://aspis.cmpt.sfu.ca/projects/singapo/data/acd_test.zip).
Prepared PWM observations are not interchangeable with those mesh packages.

Both test sets retrieve from converted PM. PM GT is converted PM; ACD GT is
converted ACD. The reference PM library includes 75 PM test IDs, so it is not a
held-out retrieval-library protocol. Follow repository documentation to rebuild
`object_pwm.json` and the PM index, run inference, and evaluate.

Original dataset terms apply; no additional license grant is implied.
Project citation details have not yet been specified. Check `SHA256SUMS` for archive
integrity and retain member checksums when extracting.

## Download

```bash
hf download Wentap/PWM-ArtGen-test pwm_pm_test.zip pwm_acd_test.zip SHA256SUMS --repo-type dataset --local-dir downloads
```
