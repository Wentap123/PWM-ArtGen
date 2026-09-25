---
language:
- en
tags:
- pwm
- articulated-objects
- image-to-3d
---
# PWM-ArtGen

Code and instructions: https://github.com/Wentap123/PWM-ArtGen.
Model repository: `Wentap/PWM-ArtGen`.

`pwm_final.pt` is the unchanged final 200,000-step PWM checkpoint. PWM predicts
part geometry and motion-conditioned images; the code retrieves meshes from a
separate PM database. The checkpoint alone does not include meshes, SDXL VAE,
DINOv2, photo segmentation models, or input data.

Use the repository's pinned PWM environment and pass
`--ckpt checkpoints/pwm_final.pt --vae_path /path/to/sdxl-vae` to
`scripts/infer_pm.sh`, `scripts/infer_acd.sh`, or `demo/infer_image.sh`.
See the code README for download and benchmark commands.

Reference size: 3,559,406,910 bytes. SHA-256:
`c9be8ab16b3eb1f978bd1b95a95f1f2292af575848684a0473c23af854a1ef5b`.
The file contains training state and is not a stripped inference export.

Supported categories: StorageFurniture, Table, Refrigerator, Oven, Microwave,
WashingMachine, Dishwasher. Frontal observations are preferred; unusual views
and imperfect masks/graphs can reduce quality. The PM retrieval library overlaps
75 PM test IDs; results do not measure a held-out retrieval-library setting.

Project license and paper/citation details have not yet been specified.
External models and datasets retain their own terms.

## Download

```bash
hf download Wentap/PWM-ArtGen pwm_final.pt SHA256SUMS --local-dir checkpoints
```
