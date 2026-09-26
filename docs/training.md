# Training

Commands run from the repository root. See [setup](setup.md) and [data preparation](data.md).


Training expects the existing PartNet-style layout:

```text
data_root/{train,test}/<category>/<object_id>/
  *_joints_info.json
  joint_<id>/
    RGB/view_idx_XX/{frame_*,edited_frame_*}.png
    RGB/view_idx_XX/info/meta.json
    Mask/view_idx_XX/frame_*.png
```

Both original and edited images are used for training. Keep their corresponding masks and metadata. The final recipe uses frames 0 and 2, with the `test` split used for validation.

Data rendering and training-image editing are not included.

For inference, download the prepared PM/ACD test archives from
[Hugging Face](https://huggingface.co/datasets/Wentap/PWM-ArtGen-test) and follow
[the inference commands](../README.md#inference). For an ordinary input photo,
follow the [single-image guide](single_image.md).

## Training recipe

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_PROCESSES=4 bash scripts/train.sh \
  --data_root /path/to/data \
  --vae_path /path/to/sdxl_vae \
  --ckpt /path/to/checkpoints/
```

The launcher contains the final training configuration. Append arguments to override it, or use `--help` to list options. Set `NUM_PROCESSES=1` for one GPU. Use a trailing slash for checkpoint directories; resume with `--resume /path/to/checkpoints/train.latest.pt`.

Training does not load DROID cotrained blocks by default. To initialize from an external checkpoint, explicitly pass `--pretrained_blocks_path /path/to/checkpoint.pt`.

Logging is optional: install the optional logging dependencies in `requirements.txt` and add `--use_wandb --wandb_project pwm --wandb_run final`. SwanLab is used when installed, otherwise W&B. Logging is disabled by default.

