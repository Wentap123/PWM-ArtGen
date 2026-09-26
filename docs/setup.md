# Environment setup

Use one `pwm` environment for training, inference, mesh retrieval, and evaluation.
Run the following commands from the repository root:

```bash
conda create -n pwm python=3.10 -y
conda activate pwm
python -m pip install torch==2.2.2 torchvision==0.17.2 \
  --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
```

`requirements.txt` includes retrieval dependencies and the Hugging Face CLI.
All launchers use the active environment's Python by default. NetworkX is pinned
to 3.4.2 so conversion and retrieval produce matching topology hashes.

## Evaluation

Install PyTorch3D in the same environment after PyTorch:

```bash
conda activate pwm
python -m pip install ninja iopath
python -m pip install --no-build-isolation \
  'git+https://github.com/facebookresearch/pytorch3d.git@V0.7.8'
```

PyTorch3D 0.7.8 supports PyTorch 2.2.2. A CUDA build requires a compiler and
CUDA toolkit matching the PyTorch installation (CUDA 11.8 for the commands
above). Set `CUDA_HOME` to your toolkit directory if it is not detected.
See the [official build instructions](https://github.com/facebookresearch/pytorch3d/blob/V0.7.8/INSTALL.md).
PyTorch3D is only needed for metric computation.

## Segmentation

For single-image preprocessing, follow the official installation and checkpoint
instructions for [SAM3](https://github.com/facebookresearch/sam3#installation)
or [SAM](https://github.com/facebookresearch/segment-anything#installation).
Set `SAM3_PYTHON` or `SAM_PYTHON` if segmentation uses a different environment.
See the [single-image guide](single_image.md) for RMBG and graph prediction.
PM/ACD test-set inference uses the provided masks and graphs.

## Model downloads

Use `hf download` from the active `pwm` environment. See the README for the
PWM-ArtGen checkpoint, test inputs, and SDXL VAE download commands.
DINOv2 loads `dinov2_vitb14_reg` from `facebookresearch/dinov2` via Torch Hub;
the first run downloads its code and weights unless already cached.

## Optional logging

```bash
conda activate pwm
python -m pip install swanlab==0.6.8 wandb==0.19.1
```

Add `--use_wandb` to the training command to enable logging. SwanLab is used
when installed; otherwise the logger uses W&B.
