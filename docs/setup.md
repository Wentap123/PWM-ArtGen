# Environment setup

Use separate environments for PWM and retrieval/evaluation. The single root
`requirements.txt` lists core pins and commented optional dependencies.
Comments document optional stacks; `pip install -r requirements.txt` installs
only PWM core dependencies. Commands below run from the repository root.

## PWM

```bash
conda create -n pwm python=3.10 -y
conda activate pwm
python -m pip install -r requirements.txt
export PYTHON_BIN="$(command -v python)"
```

The reference environment used Python 3.10.18 and PyTorch 2.2.2+cu121.
Use a compatible CUDA driver. Supply the SDXL VAE through `--vae_path` and
`checkpoints/pwm_final.pt` through `--ckpt`. DINOv2 loads
`dinov2_vitb14_reg` from `facebookresearch/dinov2` via torch.hub; first use
requires access to the repository and its pretrained weights, or a populated
Torch Hub cache. The reference weight is `dinov2_vitb14_reg4_pretrain.pth`.

## Retrieval and evaluation (optional for inference)

```bash
conda create -n pwm-retrieval python=3.10 -y
conda activate pwm-retrieval
python -m pip install numpy==2.2.6 scipy==1.15.3 networkx==3.4.2 \
  numpy-quaternion==2024.0.12 trimesh==4.8.3
python -m pip install torch==2.3.1 torchvision==0.18.1 \
  --index-url https://download.pytorch.org/whl/cu118
# Build PyTorch3D against this environment's CUDA/PyTorch stack:
python -m pip install ninja iopath
python -m pip install --no-build-isolation \
  'git+https://github.com/facebookresearch/pytorch3d.git@V0.7.8'
export RETRIEVAL_PYTHON="$(command -v python)"
```

Building PyTorch3D requires a compatible compiler and CUDA toolkit. Consult
[upstream installation instructions](https://github.com/facebookresearch/pytorch3d/blob/V0.7.8/INSTALL.md)
for platform requirements. These pins record the reference evaluation stack;
a fresh installation on every hardware/platform combination has not been tested.
NetworkX must remain **3.4.2** for matching topology hashes. Do not install the
root core requirements into this environment.

## Photo preprocessing and logging (optional)

Follow upstream [SAM3](https://github.com/facebookresearch/sam3#installation)
or [SAM](https://github.com/facebookresearch/segment-anything#installation)
setup in its own environment, then install the photo utilities listed in the
optional block:

```bash
python -m pip install numpy Pillow opencv-python openai transformers timm kornia einops
```

Set `SAM3_PYTHON` or `SAM_PYTHON` to that interpreter. Upstream segmentation
Python/CUDA requirements take precedence over the PWM environment. See
[single-image instructions](single_image.md) for checkpoints, RMBG and API use.
Benchmark inference uses prepared graphs/masks and needs no segmentation or API.

For training logging, in the PWM environment:

```bash
"$PYTHON_BIN" -m pip install swanlab==0.6.8 wandb==0.19.1
```

Logging is disabled by default; add `--use_wandb` to enable it. SwanLab is
preferred when installed.

## Hugging Face utility CLI

Install the CLI separately to avoid changing the model environment's pinned
Transformers dependencies:

```bash
python3 -m venv .venv-hf
.venv-hf/bin/python -m pip install -U huggingface_hub
export PATH="$PWD/.venv-hf/bin:$PATH"
hf --help
```

See the [official CLI guide](https://huggingface.co/docs/huggingface_hub/guides/cli).
`hf auth login` is needed for uploads or restricted resources, not public downloads.
