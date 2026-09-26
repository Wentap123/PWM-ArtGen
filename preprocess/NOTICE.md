# Single-image preprocessing

SAM mask grouping is adapted from [OmniPart](https://huggingface.co/spaces/omnipart/OmniPart), copyright (c) 2025
VAST-AI-Research and contributors. Its [MIT license](LICENSE-OMNIPART) is included.
Only mask processing is included; the original application and Detectron2
visualization dependency are omitted.

Segmentation models are external dependencies. Follow the official environment
and checkpoint instructions for the backend you select:

- **SAM**: [installation](https://github.com/facebookresearch/segment-anything#installation)
  and [checkpoints](https://github.com/facebookresearch/segment-anything#model-checkpoints).
  The original PWM path uses ViT-H.
- **SAM3** (default): [installation and checkpoint access](https://github.com/facebookresearch/sam3#installation).
  Use the SAM3 image model and `sam3.pt`, not a SAM2 or SAM3.1 checkpoint.
- **RMBG**: [model setup and terms](https://huggingface.co/briaai/RMBG-2.0).
  Download a trusted local model directory; loading it executes its model code.
  Inputs with a non-opaque alpha channel can omit RMBG.

Install the optional photo dependencies from [requirements.txt](../requirements.txt)
(see [setup](../docs/setup.md)) in the selected backend environment after
following its upstream setup. Follow the selected segmentation backend's Python and PyTorch requirements. Model weights are not redistributed here; their upstream terms apply.

The SAM3 adapter follows the local PWM text-prompt experiment: graph-derived
door/drawer prompts, confidence fallback, then a base-rooted instance graph.
Graph prompts, mask normalization, and soft observation masks come from PWM.
Handles are represented within their parent parts, not independent nodes.
