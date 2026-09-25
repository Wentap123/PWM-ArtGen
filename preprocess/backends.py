"""Lazy SAM/SAM3 adapters; install each upstream in its own environment."""
from contextlib import nullcontext
import numpy as np
from PIL import Image


def prepare_image(path, rmbg_model, device):
    import torch
    from torchvision import transforms
    image = Image.open(path)
    if 'A' not in image.getbands() or image.getchannel('A').getextrema()[0] == 255:
        if not rmbg_model:
            raise ValueError('RGB images require --rmbg_model; transparent RGBA inputs can omit it.')
        from transformers import AutoModelForImageSegmentation
        model = AutoModelForImageSegmentation.from_pretrained(
            str(rmbg_model), trust_remote_code=True, local_files_only=True).to(device).eval()
        image = image.convert('RGB')
        transform = transforms.Compose([transforms.Resize((1024, 1024)), transforms.ToTensor(),
                                        transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
        with torch.inference_mode():
            alpha = model(transform(image).unsqueeze(0).to(device))[-1].sigmoid().cpu()[0].squeeze()
        image.putalpha(transforms.ToPILImage()(alpha).resize(image.size))
        del model
        if str(device).startswith('cuda'):
            torch.cuda.empty_cache()
    image = image.convert('RGBA')
    # Preserve the original SAM path's 518-square pad followed by 512 resize.
    width, height = image.size
    scale = 518 / max(width, height)
    resized = image.resize((max(1, int(width*scale)), max(1, int(height*scale))), Image.Resampling.LANCZOS)
    square = Image.new('RGBA', (518, 518), (255, 255, 255, 0))
    square.paste(resized, ((518-resized.width)//2, (518-resized.height)//2), resized)
    return square.resize((512, 512), Image.Resampling.BILINEAR)


def white_rgb(rgba):
    background = Image.new('RGBA', rgba.size, (255, 255, 255, 255))
    return Image.alpha_composite(background, rgba).convert('RGB')


def region_visualization(groups):
    import cv2
    rgb = np.full((*groups.shape, 3), 255, dtype=np.uint8)
    for i in np.unique(groups):
        if i >= 0:
            rgb[groups == i] = [(int(i)*50+80)%256, (int(i)*120+40)%256, (int(i)*180+20)%256]
    for i in np.unique(groups):
        if i >= 0:
            ys, xs = np.where(groups == i)
            # A real region pixel nearest the centroid, also for concave shapes.
            j = np.argmin((xs-xs.mean())**2 + (ys-ys.mean())**2)
            xy = (int(xs[j]), int(ys[j]))
            cv2.putText(rgb, str(i), xy, cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 4, cv2.LINE_AA)
            cv2.putText(rgb, str(i), xy, cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 0, 0), 1, cv2.LINE_AA)
    return Image.fromarray(rgb)


def segment_sam(rgba, checkpoint, device, size_threshold=1000):
    # https://github.com/facebookresearch/segment-anything#installation
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    from .sam_regions import get_sam_mask
    import torch
    model = sam_model_registry['vit_h'](checkpoint=str(checkpoint)).to(device).eval()
    with torch.inference_mode():
        return get_sam_mask(np.asarray(white_rgb(rgba)), SamAutomaticMaskGenerator(model),
                            visual=None, rgba_image=rgba, size_threshold=size_threshold)


def segment_sam3(rgba, graph, checkpoint, device, confidence=.5):
    # https://github.com/facebookresearch/sam3#installation
    import torch
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    model = build_sam3_image_model(checkpoint_path=str(checkpoint), load_from_HF=False, device=device)
    present = {n['name'] for n in graph['diffuse_tree'][1:]}
    prompts = [name for name in ('door', 'drawer') if name in present] or ['door', 'drawer']
    thresholds = [confidence] + [v for v in (.2, .16) if v < confidence]
    result = {}
    amp = (torch.autocast('cuda', dtype=torch.bfloat16)
           if str(device).startswith('cuda') and torch.cuda.is_bf16_supported() else nullcontext())
    with torch.inference_mode(), amp:
        for prompt in prompts:
            for threshold in thresholds:
                processor = Sam3Processor(model, confidence_threshold=threshold, device=device)
                state = processor.set_image(white_rgb(rgba))
                processor.reset_all_prompts(state)
                state = processor.set_text_prompt(state=state, prompt=prompt)
                masks = state.get('masks')
                if masks is not None and masks.numel():
                    result[prompt] = masks.detach().float().cpu().numpy().reshape(-1, rgba.height, rgba.width)
                    break
                if prompt not in present:
                    break
    return result
