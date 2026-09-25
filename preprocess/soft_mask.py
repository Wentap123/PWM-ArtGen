"""Soft observation masks from the original PWM single-image inference."""
import numpy as np
import torch
import torch.nn.functional as F

def _ensure_odd(k: int) -> int:
    k = int(max(1, k))
    return k if k % 2 == 1 else k + 1

def make_soft_mask(
    hw3,                       # (H,W,C)，numpy 或 torch，值在 {0,1} 或 [0,1]
    kernel: int = 5,           # 平滑核(奇数)，0/1=不平滑
    strength: float = 0.7,     # 0=硬边, 1=全软
    out_dtype=None,            # 返回 dtype（numpy/torch 都支持）；None 时沿用输入浮点，否则 float32
    use_gauss: bool = True,    # True=高斯，False=均值
    gauss_sigma: float = 1.0,  # 高斯 sigma
    do_close: bool = False,    # 是否做闭运算(膨胀后腐蚀)
    close_kernel: int = 3      # 闭运算核(奇数)
):
    # ---- 预处理 & 类型统一 ----
    is_numpy = isinstance(hw3, np.ndarray)
    if is_numpy:
        x = torch.from_numpy(hw3)
        in_dtype = hw3.dtype
    else:
        x = hw3
        in_dtype = hw3.dtype

    assert x.ndim == 3, f"expect (H,W,C), got {tuple(x.shape)}"
    H, W, C = x.shape

    x = x.to(torch.float32)
    # 若是 0/255 类型，归一到 [0,1]
    if x.max() > 1.0:
        x = x / 255.0
    x = x.clamp_(0, 1)

    # (H,W,C) -> (1,C,H,W)
    m = x.permute(2, 0, 1).unsqueeze(0)

    # ---- 闭运算（填小裂缝） ----
    if do_close and close_kernel and close_kernel > 1:
        ck = _ensure_odd(close_kernel)
        pad = ck // 2
        m = F.max_pool2d(m, kernel_size=ck, stride=1, padding=pad)                # 膨胀
        m = 1.0 - F.max_pool2d(1.0 - m, kernel_size=ck, stride=1, padding=pad)    # 腐蚀

    # ---- 平滑 ----
    if kernel and kernel > 1:
        k = _ensure_odd(kernel)
        pad = k // 2
        if use_gauss:
            r = torch.arange(-pad, pad + 1, device=m.device, dtype=m.dtype)
            g1 = torch.exp(-0.5 * (r / (gauss_sigma + 1e-6))**2)
            g1 = g1 / g1.sum()
            kx = g1.view(1, 1, 1, k).repeat(C, 1, 1, 1)  # (C,1,1,k)
            ky = g1.view(1, 1, k, 1).repeat(C, 1, 1, 1)  # (C,1,k,1)
            y = F.pad(m, (pad, pad, 0, 0), mode="reflect")
            y = F.conv2d(y, kx, groups=C)
            y = F.pad(y, (0, 0, pad, pad), mode="reflect")
            m_soft = F.conv2d(y, ky, groups=C)
        else:
            w = torch.ones((C, 1, k, k), device=m.device, dtype=m.dtype) / (k * k)
            y = F.pad(m, (pad, pad, pad, pad), mode="reflect")
            m_soft = F.conv2d(y, w, groups=C)
        m_soft = m_soft.clamp(0, 1)
    else:
        m_soft = m

    # ---- 硬/软插值 ----
    m_hard = (m >= 0.5).to(m.dtype)
    s = float(max(0.0, min(1.0, strength)))
    out = ((1.0 - s) * m_hard + s * m_soft).clamp(0, 1)

    # (1,C,H,W) -> (H,W,C)
    out = out.squeeze(0).permute(1, 2, 0)

    # ---- 按输入类型返回 ----
    if out_dtype is not None:
        # 指定的 dtype 优先
        if is_numpy:
            out = out.to(torch.float32) if out_dtype == np.float32 else out.to(torch.float32)
        else:
            out = out.to(out_dtype)
    else:
        # 沿用输入“语义”：numpy 就返回 numpy；torch 返回 torch（浮点）
        if is_numpy:
            pass
        else:
            out = out.to(in_dtype if out.dtype.is_floating_point else torch.float32)

    if is_numpy:
        return out.cpu().numpy()
    return out
