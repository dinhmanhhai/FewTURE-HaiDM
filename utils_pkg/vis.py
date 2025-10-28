from typing import List, Tuple, Optional

import os
import numpy as np
import torch


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _to_numpy_img(img: torch.Tensor) -> np.ndarray:
    # img: [C,H,W] in [0,1] or [0,255]
    x = img.detach().cpu().float()
    if x.max() <= 1.0:
        x = x * 255.0
    x = x.clamp(0, 255).byte().permute(1, 2, 0).numpy()
    return x


def overlay_heatmap(image: torch.Tensor, heatmap: torch.Tensor, alpha: float = 0.4, cmap: str = 'jet'):
    import cv2
    img = _to_numpy_img(image)
    h = heatmap.detach().cpu().float()
    if h.dim() == 3:
        h = h.squeeze(0)
    # upsample to image size
    h = torch.nn.functional.interpolate(h.unsqueeze(0).unsqueeze(0), size=(img.shape[0], img.shape[1]),
                                        mode='bilinear', align_corners=False)[0, 0]
    h = (h - h.min()) / (h.max() - h.min() + 1e-6)
    h_np = (h.numpy() * 255.0).astype(np.uint8)
    if cmap == 'jet':
        hm_color = cv2.applyColorMap(h_np, cv2.COLORMAP_JET)
    else:
        hm_color = cv2.applyColorMap(h_np, cv2.COLORMAP_VIRIDIS)
    overlay = cv2.addWeighted(img, 1.0, hm_color, alpha, 0)
    return overlay  # np.ndarray HxWx3 uint8


def draw_boxes(image: torch.Tensor, boxes: List[Tuple[int, int, int, int]], color: Tuple[int, int, int] = (0, 255, 0),
               thickness: int = 2):
    import cv2
    img = _to_numpy_img(image)
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    return img


def make_grid(images: List[torch.Tensor], nrow: int = 3):
    from torchvision.utils import make_grid as tv_make_grid
    if len(images) == 0:
        return None
    grid = tv_make_grid(torch.stack(images, dim=0), nrow=nrow)
    return _to_numpy_img(grid)


