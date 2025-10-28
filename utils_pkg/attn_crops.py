import math
from typing import List, Tuple, Literal

import torch
import torch.nn.functional as F


def _min_max_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x_min = x.amin(dim=(-2, -1), keepdim=True)
    x_max = x.amax(dim=(-2, -1), keepdim=True)
    return (x - x_min) / (x_max - x_min + eps)


@torch.no_grad()
def vit_attention_rollout(model, images: torch.Tensor, last_k: int = 4, alpha: float = 0.5) -> torch.Tensor:
    """
    Attention rollout cho ViT: tổng hợp attention của K block cuối.
    Trả: heatmap kích thước [B, 1, H_p, W_p] theo lưới patch.
    Yêu cầu model trả về all tokens (return_all_tokens=True).
    """
    attn_list = []
    handles = []
    try:
        # Đăng ký hook trên tất cả Attention module trong ViT blocks
        for blk in getattr(model, 'blocks', []):
            attn_mod = getattr(blk, 'attn', None)
            if attn_mod is None:
                continue
            def _hook(_, __, output):
                # output: (x, attn)
                attn_list.append(output[1].detach())
            handles.append(attn_mod.register_forward_hook(_hook))

        _ = model(images)  # forward để thu thập attention

        if len(attn_list) == 0:
            raise RuntimeError('No attention captured for ViT.')

        # Lấy K block cuối
        attn_tail = attn_list[-last_k:]
        # Trung bình theo heads, thêm residual và rollout
        rollout = None
        for attn in attn_tail:
            # attn shape [B, nH, N, N]
            A = attn.mean(dim=1)
            N = A.shape[-1]
            I = torch.eye(N, device=A.device).unsqueeze(0).expand_as(A)
            A_tilde = alpha * I + (1 - alpha) * A
            rollout = A_tilde if rollout is None else torch.bmm(A_tilde, rollout)

        # Lấy tầm quan trọng theo patch từ CLS (cột 0), bỏ CLS → [B, N-1]
        cls_attn = rollout[:, 0, 1:]
        # Lưới patch
        num_patches = cls_attn.shape[-1]
        side = int(math.sqrt(num_patches))
        heat = cls_attn.reshape(images.shape[0], 1, side, side)
        heat = _min_max_norm(heat)
        return heat
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def swin_token_saliency(model, images: torch.Tensor) -> torch.Tensor:
    """
    Proxy heatmap cho Swin: dùng độ lớn kích hoạt token ở stage cuối (không cần gradient/attn stitching).
    Trả: heatmap [B, 1, H_p, W_p] theo lưới patch cuối.
    """
    tokens = model(images)  # expect return_all_tokens=True: [B, 1+L, C] (không có CLS ở Swin, nhưng code trả concat)
    # Bỏ token pooled đầu tiên nếu tồn tại
    if tokens.dim() == 3 and tokens.shape[1] > 1:
        patch_tokens = tokens[:, 1:, :]
    else:
        patch_tokens = tokens
    B, N, C = patch_tokens.shape
    side = int(math.sqrt(N))
    fmap = patch_tokens.reshape(B, side, side, C).permute(0, 3, 1, 2)  # [B, C, H_p, W_p]
    sal = fmap.pow(2).sum(dim=1, keepdim=True).sqrt()
    sal = _min_max_norm(sal)
    return sal


def _topk_peak_boxes(heat: torch.Tensor, k: int, box_hw: Tuple[int, int], iou_thr: float = 0.4):
    B, _, H, W = heat.shape
    boxes_all = []
    for b in range(B):
        h = heat[b, 0]
        # local maxima via max-pool
        pooled = F.max_pool2d(h.unsqueeze(0).unsqueeze(0), kernel_size=3, stride=1, padding=1)[0, 0]
        peaks = torch.where((h == pooled))
        scores = h[peaks]
        if scores.numel() == 0:
            boxes_all.append([])
            continue
        # sort top-k
        vals, idxs = torch.topk(scores, k=min(k, scores.numel()))
        sel_y = peaks[0][idxs]
        sel_x = peaks[1][idxs]
        # make boxes centered at peaks
        bw, bh = box_hw
        boxes = []
        scrs = []
        for y, x, s in zip(sel_y.tolist(), sel_x.tolist(), vals.tolist()):
            x1 = max(0, int(x - bw // 2))
            y1 = max(0, int(y - bh // 2))
            x2 = min(W - 1, int(x1 + bw - 1))
            y2 = min(H - 1, int(y1 + bh - 1))
            boxes.append((x1, y1, x2, y2))
            scrs.append(float(s))
        # NMS
        kept_boxes, _ = nms_boxes(boxes, scrs, iou_thr=iou_thr)
        boxes_all.append(kept_boxes)
    return boxes_all


def heatmap_to_boxes(heat: torch.Tensor,
                     top_percent: float = 0.07,
                     min_box: int = 8,
                     num_local_crops: int = 1,
                     iou_thr: float = 0.4) -> List[List[Tuple[int, int, int, int]]]:
    """
    Hai chế độ:
    - num_local_crops<=1: ngưỡng percentile → 1 box tổng quát.
    - num_local_crops>1: lấy top-k peaks + NMS tạo nhiều box nhỏ.
    """
    B, _, H, W = heat.shape
    if num_local_crops > 1:
        # box kích thước ~ 20% chiều heatmap (có thể tinh chỉnh theo crop_size ở ngoài)
        bw = max(min_box, int(0.2 * W))
        bh = max(min_box, int(0.2 * H))
        return _topk_peak_boxes(heat, k=num_local_crops, box_hw=(bw, bh), iou_thr=iou_thr)


def nms_boxes(boxes: List[Tuple[int, int, int, int]], scores: List[float], iou_thr: float = 0.4):
    if len(boxes) == 0:
        return []
    boxes_t = torch.tensor(boxes, dtype=torch.float)
    scores_t = torch.tensor(scores, dtype=torch.float)
    x1, y1, x2, y2 = boxes_t[:, 0], boxes_t[:, 1], boxes_t[:, 2], boxes_t[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores_t.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)
        if order.numel() == 1:
            break
        xx1 = torch.maximum(x1[i], x1[order[1:]])
        yy1 = torch.maximum(y1[i], y1[order[1:]])
        xx2 = torch.minimum(x2[i], x2[order[1:]])
        yy2 = torch.minimum(y2[i], y2[order[1:]])
        w = (xx2 - xx1 + 1).clamp(min=0)
        h = (yy2 - yy1 + 1).clamp(min=0)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = torch.where(ovr <= iou_thr)[0]
        order = order[inds + 1]
    return [boxes[k] for k in keep], [scores[k] for k in keep]


def extract_crops_from_boxes(images: torch.Tensor,
                             boxes: List[List[Tuple[int, int, int, int]]],
                             upscale_to: int,
                             heatmap_scale: Tuple[int, int]) -> List[List[torch.Tensor]]:
    """
    Cắt crop từ ảnh gốc theo toạ độ trên heatmap (H_p,W_p) → ánh xạ về (H,W) bằng scale.
    Trả: danh sách per-image các crop tensor [C, upscale_to, upscale_to].
    """
    B, C, H, W = images.shape
    Hp, Wp = heatmap_scale
    sx, sy = W / Wp, H / Hp
    out = []
    for b in range(B):
        crops_b = []
        for (x1, y1, x2, y2) in boxes[b]:
            X1 = int(x1 * sx)
            Y1 = int(y1 * sy)
            X2 = int((x2 + 1) * sx)
            Y2 = int((y2 + 1) * sy)
            X1, Y1 = max(0, X1), max(0, Y1)
            X2, Y2 = min(W, X2), min(H, Y2)
            crop = images[b:b + 1, :, Y1:Y2, X1:X2]
            crop = F.interpolate(crop, size=(upscale_to, upscale_to), mode='bilinear', align_corners=False)[0]
            crops_b.append(crop)
        out.append(crops_b)
    return out


def merge_probs(p_global: torch.Tensor, p_locals: List[torch.Tensor], w_local: float = 0.65) -> torch.Tensor:
    if len(p_locals) == 0:
        return p_global
    return w_local * torch.stack(p_locals).mean(0) + (1.0 - w_local) * p_global


