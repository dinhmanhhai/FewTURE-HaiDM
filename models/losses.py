import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassBalancedWeights:
    """
    Tính trọng số Class-Balanced theo Effective Number of Samples.
    counts: list/1D tensor số mẫu theo lớp (theo chỉ mục lớp đang dùng trong episode).
    beta: hệ số làm mịn (0.9–0.9999). Trả về tensor trọng số có kích thước [num_classes].
    """
    def __init__(self, counts, beta: float = 0.9995):
        if isinstance(counts, list):
            counts = torch.tensor(counts, dtype=torch.float64)
        elif not torch.is_tensor(counts):
            counts = torch.tensor(counts, dtype=torch.float64)
        self.counts = counts
        self.beta = torch.tensor(float(beta), dtype=torch.float64)

    def get_weights(self) -> torch.Tensor:
        counts = self.counts.clamp_min(1.0)
        effective_num = 1.0 - torch.pow(self.beta, counts)
        weights = (1.0 - self.beta) / (effective_num + 1e-8)
        # Chuẩn hóa để tổng ~ num_classes (thói quen phổ biến khi đưa vào weight của CE)
        weights = weights / weights.sum() * weights.numel()
        return weights.to(torch.float64)


class FocalLoss(nn.Module):
    """
    Focal Loss trên logits: CE(weight=alpha) * (1 - pt)^gamma
    - gamma: độ tập trung vào mẫu khó
    - alpha: tensor [C] hoặc None; có thể đặt bằng CB-weights normalize
    - reduction: 'mean' | 'sum'
    """
    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None, reduction: str = 'mean'):
        super().__init__()
        self.gamma = float(gamma)
        self.register_buffer('alpha', alpha if alpha is not None else None)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce)
        loss = ((1.0 - pt) ** self.gamma) * ce
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss


class CBFocalLoss(nn.Module):
    """
    Class-Balanced Focal Loss: CE với weight theo Effective Number kết hợp modulating (1-pt)^gamma.
    - counts phải tương ứng các lớp trong logits (thứ tự class index 0..C-1).
    - beta ~ 0.9995 mặc định; gamma=2.
    """
    def __init__(self, counts, beta: float = 0.9995, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        weights = ClassBalancedWeights(counts, beta).get_weights()
        self.register_buffer('alpha', weights)
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce)
        loss = ((1.0 - pt) ** self.gamma) * ce
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss


def build_query_loss(loss_name: str,
                     num_classes: int,
                     class_counts: list | torch.Tensor | None = None,
                     beta_cb: float = 0.9995,
                     gamma_focal: float = 2.0,
                     alpha_focal: torch.Tensor | None = None,
                     label_smoothing: float = 0.0):
    """
    Helper dựng loss cho truy vấn theo tham số CLI.
    Hỗ trợ: 'ce', 'focal', 'cb', 'cb_focal'.
    Trả về callable (logits, targets) -> loss.
    """
    loss_name = (loss_name or 'ce').lower()

    # CE với label smoothing sử dụng F.cross_entropy trực tiếp
    def ce_loss(logits, targets):
        return F.cross_entropy(logits, targets, label_smoothing=label_smoothing)

    if loss_name == 'ce':
        return ce_loss

    if loss_name == 'focal':
        alpha = None
        if alpha_focal is not None:
            alpha = alpha_focal
        return FocalLoss(gamma=gamma_focal, alpha=alpha)

    if loss_name == 'cb':
        assert class_counts is not None, "class_counts phải được cung cấp cho CB loss"
        weights = ClassBalancedWeights(class_counts, beta_cb).get_weights().to(torch.float64)

        def cb_ce_loss(logits, targets):
            return F.cross_entropy(logits.to(torch.float64), targets, weight=weights.to(logits.device), label_smoothing=label_smoothing)

        return cb_ce_loss

    if loss_name in ['cb_focal', 'cb-focal']:
        assert class_counts is not None, "class_counts phải được cung cấp cho CB-Focal"
        return CBFocalLoss(class_counts, beta=beta_cb, gamma=gamma_focal)

    raise ValueError(f"Unknown query loss: {loss_name}")


