import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_avg_pool(
    feat: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """
    Average pool features within the (soft/binary) mask region.

    Args:
        feat: (N, C, H, W)
        mask: (N, H, W) or (N, 1, H, W)
    Returns:
        pooled: (N, C)
    """
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    mask = mask.to(feat.dtype)
    num = (feat * mask).sum(dim=(2, 3))  # (N, C)
    den = mask.sum(dim=(2, 3)).clamp_min(eps)  # (N, 1)
    return num / den


def tmc_loss(
    visual: torch.Tensor, text: torch.Tensor, temperature: float = 0.07
) -> torch.Tensor:
    """
    Text–Mask Contrastive loss (symmetric InfoNCE) between visual region features and text embeddings.

    Args:
        visual: (N, D)
        text:   (N, D)
    Returns:
        scalar loss
    """
    assert visual.shape == text.shape, f"Shape mismatch: {visual.shape} vs {text.shape}"
    v = F.normalize(visual.float(), dim=-1)
    t = F.normalize(text.float(), dim=-1)
    logits = (v @ t.t()) / max(temperature, 1e-6)
    labels = torch.arange(v.size(0), device=visual.device)
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_i + loss_t)


class _Sobel(nn.Module):
    def __init__(self):
        super().__init__()
        # 3x3 Sobel kernels
        kx = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)
        ky = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32)
        self.register_buffer("kx", kx.view(1, 1, 3, 3))
        self.register_buffer("ky", ky.view(1, 1, 3, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, 1, H, W), return gradient magnitude in (N, 1, H, W)
        gx = F.conv2d(x, self.kx, padding=1)
        gy = F.conv2d(x, self.ky, padding=1)
        return torch.sqrt(gx * gx + gy * gy + 1e-6)


_sobel = _Sobel()


def boundary_loss(
    pred_logits: torch.Tensor, gt_masks: torch.Tensor, reduction: str = "mean"
) -> torch.Tensor:
    """
    Boundary-aware loss using Sobel edge magnitude.

    Args:
        pred_logits: (N, H, W) raw logits
        gt_masks:    (N, H, W) binary {0,1}
    """
    if pred_logits.dim() == 3:
        pred_logits = pred_logits.unsqueeze(1)
    if gt_masks.dim() == 3:
        gt_masks = gt_masks.unsqueeze(1)

    # convert to prob and float masks
    pred_prob = pred_logits.float().sigmoid()
    gt = gt_masks.float()

    # compute edges in float32 for numerical stability
    edges_pred = _sobel(pred_prob)
    edges_gt = _sobel(gt)

    # normalize edge magnitudes to [0,1] approximately per-sample (optional but stabilizes)
    def _norm(e):
        e = e - e.amin(dim=(2, 3), keepdim=True)
        e = e / (e.amax(dim=(2, 3), keepdim=True)[0].clamp_min(1e-6))
        return e

    edges_pred = _norm(edges_pred)
    edges_gt = _norm(edges_gt)

    loss = F.l1_loss(edges_pred, edges_gt, reduction=reduction)
    return loss
