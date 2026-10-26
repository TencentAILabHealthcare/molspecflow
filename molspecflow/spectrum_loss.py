# -*- coding: utf-8 -*-
"""Binned spectrum loss used by the flow matcher during training.

This module is not a benchmark evaluation implementation.
"""
from __future__ import annotations

import math
from typing import Optional

import torch

# Training loss defaults
MZ_MAX = 1005.0
BIN_RES = 0.01
EPS = 1e-12
NORM_EPS = 1e-6


def num_bins(mz_max: float = MZ_MAX, bin_res: float = BIN_RES) -> int:
    """Number of bins = len(arange(bin_res, mz_max + bin_res, bin_res))."""
    return int(round(mz_max / bin_res))


def make_bin_edges(
    mz_max: float = MZ_MAX,
    bin_res: float = BIN_RES,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Bin lower edges: arange(bin_res, mz_max+bin_res, bin_res)."""
    return torch.arange(bin_res, mz_max + bin_res, bin_res, device=device, dtype=dtype)


# PyTorch training loss
def bin_peaks_torch(
    mz: torch.Tensor,
    intensity: torch.Tensor,
    valid: torch.Tensor,
    bin_edges: torch.Tensor,
    num_bins_val: int,
    mz_max: float = MZ_MAX,
    use_sqrt: bool = False,
) -> torch.Tensor:
    """Scatter-add peak intensities into binned vectors. Returns [B, N_bins]."""
    B, P = mz.shape
    finite = torch.isfinite(mz) & torch.isfinite(intensity)
    mz = torch.nan_to_num(mz, nan=0.0, posinf=mz_max, neginf=0.0)
    intensity = torch.nan_to_num(intensity, nan=0.0, posinf=0.0, neginf=0.0)
    ints = intensity.clamp_min(0.0)
    if use_sqrt:
        ints = torch.sqrt(ints)
    keep = valid & finite & (mz > 0.0) & (mz < mz_max) & (ints > 0.0)
    idx = torch.searchsorted(bin_edges, mz.clamp_min(0.0), right=True)
    idx = idx.clamp(max=num_bins_val - 1)
    idx = idx * keep.long()
    weighted = ints * keep.float()
    out = torch.zeros(B, num_bins_val, device=mz.device, dtype=mz.dtype)
    out.scatter_add_(1, idx, weighted)
    return out


def soft_bin_peaks_torch(
    mz: torch.Tensor,
    intensity: torch.Tensor,
    valid: torch.Tensor,
    bin_edges: torch.Tensor,
    num_bins_val: int,
    mz_max: float = MZ_MAX,
    sigma_da: float = 0.05,
    use_sqrt: bool = False,
) -> torch.Tensor:
    """Differentiable soft binning: Gaussian spread over ±4σ bins around each peak.

    Each peak's intensity is distributed across nearby bins (weights sum to 1 per peak),
    so ``∂loss/∂m/z`` is non-zero inside the kernel support.
    """
    sigma_da = max(float(sigma_da), 1e-6)
    bin_res = float(bin_edges[1] - bin_edges[0])
    half_w = max(int(math.ceil(4.0 * sigma_da / bin_res)), 1)
    offsets = torch.arange(-half_w, half_w + 1, device=mz.device, dtype=torch.long)

    finite = torch.isfinite(mz) & torch.isfinite(intensity)
    mz = torch.nan_to_num(mz, nan=0.0, posinf=mz_max, neginf=0.0)
    intensity = torch.nan_to_num(intensity, nan=0.0, posinf=0.0, neginf=0.0)
    ints = intensity.clamp_min(0.0)
    if use_sqrt:
        ints = torch.sqrt(ints)
    keep = valid & finite & (mz > 0.0) & (mz < mz_max) & (ints > 0.0)
    weighted_i = ints * keep.float()

    idx_center = torch.searchsorted(bin_edges, mz.clamp_min(0.0), right=True)
    idx = idx_center.unsqueeze(-1) + offsets.view(1, 1, -1)
    idx = idx.clamp(0, num_bins_val - 1)

    centers = bin_edges + bin_res * 0.5
    bin_center = centers[idx]
    d = mz.unsqueeze(-1) - bin_center
    w = torch.exp(-0.5 * (d / sigma_da) ** 2) * keep.unsqueeze(-1).float()
    w = w / w.sum(dim=-1, keepdim=True).clamp_min(EPS)

    contrib = weighted_i.unsqueeze(-1) * w
    out = torch.zeros(mz.size(0), num_bins_val, device=mz.device, dtype=mz.dtype)
    out.scatter_add_(1, idx.reshape(mz.size(0), -1), contrib.reshape(mz.size(0), -1))
    return out


def precursor_bin_torch(
    prec_ion_mz: torch.Tensor,
    bin_edges: torch.Tensor,
    mz_max: float = MZ_MAX,
) -> torch.Tensor:
    """Per-sample precursor bin index; -1 = skip removal."""
    valid = (prec_ion_mz > 0.0) & (prec_ion_mz < mz_max) & torch.isfinite(prec_ion_mz)
    idx = torch.searchsorted(bin_edges, prec_ion_mz.clamp_min(0.0).unsqueeze(-1), right=True).squeeze(-1)
    return torch.where(valid, idx, torch.full_like(idx, -1))


def drop_precursor_bins(binned: torch.Tensor, prec_bin: torch.Tensor) -> torch.Tensor:
    """Zero precursor bin per batch row. ``prec_bin`` -1 = no-op."""
    out = binned.clone()
    B = out.size(0)
    batch_idx = torch.arange(B, device=out.device)
    mask = prec_bin >= 0
    if mask.any():
        safe_prec_bin = prec_bin.clamp(min=0, max=out.size(1) - 1)
        out[batch_idx[mask], safe_prec_bin[mask]] = 0.0
    return out


def binned_cosine_torch(
    pred_binned: torch.Tensor,
    true_binned: torch.Tensor,
) -> torch.Tensor:
    """L2-normalized cosine per row. Returns (B,) in [0, 1]."""
    dot = (pred_binned * true_binned).sum(dim=1)
    pn = pred_binned.norm(dim=1)
    tn = true_binned.norm(dim=1)
    valid = torch.isfinite(pn) & torch.isfinite(tn) & (pn > NORM_EPS) & (tn > NORM_EPS)
    denom = pn.clamp_min(NORM_EPS) * tn.clamp_min(NORM_EPS)
    cos = torch.nan_to_num(dot / denom, nan=0.0, posinf=0.0, neginf=0.0)
    cos = torch.where(valid, cos, torch.zeros_like(cos))
    return cos.clamp(0.0, 1.0)


def binned_spectrum_cosine_loss(
    pred_peaks: torch.Tensor,
    true_peaks: torch.Tensor,
    true_valid: torch.Tensor,
    prec_ion_mz: Optional[torch.Tensor] = None,
    mz_max: float = MZ_MAX,
    bin_res: float = BIN_RES,
    remove_precursor: bool = True,
    use_sqrt: bool = False,
    bin_edges: Optional[torch.Tensor] = None,
    soft_sigma_da: Optional[float] = None,
) -> torch.Tensor:
    """
    ``1 - cos_sim`` on binned spectra.

    Training: set ``soft_sigma_da > 0`` for pred soft-binning (m/z gradients).
    Hard mode: ``soft_sigma_da`` None or <= 0 uses ``searchsorted`` hard bins.

    Args:
        pred_peaks: (B, P, 2) absolute Da m/z + intensity.
        true_peaks: (B, P, 2) ground truth.
        true_valid: (B, P) bool, valid GT peaks.
        prec_ion_mz: (B,) precursor *ion* m/z (parent + adduct); optional.
        use_sqrt: if True, apply sqrt to intensities before binning (``cos_sim_sqrt``).
        bin_edges: optional cached [N_bins] tensor on device.

    Returns:
        (B,) loss = 1 - cosine.
    """
    pm = pred_peaks[..., 0]
    pi = pred_peaks[..., 1]
    tm = true_peaks[..., 0]
    ti = true_peaks[..., 1]
    tv = true_valid if true_valid.dtype == torch.bool else true_valid.bool()

    n_bins = num_bins(mz_max, bin_res)
    if bin_edges is None:
        bin_edges = make_bin_edges(mz_max, bin_res, device=pm.device, dtype=pm.dtype)
    elif bin_edges.device != pm.device:
        bin_edges = bin_edges.to(device=pm.device, dtype=pm.dtype)

    pred_valid = torch.ones_like(tv, dtype=torch.bool)
    use_soft = soft_sigma_da is not None and float(soft_sigma_da) > 0.0
    if use_soft:
        pred_b = soft_bin_peaks_torch(
            pm, pi, pred_valid, bin_edges, n_bins,
            mz_max=mz_max, sigma_da=float(soft_sigma_da), use_sqrt=use_sqrt,
        )
        true_b = bin_peaks_torch(
            tm, ti, tv, bin_edges, n_bins, mz_max=mz_max, use_sqrt=use_sqrt,
        )
    else:
        pred_b = bin_peaks_torch(
            pm, pi, pred_valid, bin_edges, n_bins, mz_max=mz_max, use_sqrt=use_sqrt,
        )
        true_b = bin_peaks_torch(
            tm, ti, tv, bin_edges, n_bins, mz_max=mz_max, use_sqrt=use_sqrt,
        )

    if remove_precursor and prec_ion_mz is not None:
        pb = precursor_bin_torch(prec_ion_mz, bin_edges, mz_max=mz_max)
        pred_b = drop_precursor_bins(pred_b, pb)
        true_b = drop_precursor_bins(true_b, pb)

    cos = binned_cosine_torch(pred_b, true_b)
    return 1.0 - cos
