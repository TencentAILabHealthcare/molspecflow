# -*- coding: utf-8 -*-
# meteor_flow_model_arch.py
# 包含模型架构 (nn.Module)
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, List, Dict
from .formula_utils import VECTOR_SIZE, get_element_mono_masses_tensor
# =============================================================================
# PART 1: Core Building Blocks
# =============================================================================
class AdaLayerNorm(nn.Module):
    def __init__(self, d_model: int, cond_dim: int):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.ada_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * d_model)
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if cond.dim() == 3:
            cond = cond[:, 0]
        adaptive_params = self.ada_mlp(cond)
        scale, shift = torch.chunk(adaptive_params, 2, dim=-1)
        return scale.unsqueeze(1) * self.layer_norm(x) + shift.unsqueeze(1)


class LearnableGaussianBasis(nn.Module):
    def __init__(self, num_basis: int, min_val: float, max_val: float, d_model: int):
        super().__init__()
        assert max_val > min_val
        initial_centers = torch.linspace(min_val, max_val, num_basis)
        self.centers = nn.Parameter(initial_centers)
        initial_sigma = (max_val - min_val) / (num_basis - 1) if num_basis > 1 else 1.0
        self.log_sigma = nn.Parameter(torch.full((num_basis,), math.log(initial_sigma)))
        self.projection = nn.Linear(num_basis, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sigma = torch.exp(self.log_sigma).clamp_min(1e-6)
        expanded_x = torch.exp(-((x - self.centers) ** 2) / (2 * sigma ** 2))
        return self.projection(expanded_x)


class ICLRSinusoidalMzEncoding(nn.Module):
    """ICLR 2023 multi-scale sinusoidal m/z encoding (Eq. 4–5).

    Wavelengths log-spaced from ``lam_min`` to ``lam_max`` Da; output dim = ``embed_dim``
    (half sin, half cos).  Operates on absolute m/z in Daltons.
    """

    def __init__(
        self,
        embed_dim: int = 512,
        lam_min: float = 10 ** -2.5,
        lam_max: float = 10 ** 3.3,
    ):
        super().__init__()
        assert embed_dim >= 2 and embed_dim % 2 == 0
        n_freq = embed_dim // 2
        i = torch.arange(n_freq, dtype=torch.float64)
        denom = max(embed_dim - 2, 1)
        wavelengths = lam_min * (lam_max / lam_min) ** (2.0 * i / denom)
        self.register_buffer('wavelengths', wavelengths.float(), persistent=False)
        self.out_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., 1)
        mz = x.squeeze(-1).unsqueeze(-1).to(dtype=self.wavelengths.dtype)
        angles = (2.0 * math.pi) * mz / self.wavelengths
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1).to(dtype=x.dtype)


class FourierMzEncoding(nn.Module):
    """MassSpecGym DeepSets+Fourier style: ``num_wavelengths`` log-spaced λ → sin/cos (6000-d).

    Reference: MSG supplement §3.2 (3000 wavelengths → 6000 features).
    """

    def __init__(
        self,
        num_wavelengths: int = 3000,
        lam_min: float = 10 ** -2.5,
        lam_max: float = 10 ** 3.3,
    ):
        super().__init__()
        assert num_wavelengths >= 1
        i = torch.arange(num_wavelengths, dtype=torch.float64)
        if num_wavelengths == 1:
            wavelengths = torch.tensor([lam_min], dtype=torch.float64)
        else:
            wavelengths = lam_min * (lam_max / lam_min) ** (i / (num_wavelengths - 1))
        self.register_buffer('wavelengths', wavelengths.float(), persistent=False)
        self.out_dim = num_wavelengths * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mz = x.squeeze(-1).unsqueeze(-1).to(dtype=self.wavelengths.dtype)
        angles = (2.0 * math.pi) * mz / self.wavelengths
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1).to(dtype=x.dtype)


class HybridMzEncoder(nn.Module):
    """Coarse learnable Gaussian + ICLR sinusoidal + MSG Fourier (6000 sin/cos)."""

    def __init__(
        self,
        d_model: int,
        num_basis: int,
        mz_max_val: float,
        *,
        use_coarse_gaussian: bool = True,
        sinusoidal_dim: int = 512,
        fourier_num_wavelengths: int = 3000,
        wavelength_min_da: float = 10 ** -2.5,
        wavelength_max_da: float = 10 ** 3.3,
    ):
        super().__init__()
        self.use_coarse_gaussian = use_coarse_gaussian
        if use_coarse_gaussian:
            self.gaussian = LearnableGaussianBasis(num_basis, 0.0, mz_max_val, d_model)
        self.sinusoidal = ICLRSinusoidalMzEncoding(
            embed_dim=sinusoidal_dim,
            lam_min=wavelength_min_da,
            lam_max=wavelength_max_da,
        )
        self.fourier = FourierMzEncoding(
            num_wavelengths=fourier_num_wavelengths,
            lam_min=wavelength_min_da,
            lam_max=wavelength_max_da,
        )
        self.sin_proj = nn.Linear(sinusoidal_dim, d_model)
        self.fourier_proj = nn.Linear(self.fourier.out_dim, d_model)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.sin_proj(self.sinusoidal(x)) + self.fourier_proj(self.fourier(x))
        if self.use_coarse_gaussian:
            out = out + self.gaussian(x)
        return self.out_norm(out)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        if self.use_coarse_gaussian:
            remap_legacy_gaussian_state_dict(state_dict, prefix)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )


def remap_legacy_gaussian_state_dict(state_dict: Dict, prefix: str) -> None:
    """``LearnableGaussianBasis`` ckpt → ``HybridMzEncoder.gaussian``（就地 remap）。"""
    if f'{prefix}gaussian.centers' in state_dict:
        return
    for suffix in ('centers', 'log_sigma', 'projection.weight', 'projection.bias'):
        old_k = f'{prefix}{suffix}'
        new_k = f'{prefix}gaussian.{suffix}'
        if old_k in state_dict:
            state_dict[new_k] = state_dict.pop(old_k)


def build_unified_mass_encoder(
    config: Dict,
    d_model: int,
    value_max: float,
    num_basis: Optional[int] = None,
) -> nn.Module:
    """统一标量 Da 编码：粗高斯 + ICLR Sinusoidal + MSG Fourier(6000)。"""
    eff = dict(config)
    eff['mz_max_val'] = float(value_max)
    if num_basis is not None:
        eff['mz_num_basis'] = int(num_basis)
    # Checkpoints predating HybridMzEncoder have neither mode key and contain
    # only the legacy Gaussian parameters. Preserve that architecture unless
    # hybrid encoding is requested explicitly.
    mode = str(eff.get('mass_encoder_mode', eff.get('mz_encoder_mode', 'gaussian'))).lower()
    eff['mz_encoder_mode'] = mode
    if mode == 'gaussian':
        nb = int(num_basis if num_basis is not None else eff['mz_num_basis'])
        return LearnableGaussianBasis(nb, 0.0, float(value_max), d_model)
    lam_min = float(eff.get('mz_wavelength_min_da', 10 ** -2.5))
    lam_max = float(eff.get('mz_wavelength_max_da', 10 ** 3.3))
    return HybridMzEncoder(
        d_model=d_model,
        num_basis=int(num_basis if num_basis is not None else eff['mz_num_basis']),
        mz_max_val=float(value_max),
        use_coarse_gaussian=bool(eff.get('mz_use_coarse_gaussian', True)),
        sinusoidal_dim=int(eff.get('mz_sinusoidal_dim', 512)),
        fourier_num_wavelengths=int(eff.get('mz_fourier_num_wavelengths', 3000)),
        wavelength_min_da=lam_min,
        wavelength_max_da=lam_max,
    )


def build_mz_encoder(config: Dict, d_model: int) -> nn.Module:
    """Peak / precursor m/z encoder（同 unified mass encoder）。"""
    return build_unified_mass_encoder(config, d_model, float(config['mz_max_val']))


class TokenMassPeakCrossAttention(nn.Module):
    """每个谱峰 slot cross-attend SMILES token 质量特征（隐式 token mass → 峰 m/z 先验）。"""

    def __init__(self, d_spec: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_spec % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_spec // n_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(d_spec, d_spec)
        self.k_proj = nn.Linear(d_spec, d_spec)
        self.v_proj = nn.Linear(d_spec, d_spec)
        self.out_proj = nn.Linear(d_spec, d_spec)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x_peaks: torch.Tensor,
        token_mass_feat: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        # x_peaks: (B, P, D); token_mass_feat: (B, S, D); token_mask: (B, S) True=valid
        B, P, D = x_peaks.shape
        S = token_mass_feat.shape[1]
        q = self.q_proj(x_peaks).view(B, P, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(token_mass_feat).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(token_mass_feat).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if token_mask is not None:
            pad = ~token_mask.bool()
            attn_mask = pad.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(attn_mask, torch.finfo(attn.dtype).min)
        attn = torch.softmax(attn, dim=-1)
        if token_mask is not None:
            attn = attn.masked_fill(attn_mask, 0.0)
            attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, P, D)
        return self.out_proj(out)


class PeakTokenMassCrossAttention(nn.Module):
    """每个 SMILES token cross-attend 谱峰 m/z（spec→smiles：token mass ↔ 峰质量对齐）。"""

    def __init__(self, d_model: int, d_spec: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_spec, d_model)
        self.v_proj = nn.Linear(d_spec, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        token_mass_feat: torch.Tensor,
        mz_peak_feat: torch.Tensor,
        peak_mask: torch.Tensor,
        L: int,
    ) -> torch.Tensor:
        # token_mass_feat: (BL, S, d_model); mz_peak_feat: (B, P, d_spec)
        BL, S, D = token_mass_feat.shape
        B = BL // L
        P = mz_peak_feat.shape[1]
        mz_exp = mz_peak_feat.repeat_interleave(L, dim=0)
        q = self.q_proj(token_mass_feat).view(BL, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(mz_exp).view(BL, P, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(mz_exp).view(BL, P, self.n_heads, self.head_dim).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if peak_mask is not None:
            pm = peak_mask.repeat_interleave(L, dim=0)
            attn_mask = ~pm.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(attn_mask, torch.finfo(attn.dtype).min)
        attn = torch.softmax(attn, dim=-1)
        if peak_mask is not None:
            attn = attn.masked_fill(attn_mask, 0.0)
            attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(BL, S, D)
        return self.out_proj(out)


class FormulaMassHead(nn.Module):
    """Use the legacy m/z head as a proposal, then select a nearby integer formula."""

    def __init__(
        self,
        d_spec: int,
        formula_dim: int = VECTOR_SIZE,
        max_count: float = 64.0,
        candidate_k: int = 32,
        candidate_temperature_da: float = 5.0,
        scorer_dim: int = 64,
        mz_max: float = 1005.0,
    ):
        super().__init__()
        self.formula_dim = int(formula_dim)
        self.max_count = float(max_count)
        self.candidate_k = int(candidate_k)
        self.candidate_temperature_da = float(candidate_temperature_da)
        self.scorer_dim = int(scorer_dim)
        self.mz_max = float(mz_max)
        self.count_logits = nn.Linear(d_spec, self.formula_dim)
        nn.init.zeros_(self.count_logits.weight)
        # Start near the empty formula.  The legacy scalar head supplies the
        # initial mass scale, while integer candidates provide exact outputs.
        nn.init.constant_(self.count_logits.bias, -6.0)
        self.formula_query = nn.Linear(d_spec, self.scorer_dim)
        self.formula_candidate_proj = nn.Linear(
            self.formula_dim, self.scorer_dim, bias=False,
        )
        nn.init.zeros_(self.formula_query.weight)
        nn.init.zeros_(self.formula_query.bias)
        self.register_buffer(
            'element_masses',
            get_element_mono_masses_tensor(),
            persistent=False,
        )

    def _integer_candidates(
        self,
        x_spec: torch.Tensor,
        mass_proposal: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build exact integer formulas near the learned composition and m0.

        The base formula uses straight-through rounding.  For every element,
        three candidates adjust its count by the nearest residual-mass count
        and its two neighbours.  Candidate construction needs no molecule
        annotation and covers all configured elements.
        """
        soft_count = torch.sigmoid(self.count_logits(x_spec)) * self.max_count
        masses = self.element_masses.to(dtype=x_spec.dtype, device=x_spec.device)
        hard_count = soft_count.round().clamp(min=0.0, max=self.max_count)
        base_count = soft_count + (hard_count - soft_count).detach()
        base_mass = torch.matmul(base_count, masses)

        proposal = torch.nan_to_num(
            mass_proposal, nan=0.0, posinf=self.mz_max, neginf=0.0,
        ).clamp(min=0.0, max=self.mz_max)
        residual = proposal - base_mass
        delta_center = torch.round(residual.unsqueeze(-1) / masses)
        eye = torch.eye(
            self.formula_dim, dtype=x_spec.dtype, device=x_spec.device,
        ).view(1, 1, self.formula_dim, self.formula_dim)

        candidates = [base_count.unsqueeze(-2)]
        for offset in (-1.0, 0.0, 1.0):
            delta = delta_center + offset
            adjusted = base_count.unsqueeze(-2) + eye * delta.unsqueeze(-1)
            candidates.append(adjusted.clamp(min=0.0, max=self.max_count))
        candidate_counts = torch.cat(candidates, dim=-2)
        candidate_masses = torch.matmul(candidate_counts, masses)

        num_candidates = candidate_masses.size(-1)
        k = min(self.candidate_k, num_candidates)
        nearest = (candidate_masses - proposal.unsqueeze(-1)).abs().topk(
            k, dim=-1, largest=False,
        ).indices
        gather_counts = nearest.unsqueeze(-1).expand(
            *nearest.shape, self.formula_dim,
        )
        candidate_counts = torch.gather(
            candidate_counts, -2, gather_counts,
        )
        candidate_masses = torch.gather(candidate_masses, -1, nearest)
        return candidate_counts, candidate_masses

    def _select_formula(
        self,
        x_spec: torch.Tensor,
        mass_proposal: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        candidate_counts, candidate_masses = self._integer_candidates(
            x_spec, mass_proposal,
        )
        proposal = torch.nan_to_num(
            mass_proposal, nan=0.0, posinf=self.mz_max, neginf=0.0,
        ).clamp(min=0.0, max=self.mz_max)
        query = self.formula_query(x_spec).unsqueeze(-2)
        candidate_features = self.formula_candidate_proj(
            candidate_counts / max(self.max_count, 1.0),
        )
        learned_score = (
            query * candidate_features
        ).sum(dim=-1) / math.sqrt(float(self.scorer_dim))
        distance_score = -(
            candidate_masses - proposal.unsqueeze(-1)
        ).abs() / max(self.candidate_temperature_da, 1e-6)
        score = torch.nan_to_num(
            learned_score + distance_score,
            nan=-1e4, posinf=1e4, neginf=-1e4,
        )

        soft_choice = torch.softmax(score, dim=-1)
        hard_idx = score.argmax(dim=-1)
        hard_choice = F.one_hot(
            hard_idx, num_classes=score.size(-1),
        ).to(dtype=soft_choice.dtype)
        choice = soft_choice + (hard_choice - soft_choice).detach()
        selected_counts = (
            choice.unsqueeze(-1) * candidate_counts
        ).sum(dim=-2)
        selected_mass = (choice * candidate_masses).sum(dim=-1)
        return selected_counts, selected_mass

    def forward(
        self,
        x_spec: torch.Tensor,
        mass_proposal: torch.Tensor,
    ) -> torch.Tensor:
        # Forward values are exact masses of integer element-count vectors.
        _, selected_mass = self._select_formula(x_spec, mass_proposal)
        return selected_mass.clamp_min(0.0)

    def predict_formula(
        self,
        x_spec: torch.Tensor,
        mass_proposal: torch.Tensor,
    ) -> torch.Tensor:
        """Return the selected integer element counts (B, P, formula_dim)."""
        with torch.no_grad():
            selected_counts, _ = self._select_formula(x_spec, mass_proposal)
            return selected_counts.round().clamp(min=0.0).long()


class GridMzHead(nn.Module):
    """m/z = softmax(grid) @ centers + small residual（训练）；推理吸附后续再加。"""

    def __init__(
        self,
        d_spec: int,
        num_grid_bins: int = 512,
        mz_max: float = 1005.0,
        residual_max_da: float = 0.5,
    ):
        super().__init__()
        self.num_grid_bins = int(num_grid_bins)
        self.residual_max_da = float(residual_max_da)
        centers = torch.linspace(0.0, mz_max, self.num_grid_bins)
        self.register_buffer('grid_centers', centers, persistent=False)
        self.grid_logits = nn.Linear(d_spec, self.num_grid_bins)
        self.residual = nn.Linear(d_spec, 1)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(self, x_spec: torch.Tensor) -> torch.Tensor:
        logits = self.grid_logits(x_spec)
        w = torch.softmax(logits, dim=-1)
        mz = torch.matmul(w, self.grid_centers.to(dtype=x_spec.dtype))
        mz = mz + torch.tanh(self.residual(x_spec).squeeze(-1)) * self.residual_max_da
        return mz.clamp_min(0.0)


class FormulaEncoder(nn.Module):
    """
    支持多维输入：
    - formula_vec: [..., V] (如 [B, V] 或 [B, L, V])
    - formula_is_dummy: [...] (如 [B] 或 [B, L])
    输出: [..., D]
    """
    def __init__(
        self,
        formula_dim: int,
        cond_dim: int,
        num_basis: int = 32,
        min_count: float = 0.0,
        max_count: float = 64.0,
        pool: str = "attn",   
        dropout: float = 0.0,
    ):
        super().__init__()
        self.formula_dim = formula_dim
        self.cond_dim    = cond_dim
        self.pool        = pool

        self.basis = LearnableGaussianBasis(
            num_basis=num_basis,
            min_val=min_count,
            max_val=max_count,
            d_model=cond_dim,
        )

        if self.pool == "attn":
            self.attn = nn.Sequential(
                nn.Linear(cond_dim, cond_dim),
                nn.GELU(),
                nn.Linear(cond_dim, 1),
            )
        else:
            self.attn = None

        self.dropout = nn.Dropout(dropout)
        self.out_norm = nn.LayerNorm(cond_dim)

    def forward(
        self,
        formula_vec: torch.Tensor,                 # [..., V]
        formula_is_dummy: torch.Tensor = None,     # [...] (bool)
    ) -> torch.Tensor:
        
        # 1. 动态获取前导维度 (prefix shape)
        # 无论输入是 [B, V] 还是 [B, L, V]，V 必须匹配
        if formula_vec.shape[-1] != self.formula_dim:
            raise ValueError(f"Expected last dim {self.formula_dim}, got {formula_vec.shape[-1]}")
        
        prefix_shape = formula_vec.shape[:-1]  # 获取 [B] 或 [B, L]
        V = self.formula_dim
        
        # 2. 展平处理：将所有前导维度合并为一维 N
        x = formula_vec.reshape(-1, V).float() # [N, V]
        N = x.size(0)

        # Clamp 到 basis 支持范围
        x = x.clamp(min=self.basis.centers.min().item(), max=self.basis.centers.max().item())

        # 3. Gaussian basis 映射
        # x: [N, V] -> reshape为每个原子计数单独处理 -> [N*V, 1]
        x_flat = x.reshape(-1, 1)              # [N*V, 1]
        emb_flat = self.basis(x_flat)          # [N*V, D]
        emb = emb_flat.reshape(N, V, self.cond_dim)  # [N, V, D]
        emb = self.dropout(emb)

        # 4. 在 V 维度聚合
        if self.pool == "sum":
            pooled = emb.sum(dim=1)            # [N, D]
        elif self.pool == "mean":
            pooled = emb.mean(dim=1)           # [N, D]
        elif self.pool == "attn":
            w = self.attn(emb)                 # [N, V, 1]
            w = torch.softmax(w, dim=1)        # [N, V, 1]
            pooled = (emb * w).sum(dim=1)      # [N, D]
        else:
            raise ValueError(f"Unknown pool={self.pool}")

        pooled = self.out_norm(pooled)         # [N, D]

        # 5. 恢复原始形状 [..., D]
        out = pooled.reshape(*prefix_shape, self.cond_dim)

        # 6. 扩展后的 formula_is_dummy 处理
        if formula_is_dummy is not None:
            mask = formula_is_dummy.bool()
            
            # 策略：如果 mask 维度不够（比如只有 [B]），自动扩展到 [B, L]
            # 逻辑：只要 mask 的前几维能和 out 对上即可
            while mask.dim() < out.dim() - 1:
                mask = mask.unsqueeze(-1)
            
            # 此时 mask 可能是 [B, 1] 或 [B, L]
            # unsqueeze(-1) 是为了对齐 D 维度：[B, L, 1]
            mask_for_mul = mask.unsqueeze(-1)
            
            # 使用 ~mask 进行置零 (True 表示是 dummy，需要变成 0)
            out = out * (~mask_for_mul).float()

        return out



class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_len: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._set_cos_sin_cache(max_len)

    def _set_cos_sin_cache(self, seq_len: int):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(0).unsqueeze(1)
    sin = sin.unsqueeze(0).unsqueeze(1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class RelativeMzBias(nn.Module):
    """
    基于 m/z 的相对位置偏置：
    - 计算 |mz_i - mz_j|，按 **precursor m/z（每样本固定）** 或全局 mz_ref_scale 归一化后分桶
    - 不再使用 batch 内 min–max（避免 batch 组成影响 attention 几何）
    """
    def __init__(self, num_heads: int, num_buckets: int = 64, mz_ref_scale: float = 1005.0):
        super().__init__()
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.bias = nn.Embedding(num_buckets, num_heads)
        self.register_buffer('mz_ref_scale', torch.tensor(float(mz_ref_scale)))

    def forward(
        self,
        mz_seq: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        ref_mz: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        mz_seq: [B,L] 绝对 Da m/z（谱峰序列）
        ref_mz: [B] precursor m/z；优先于 mz_ref_scale 作为归一化尺度
        key_padding_mask: [B,L]（True=有效，False=pad）
        Return: bias [B,H,L,L]
        """
        B, L = mz_seq.shape
        delta = (mz_seq.unsqueeze(2) - mz_seq.unsqueeze(1)).abs()  # [B,L,L]

        if ref_mz is not None:
            scale = ref_mz.to(mz_seq.device, dtype=mz_seq.dtype).view(B, 1, 1).clamp_min(1e-6)
        else:
            scale = self.mz_ref_scale.to(mz_seq.device, dtype=mz_seq.dtype).clamp_min(1e-6)
        delta_norm = (delta / scale).clamp(0.0, 1.0)

        bucket = (delta_norm * (self.num_buckets - 1)).floor().to(torch.long)  # [B,L,L]

        # 对 pad 的列/行置 0（若提供了 mask）
        if key_padding_mask is not None:
            keep = key_padding_mask.to(torch.bool)                 # True=有效
            pad_mask = (~keep).unsqueeze(1) | (~keep).unsqueeze(2) # 任一维 pad 即 True
            bucket = bucket.masked_fill(pad_mask, 0)

        bias = self.bias(bucket)  # [B,L,L,H]
        return bias.permute(0, 3, 1, 2)  # [B,H,L,L]


class SelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, dropout: float, max_len: int):
        super().__init__()
        assert d_model % n_heads == 0 and n_heads % n_kv_heads == 0
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_head_groups = n_heads // n_kv_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(dim=self.head_dim, max_len=max_len)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        attn_bias: Optional[torch.Tensor] = None,  # 新增：可以传入 [B,H,L,L] 的 bias
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rope(seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if self.n_head_groups > 1:
            k = k.repeat_interleave(self.n_head_groups, dim=1)
            v = v.repeat_interleave(self.n_head_groups, dim=1)

        # PyTorch 2.x: attn_mask 支持形状 [B,H,L,L] 的 float mask，直接当作 bias 加到 score 上
        attn_mask = attn_bias

        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.out_proj(attn_output)


class CrossAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, cond_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout,
            batch_first=True, kdim=cond_dim, vdim=cond_dim
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        context_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        attn_output, _ = self.attn(
            x, context, context, need_weights=False, key_padding_mask=context_mask
        )
        return attn_output


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t.unsqueeze(-1) * embeddings.unsqueeze(0)
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

# =============================================================================
# PART 2: Multi-Modal Fusion Architecture
# =============================================================================

class MeteorFlowTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_spec: int,
        n_heads: int,
        n_kv_heads: int,
        cond_dim: int,
        d_ff: int,
        dropout: float,
        max_seq_len: int,
        d_fp: Optional[int] = None,   # 新增：FP 的 hidden 维度
    ):
        super().__init__()

        if d_fp is None:
            d_fp = d_spec  # 默认 FP 通道用和谱图一样的维度

        # ---------------- SMILES ----------------
        self.smiles_intra_path_attn = SelfAttention(d_model, n_heads, n_kv_heads, dropout, max_len=max_seq_len)
        self.smiles_adaln_sa_intra = AdaLayerNorm(d_model, cond_dim)

        self.smiles_inter_path_attn = SelfAttention(d_model, n_heads, n_kv_heads, dropout, max_len=max_seq_len)
        self.smiles_adaln_sa_inter = AdaLayerNorm(d_model, cond_dim)

        self.smiles_ffn = SwiGLU(d_model, d_ff, dropout)
        self.smiles_adaln_ffn = AdaLayerNorm(d_model, cond_dim)

        # ---------------- SPEC ----------------
        self.spec_self_attn = SelfAttention(d_spec, n_heads, n_kv_heads, dropout, max_len=max_seq_len)
        self.spec_adaln_sa = AdaLayerNorm(d_spec, cond_dim)
        self.spec_ffn = SwiGLU(d_spec, d_ff, dropout)
        self.spec_adaln_ffn = AdaLayerNorm(d_spec, cond_dim)

        # ---------------- FP（新通道）----------------
        self.d_fp = d_fp
        self.fp_self_attn = SelfAttention(d_fp, n_heads, n_kv_heads, dropout, max_len=max_seq_len)
        self.fp_adaln_sa = AdaLayerNorm(d_fp, cond_dim)
        self.fp_ffn = SwiGLU(d_fp, d_ff, dropout)
        self.fp_adaln_ffn = AdaLayerNorm(d_fp, cond_dim)

        # ---------------- Cross-Attention ----------------
        # 原有：Spec -> Smiles， Smiles -> Spec
        self.cross_attn_spec_to_smiles = CrossAttention(d_model, n_heads, d_spec, dropout)
        self.smiles_adaln_cross = AdaLayerNorm(d_model, cond_dim)

        self.cross_attn_smiles_to_spec = CrossAttention(d_spec, n_heads, d_model, dropout)
        self.spec_adaln_cross = AdaLayerNorm(d_spec, cond_dim)

        # 新增：FP -> Smiles， FP -> Spec（FP 作为额外上下文）
        self.cross_attn_fp_to_smiles = CrossAttention(d_model, n_heads, d_fp, dropout)
        self.smiles_adaln_cross_fp = AdaLayerNorm(d_model, cond_dim)

        self.cross_attn_fp_to_spec = CrossAttention(d_spec, n_heads, d_fp, dropout)
        self.spec_adaln_cross_fp = AdaLayerNorm(d_spec, cond_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x_smiles: torch.Tensor,                 # (BL, S_len, d_model)
        x_spec: torch.Tensor,                   # (B,  P_len, d_spec)
        cond_vec: torch.Tensor,                 # (BL, cond_dim)
        L: int,
        smiles_mask: torch.Tensor,              # (BL, S_len) 目前没用 mask 到注意力里
        spec_mask: torch.Tensor,                # (B,  P_len)
        spec_rel_bias: Optional[torch.Tensor] = None,  # [B,H,P,P]
        x_fp: Optional[torch.Tensor] = None,    # (B, T_fp, d_fp)
        fp_mask: Optional[torch.Tensor] = None  # (B, T_fp)
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:

        BL, S_len, D_model = x_smiles.shape
        B, P_len, D_spec = x_spec.shape
        cond_B = cond_vec[::L]  # (B, cond_dim)

        # ---- 1a. SMILES Self-Attention (intra path) ----
        x_smiles_sa_in = self.smiles_adaln_sa_intra(x_smiles, cond_vec)
        sa_out_intra = self.smiles_intra_path_attn(x_smiles_sa_in, padding_mask=None)
        x_smiles = x_smiles + self.dropout(sa_out_intra)

        # ---- 1b. SMILES Self-Attention (inter path) ----
        path_summary = x_smiles.mean(dim=1)                      # (BL, D_model)
        path_summary_reshaped = path_summary.view(B, L, D_model) # (B, L, D_model)
        path_summary_adaln = self.smiles_adaln_sa_inter(
            path_summary_reshaped,
            cond_B
        )
        sa_out_inter = self.smiles_inter_path_attn(path_summary_adaln, padding_mask=None)
        x_smiles = x_smiles + self.dropout(sa_out_inter.view(BL, 1, D_model))

        # ---- 1c. Spectrum Self-Attention ----
        x_spec = x_spec + self.dropout(
            self.spec_self_attn(
                self.spec_adaln_sa(x_spec, cond_B),
                padding_mask=None,
                attn_bias=spec_rel_bias,
            )
        )

        # ---- 1d. FP Self-Attention（如果有）----
        if x_fp is not None:
            x_fp = x_fp + self.dropout(
                self.fp_self_attn(
                    self.fp_adaln_sa(x_fp, cond_B),
                    padding_mask=None,
                    attn_bias=None
                )
            )

        # ---- 2a. Spec-to-SMILES ----
        x_spec_expanded = x_spec.repeat_interleave(L, dim=0)  # (BL, P_len, d_spec)
        x_smiles = x_smiles + self.dropout(
            self.cross_attn_spec_to_smiles(
                self.smiles_adaln_cross(x_smiles, cond_vec),
                context=x_spec_expanded,
                context_mask=None
            )
        )

        # ---- 2b. FP-to-SMILES（如果有）----
        if x_fp is not None:
            x_fp_expanded = x_fp.repeat_interleave(L, dim=0)  # (BL, T_fp, d_fp)
            x_smiles = x_smiles + self.dropout(
                self.cross_attn_fp_to_smiles(
                    self.smiles_adaln_cross_fp(x_smiles, cond_vec),
                    context=x_fp_expanded,
                    context_mask=None
                )
            )

        # ---- 2c. SMILES-to-Spec ----
        smiles_kv_reshaped = x_smiles.view(B, L * S_len, D_model)
        x_spec = x_spec + self.dropout(
            self.cross_attn_smiles_to_spec(
                self.spec_adaln_cross(x_spec, cond_B),
                context=smiles_kv_reshaped,
                context_mask=None
            )
        )

        # ---- 2d. FP-to-Spec（如果有）----
        if x_fp is not None:
            x_spec = x_spec + self.dropout(
                self.cross_attn_fp_to_spec(
                    self.spec_adaln_cross_fp(x_spec, cond_B),
                    context=x_fp,
                    context_mask=None
                )
            )

        # ---- 3. FFN ----
        x_smiles = x_smiles + self.dropout(
            self.smiles_ffn(self.smiles_adaln_ffn(x_smiles, cond_vec))
        )
        x_spec = x_spec + self.dropout(
            self.spec_ffn(self.spec_adaln_ffn(x_spec, cond_B))
        )

        if x_fp is not None:                                 
            x_fp = x_fp + self.dropout(
                self.fp_ffn(self.fp_adaln_ffn(x_fp, cond_B))
            )

        return x_smiles, x_spec, x_fp


class MeteorFlowModel(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        d_model = config['d_model']
        d_spec = config['d_spec']
        cond_dim = config['cond_dim']
        n_heads = config['n_heads']
        n_kv_heads = config.get('n_kv_heads', n_heads)

        # --- Embeddings ---
        self.token_embedding = nn.Embedding(
            config['vocab_size'], d_model, padding_idx=config['pad_token_id']
        )
        self.token_mass_embedding = build_unified_mass_encoder(
            config, d_model, value_max=500.0, num_basis=config['mass_num_basis'],
        )

        # 谱图编码：统一 hybrid mass encoder
        self.mz_encoder = build_unified_mass_encoder(config, d_spec, value_max=config['mz_max_val'])
        self.intensity_encoder = LearnableGaussianBasis(
            config['intensity_num_basis'], 0.0, 1.0, d_spec
        )
        # Preserve the pretrained Gaussian intensity path and add fine/coarse
        # periodic features as a zero-initialized residual.  This makes
        # checkpoint migration behavior-preserving at step zero.
        self.intensity_multiscale_encoder = HybridMzEncoder(
            d_model=d_spec,
            num_basis=config['intensity_num_basis'],
            mz_max_val=1.0,
            use_coarse_gaussian=False,
            sinusoidal_dim=int(config.get('intensity_sinusoidal_dim', 128)),
            fourier_num_wavelengths=int(
                config.get('intensity_fourier_num_wavelengths', 256)
            ),
            wavelength_min_da=float(
                config.get('intensity_wavelength_min', 1.0e-4)
            ),
            wavelength_max_da=float(
                config.get('intensity_wavelength_max', 1.0)
            ),
        )
        nn.init.zeros_(self.intensity_multiscale_encoder.sin_proj.weight)
        nn.init.zeros_(self.intensity_multiscale_encoder.sin_proj.bias)
        nn.init.zeros_(self.intensity_multiscale_encoder.fourier_proj.weight)
        nn.init.zeros_(self.intensity_multiscale_encoder.fourier_proj.bias)

        # --- Condition Embeddings ---
        self.total_mass_encoder = build_unified_mass_encoder(
            config, cond_dim,
            value_max=config.get('total_mass_max_val', 1005.0),
            num_basis=config.get('total_mass_num_basis', config['mz_num_basis']),
        )
        self.precursor_mz_encoder = build_unified_mass_encoder(
            config, cond_dim, value_max=config['mz_max_val'],
        )
        self.adduct_embedding = build_unified_mass_encoder(
            config, cond_dim, value_max=config['mz_max_val'],
        )
        self.charge_embedding = LearnableGaussianBasis(64, -10, 10, cond_dim)
        self.energy_embedding = LearnableGaussianBasis(128, 0.0, 1000.0, cond_dim)
        self.formula_dim = VECTOR_SIZE
        self.formula_encoder = FormulaEncoder(VECTOR_SIZE, cond_dim, 64)
        self.token_formula_encoder = FormulaEncoder(VECTOR_SIZE, d_model, 64)

        # --- Time Embeddings ---
        self.time_embed = SinusoidalTimeEmbedding(config['time_embed_dim'])
        self.time_to_cond_mlp = nn.Sequential(
            nn.Linear(config['time_embed_dim'], cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, cond_dim)
        )
        
        # ---------------- FP 通道配置 ----------------
        self.fp_enable: bool = bool(config.get('fp_enable', False))
        self.fp_input_dim: int = int(config.get('fp_input_dim', 2048))
        self.fp_token_len: int = int(config.get('fp_token_len', 50))
        self.d_fp: int = int(config.get('d_fp', d_spec))
        
        self.fp_enable = True
        if self.fp_enable:
            # 将 (B, 1024) 压成 (B, 100, d_fp)
            self.fp_compress = nn.Linear(
                self.fp_input_dim,
                self.fp_token_len * self.d_fp
            )
            self.fp_ln = nn.LayerNorm(self.d_fp)
            # dummy fp 特征，用于 dummy 样本
            self.dummy_fp_feature = nn.Parameter(torch.randn(1, 1, self.d_fp))

        # Dummy Features & Initial Guess
        self.dummy_smiles_feature = nn.Parameter(torch.randn(1, 1, d_model))
        self.dummy_spec_feature = nn.Parameter(torch.randn(1, 1, d_spec))
        self.smiles_mass_to_spec_proj = nn.Linear(d_model, d_spec)
        self.dummy_smiles_to_spec_feat = nn.Parameter(torch.randn(1, 1, d_spec))
        attn_heads = int(config.get('token_mass_peak_attn_heads', 4))
        while d_spec % attn_heads != 0 and attn_heads > 1:
            attn_heads -= 1
        self.token_mass_peak_attn = TokenMassPeakCrossAttention(
            d_spec, n_heads=max(attn_heads, 1), dropout=config.get('dropout', 0.1),
        )
        self.token_mass_peak_gate = nn.Parameter(torch.zeros(()))
        peak_tm_heads = int(config.get(
            'peak_token_mass_cross_attn_heads',
            config.get('peak_token_mass_attn_heads', attn_heads),
        ))
        while d_model % peak_tm_heads != 0 and peak_tm_heads > 1:
            peak_tm_heads -= 1
        self.peak_token_mass_cross_attn = PeakTokenMassCrossAttention(
            d_model, d_spec, n_heads=max(peak_tm_heads, 1), dropout=config.get('dropout', 0.1),
        )
        self.peak_token_mass_gate = nn.Parameter(torch.zeros(()))

        mz_head_mode = str(config.get('mz_head_mode', '')).lower()
        if not mz_head_mode:
            if config.get('mz_formula_head_enabled', True):
                mz_head_mode = 'formula'
            elif config.get('mz_grid_enabled', False):
                mz_head_mode = 'grid'
            else:
                mz_head_mode = 'linear'
        self.mz_head_mode = mz_head_mode

        if self.mz_head_mode == 'formula':
            self.formula_mass_head = FormulaMassHead(
                d_spec,
                formula_dim=int(config.get('formula_dim', VECTOR_SIZE)),
                max_count=float(config.get('formula_max_count', 64.0)),
                candidate_k=int(config.get('formula_candidate_k', 32)),
                candidate_temperature_da=float(
                    config.get('formula_candidate_temperature_da', 5.0)
                ),
                scorer_dim=int(config.get('formula_scorer_dim', 64)),
                mz_max=float(config.get('spectral_cos_mz_max', 1005.0)),
            )
        elif self.mz_head_mode == 'grid':
            self.grid_mz_head = GridMzHead(
                d_spec,
                num_grid_bins=int(config.get('mz_grid_num_bins', 512)),
                mz_max=float(config.get('mz_grid_max_da', config.get('spectral_cos_mz_max', 1005.0))),
                residual_max_da=float(config.get('mz_grid_residual_max_da', 0.5)),
            )

        # 相对 m/z 偏置模块（只在谱图 self-attn 中用）
        self.spec_mz_bias = RelativeMzBias(
            num_heads=n_heads,
            num_buckets=config.get('mz_bias_num_buckets', 64),
            mz_ref_scale=float(config.get('mz_bias_ref_scale', min(config.get('mz_max_val', 5000.0), 1005.0))),
        )

        # Fusion layers
        self.fusion_layers = nn.ModuleList([
            MeteorFlowTransformerBlock(
                d_model, d_spec, n_heads, n_kv_heads, cond_dim,
                config['d_ff'], config['dropout'],
                max_seq_len=config['max_pos_len'],
                d_fp=self.d_fp if self.fp_enable else None
            )
            for _ in range(config['num_layers'])
        ])

        # --- Output Heads ---
        self.final_smiles_ln = nn.LayerNorm(d_model)
        self.smiles_output_projection = nn.Linear(d_model, config['vocab_size'])
        self.final_spec_ln = nn.LayerNorm(d_spec)
        self.spec_output_projection = nn.Linear(d_spec, 2)
        self.final_fp_ln = nn.LayerNorm(self.d_fp * self.fp_token_len) if self.fp_enable else None
        self.fp_output_projection = nn.Linear(self.d_fp * self.fp_token_len, self.fp_input_dim) if self.fp_enable else None

    _LEGACY_GAUSSIAN_PREFIXES = (
        'token_mass_embedding.',
        'total_mass_encoder.',
        'adduct_embedding.',
        'mz_encoder.',
        'precursor_mz_encoder.',
    )

    def load_state_dict(self, state_dict, strict: bool = True):
        sd = dict(state_dict)
        for module_prefix in self._LEGACY_GAUSSIAN_PREFIXES:
            remap_legacy_gaussian_state_dict(sd, module_prefix)
        return super().load_state_dict(sd, strict=strict)

    def forward(
        self,
        S_t_smiles_tokens: torch.Tensor,
        S_t_token_masses: torch.Tensor,
        S_t_peaks: torch.Tensor,
        t: torch.Tensor,
        condition: Dict[str, torch.Tensor],
        smiles_mask: torch.Tensor,
        spec_mask: torch.Tensor,
        smiles_is_dummy: torch.Tensor,  # (B*L,)
        spec_is_dummy: torch.Tensor,    # (B,)
        S_t_fp: Optional[torch.Tensor] = None,   # (B, fp_input_dim) at time t
        fp_is_dummy: Optional[torch.Tensor] = None,  # (BL)
        S_t_token_formular: Optional[torch.Tensor] = None,  # (BL, V)
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        BL, S_len = S_t_smiles_tokens.shape
        B, P_len, _ = S_t_peaks.shape
        L = BL // B
        if fp_is_dummy.size(0) == BL:
            fp_is_dummy = fp_is_dummy.view(B, L)[:, 0]
        
        x_fp = None
        fp_mask = None
        if self.fp_enable and (S_t_fp is not None):
            # S_t_fp: (B, fp_input_dim)
            assert S_t_fp.shape[0] == B, "S_t_fp batch size must match S_t_peaks"

        # (B, fp_token_len * d_fp) -> (B, fp_token_len, d_fp)
        fp_tokens = self.fp_compress(S_t_fp)                      # (B, T_fp * d_fp)
        fp_tokens = fp_tokens.view(B, self.fp_token_len, self.d_fp)
        fp_tokens = self.fp_ln(fp_tokens)

        # dummy 处理
        if fp_is_dummy is None:
            fp_is_dummy = torch.ones(B, dtype=torch.bool, device=fp_tokens.device)

        fp_dummy_mask = fp_is_dummy.unsqueeze(-1).unsqueeze(-1).float()  # (B,1,1)
        dummy_fp = self.dummy_fp_feature.expand(B, self.fp_token_len, -1)

        x_fp   = fp_tokens * (1.0 - fp_dummy_mask) + dummy_fp * fp_dummy_mask  # (B,T_fp,d_fp)
        # 目前先简单全 True（后面你可以按需设置 pad）
        fp_mask = torch.ones(B, self.fp_token_len, dtype=torch.bool, device=x_fp.device)

        # 1. Condition Vector（标量 encoder 统一输出 (B, cond_dim)，避免 hybrid 产生 (B,1,D) 导致 += 广播失败）
        def _scalar_cond_emb(encoder: nn.Module, val: torch.Tensor) -> torch.Tensor:
            if val.dim() == 2 and val.shape[-1] == 1:
                val = val.squeeze(-1)
            return encoder(val.unsqueeze(-1))

        cond_vec_base = _scalar_cond_emb(self.total_mass_encoder, condition['total_mass'])
        cond_vec_base = cond_vec_base + _scalar_cond_emb(self.precursor_mz_encoder, condition['precursor_mz'])
        cond_vec_base = cond_vec_base + _scalar_cond_emb(self.adduct_embedding, condition['adduct_weight'])
        cond_vec_base = cond_vec_base + _scalar_cond_emb(self.charge_embedding, condition['charge'])
        cond_vec_base = cond_vec_base + _scalar_cond_emb(self.energy_embedding, condition['energy'])
        cond_vec_base = cond_vec_base + self.formula_encoder(condition['formular_vector'].float())

        time_emb = self.time_embed(t)
        time_cond_emb = self.time_to_cond_mlp(time_emb)
        final_cond_vec_base = cond_vec_base + time_cond_emb
        cond_for_adaln = final_cond_vec_base.repeat_interleave(L, dim=0)  # (B*L, cond_dim)

        # 2. SMILES Input
        tok_emb = self.token_embedding(S_t_smiles_tokens)
        
        token_masses   = S_t_token_masses * smiles_mask.float()
        mass_emb       = self.token_mass_embedding(token_masses.unsqueeze(-1))
        token_formular = S_t_token_formular if S_t_token_formular is not None else torch.zeros(BL, S_len, self.formula_dim, device=tok_emb.device)
        token_formular = token_formular * smiles_mask.unsqueeze(-1).float()
        formula_emb    = self.token_formula_encoder(token_formular, formula_is_dummy=smiles_is_dummy)
        x_smiles_real = tok_emb + mass_emb + formula_emb

        # Dummy SMILES
        smiles_dummy_mask = smiles_is_dummy.unsqueeze(-1).unsqueeze(-1).float()  # (B*L,1,1)
        dummy_feat_smiles = self.dummy_smiles_feature.expand(BL, S_len, -1)
        x_smiles = x_smiles_real * (1.0 - smiles_dummy_mask) + dummy_feat_smiles * smiles_dummy_mask

        # 3. Spectrum Input
        mz = S_t_peaks[..., 0]          # (B, P_len)
        intensity = S_t_peaks[..., 1].unsqueeze(-1)   # (B, P_len, 1)

        mz_emb = self.mz_encoder(mz.unsqueeze(-1))        # (B, P_len, d_spec)
        int_emb = self.intensity_encoder(intensity)       # (B, P_len, d_spec)
        int_emb = int_emb + self.intensity_multiscale_encoder(intensity)

        x_spec_real = mz_emb + int_emb

        # paired = 同时有真实 spec 与 smiles（spectrum-only / molecule-only 不走 mass↔m/z 通路）
        smiles_real_B = ~smiles_is_dummy.view(B, L).all(dim=1)
        spec_real_B = ~spec_is_dummy.bool()
        paired_B = (smiles_real_B & spec_real_B).float().view(B, 1, 1)
        peak_valid_mask = spec_mask & (mz > 0)

        # token mass → 各峰（smiles→spec）
        token_mass_spec = self.smiles_mass_to_spec_proj(mass_emb).view(B, L, S_len, -1)
        path_w = (~smiles_is_dummy.view(B, L)).float().view(B, L, 1, 1)
        token_mass_spec = (token_mass_spec * path_w).sum(dim=1) / path_w.sum(dim=1).clamp_min(1.0)
        sm_mask_b = smiles_mask.view(B, L, S_len)
        token_mask_b = (sm_mask_b & path_w.squeeze(-1).bool()).any(dim=1)
        peak_mass_ctx = self.token_mass_peak_attn(
            x_spec_real, token_mass_spec, token_mask_b,
        )
        x_spec_with_guess = (
            x_spec_real
            + torch.tanh(self.token_mass_peak_gate) * peak_mass_ctx * paired_B
        )
        if hasattr(self, 'dummy_smiles_to_spec_feat'):
            dummy_expand = self.dummy_smiles_to_spec_feat.expand(B, P_len, -1)
            x_spec_with_guess = x_spec_with_guess + dummy_expand * (1.0 - paired_B)

        # 峰 m/z → 各 token mass（spec→smiles）
        peak_to_token_ctx = self.peak_token_mass_cross_attn(
            mass_emb, mz_emb, peak_valid_mask, L,
        )
        paired_BL = paired_B.view(B, 1, 1).repeat_interleave(L, dim=0)
        x_smiles = (
            x_smiles
            + torch.tanh(self.peak_token_mass_gate)
            * peak_to_token_ctx
            * paired_BL
        )

        # Dummy Spectrum
        spec_dummy_mask = spec_is_dummy.unsqueeze(-1).unsqueeze(-1).float()  # (B,1,1)
        dummy_feat_spec = self.dummy_spec_feature.expand(B, P_len, -1)
        x_spec = x_spec_with_guess * (1.0 - spec_dummy_mask) + dummy_feat_spec * spec_dummy_mask

        # 3.1 相对 m/z bias（只依赖 m/z 和 spec_mask，整个网络共享）
        # spec_mask: [B,P_len]，约定 True=有效，False=pad
        prec_ref = condition['precursor_mz']
        if prec_ref.dim() > 1:
            prec_ref = prec_ref.view(B, -1)[:, 0]
        spec_rel_bias = self.spec_mz_bias(
            mz_seq=mz,
            key_padding_mask=spec_mask,
            ref_mz=prec_ref,
        )  # [B,H,P_len,P_len]

        # 4. Fusion Layers
        for layer in self.fusion_layers:
            x_smiles, x_spec, x_fp = layer(
                x_smiles, x_spec,
                cond_for_adaln, L,
                smiles_mask, spec_mask,
                spec_rel_bias=spec_rel_bias,
                x_fp=x_fp,
                fp_mask=fp_mask
            )
        # 5. Output
        pred_smiles_logits = self.smiles_output_projection(self.final_smiles_ln(x_smiles))
        x_spec_out = self.final_spec_ln(x_spec)
        spec_raw = self.spec_output_projection(x_spec_out)
        if self.mz_head_mode == 'grid':
            mz_pred = self.grid_mz_head(x_spec_out)
            intensity_pred = spec_raw[..., 1].clamp_min(0.0)
            pred_spec = torch.stack([mz_pred, intensity_pred], dim=-1)
        else:
            # Keep the flow field continuous. Formula projection is an
            # inference-only post-processing operation and must not replace
            # the pretrained continuous m/z proposal inside Euler integration.
            pred_spec = spec_raw
        pred_fp = self.fp_output_projection(self.final_fp_ln(x_fp.view(B, -1))) if self.fp_enable else None

        return pred_smiles_logits, pred_spec, pred_fp
