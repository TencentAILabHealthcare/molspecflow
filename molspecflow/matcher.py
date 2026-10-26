# -*- coding: utf-8 -*-
# meteor_flow_matcher.py (drop-in improved version)
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict
from tqdm import tqdm
from .model import MeteorFlowModel
from .formula_utils import VECTOR_SIZE
from .spectrum_loss import (
    make_bin_edges,
    binned_spectrum_cosine_loss,
)
# ----------------------------- Helpers (DFM/Step Size) -----------------------------
def mix_with_mask_p0(y_true_idx: torch.Tensor, kappa: torch.Tensor, vocab_size: int, mask_id: int, smi0: Optional[torch.Tensor] = None):
    """
    κ * onehot(y_true) + (1-κ) * onehot(<mask>)
    y_true_idx: (BL, S)
    kappa:      (B,) or (BL,) -- 会自动广播到 (BL,1)
    返回        (BL, S, V)
    """
    k = kappa
    one_y = F.one_hot(y_true_idx, num_classes=vocab_size).float()
    if smi0 is not None:
        one_m = F.one_hot(smi0, num_classes=vocab_size).float()
    else:
        one_m = F.one_hot(torch.full_like(y_true_idx, mask_id), num_classes=vocab_size).float()
    while k.dim() < one_y.dim():
        k = k.unsqueeze(-1)  # 到 (BL,1)
    return k * one_y + (1 - k) * one_m


def mix_with_uniform_p0(y_true_idx: torch.Tensor,
                        kappa: torch.Tensor,
                        vocab_size: int,
                        exclude_id: Optional[int] = None):
    """
    返回 κ*onehot(y_true) + (1-κ)*Uniform
    Uniform 为 V 维均匀分布（可选排除某个类别 exclude_id）
    """
    one_y = F.one_hot(y_true_idx, num_classes=vocab_size).float()  # (..., V)

    if exclude_id is None:
        p0 = torch.full_like(one_y, 1.0 / float(vocab_size))
    else:
        p0 = torch.full_like(one_y, 1.0 / float(vocab_size - 1))
        p0[..., exclude_id] = 0.0  # 排除该类后重新归一化上面已处理

    k = kappa
    while k.dim() < one_y.dim():
        k = k.unsqueeze(-1)
    return k * one_y + (1.0 - k) * p0


def unified_step_size(h: torch.Tensor,
                      alpha_t: torch.Tensor, beta_t: torch.Tensor,
                      kappa: torch.Tensor, kappa_prime: torch.Tensor) -> torch.Tensor:
    """封闭式步长（离散/连续共用），近似限制一步内质量迁移 ≤ rho。返回 (B,)"""
    den = kappa_prime * (alpha_t / (1 - kappa).clamp_min(1e-6)
                         + beta_t  /  kappa.clamp_min(1e-6)).clamp_min(1e-6)
    return torch.minimum(h, 1 / den.clamp_min(1e-6))


class DFMSequence:
    """
    教程式 DFM 的序列版本，一步更新：
      u_f = κ'(t)/(1-κ) * ( p_{1|t}(·) - δ_{x_t} )
      u_b = κ'(t)/κ     * ( δ_{x_t} - p0 ),  p0 = δ_{<mask>}
      u   = α_t * u_f - β_t * u_b
      p_{t+h} ≈ δ_{x_t} + h * u,  然后 x_{t+h} ~ Categorical(p_{t+h})
    """
    def __init__(self, vocab_size: int, mask_id: int,
                 kappa_fn, kappa_prime_fn, alpha_fn, beta_fn):
        self.V = vocab_size
        self.mask_id = mask_id
        self.kappa = kappa_fn
        self.kappa_p = kappa_prime_fn
        self.alpha_t = alpha_fn
        self.beta_t = beta_fn

    def _dirac_onehot(self, idx: torch.Tensor) -> torch.Tensor:
        return F.one_hot(idx, num_classes=self.V).float()

    def _p0_mask(self, like_idx: torch.Tensor) -> torch.Tensor:
        return self._dirac_onehot(torch.full_like(like_idx, self.mask_id))
    
    def _uniform_p0(self, like_idx: torch.Tensor) -> torch.Tensor:
        V = self.V
        B_shape = like_idx.shape
        device = like_idx.device
        return torch.full((*B_shape, V), 1.0 / float(V), device=device)

    def step(
        self,
        t_B: torch.Tensor, h_B: torch.Tensor,
        x_idx_BL_S: torch.Tensor, p1_logits_BL_S_V: torch.Tensor,
        L: int, x0: torch.Tensor,
        valid_mask_BL_S: Optional[torch.Tensor] = None,
        debug: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        一步 DFM 更新（Euler）：
          p^+ = δ_{x_t} + h * u_t(·, x_t)
        并通过 multinomial 从 p^+ 采样下一个 token。

        t_B:           (B,)
        h_B:           (B,)
        x_idx_BL_S:    (BL, S)
        p1_logits:     (BL, S, V)
        L:             multi-path 数
        x0:            (BL, S) 初始状态（mask path 下通常是全 mask）
        valid_mask_BL_S: (BL, S) True=有效 token（PAD 位置会被冻结不更新）
        debug:         若 True，会严格检查 sum(u)=0、pt_plus>=0、sum(pt_plus)=1
        Return:
          x_new:   (BL, S)
          scores:  (BL,)  每条 path 的平均 token 概率（exp(mean log p))，可用于筛选
        """
        BL, S = x_idx_BL_S.shape
        B = t_B.shape[0]
        assert BL % B == 0 and (BL // B) == L, "BL 与 B, L 不匹配"

        # 广播到 (BL,1,1)
        def for_brd(z: torch.Tensor) -> torch.Tensor:
            return z.view(B, 1).repeat_interleave(L, dim=1).reshape(-1, 1, 1)

        k  = for_brd(self.kappa(t_B))
        kp = for_brd(self.kappa_p(t_B))
        a  = for_brd(self.alpha_t(t_B))
        b  = for_brd(self.beta_t(t_B))
        h  = for_brd(h_B)

        safe_logits = torch.nan_to_num(
            p1_logits_BL_S_V, nan=0.0, posinf=1e4, neginf=-1e4,
        )
        p1_hat  = F.softmax(safe_logits, dim=-1)            # (BL,S,V)
        dirac_x = self._dirac_onehot(x_idx_BL_S)           # (BL,S,V)
        p0      = self._dirac_onehot(x0)                   # (BL,S,V)  mask-path: δ_mask

        # u_t = α * û_t - β * ǔ_t
        uf = (kp / (1 - k).clamp_min(1e-8)) * (p1_hat - dirac_x)   # (BL,S,V)
        ub = (kp / k.clamp_min(1e-8))       * (dirac_x - p0)       # (BL,S,V)
        u  = a * uf - b * ub

        # Euler update: p^+ = δ + h u
        pt_plus = dirac_x + h * u  # (BL,S,V)

        # 严格核对：sum(u)=0；pt_plus>=0；sum(pt_plus)=1
        if debug:
            u_sum = u.sum(dim=-1)  # (BL,S)
            if u_sum.abs().max() > 1e-5:
                raise RuntimeError(f"[DFM] sum(u)!=0, max={u_sum.abs().max().item()}")
            minv = pt_plus.min().item()
            if minv < -1e-6:
                raise RuntimeError(f"[DFM] negative mass {minv} (try smaller h/T or check logits)")
            s = pt_plus.sum(dim=-1)  # (BL,S)
            if (s - 1.0).abs().max() > 1e-5:
                raise RuntimeError(f"[DFM] not normalized, max|sum-1|={(s-1).abs().max().item()}")

        # 数值保险：仅用于浮点误差（不再依赖 clamp “修正过程”）
        pt_plus = torch.nan_to_num(
            pt_plus, nan=0.0, posinf=0.0, neginf=0.0,
        ).clamp_min(0.0)
        prob_sum = pt_plus.sum(dim=-1, keepdim=True)
        pt_plus = torch.where(
            prob_sum > 1e-8,
            pt_plus / prob_sum.clamp_min(1e-8),
            dirac_x,
        )

        # PAD 冻结：无效位置保持原样
        if valid_mask_BL_S is not None:
            m = valid_mask_BL_S.unsqueeze(-1)  # (BL,S,1)
            pt_plus = pt_plus * m + dirac_x * (~m)

        x_new = torch.multinomial(pt_plus.reshape(-1, self.V), 1).view(BL, S)

        if valid_mask_BL_S is not None:
            x_new = torch.where(valid_mask_BL_S, x_new, x_idx_BL_S)

        # 正确的 log-prob：log(p(x_new))
        token_logp = pt_plus.gather(-1, x_new.unsqueeze(-1)).squeeze(-1).clamp_min(1e-12).log()
        scores = token_logp.mean(dim=-1).exp()  # (BL,)

        return x_new, scores


# ----------------------------- Scheduler/Corrector -----------------------------
class KappaScheduler:
    def __init__(self) -> None:
        pass
    def __call__(self, t: float | torch.Tensor) ->  float | torch.Tensor:
        raise NotImplementedError

    def derivative(self, t: float | torch.Tensor) -> float | torch.Tensor:
        raise NotImplementedError

class CubicScheduler(KappaScheduler):
    def __init__(self, a: float = 2.0, b: float = 0.5) -> None:
        self.a = a
        self.b = b

    def __call__(self, t: float | torch.Tensor) -> float | torch.Tensor:
        return -2* (t**3) + 3 * (t**2) + self.a * (t ** 3 - 2* t**2 + t) + self.b * (t**3 - t**2)

    def derivative(self, t: float | torch.Tensor) -> float | torch.Tensor:
        return -6 * (t**2) + 6 * t + self.a * (3 * t**2 - 4 * t + 1) + self.b * (3 * t**2 - 2 * t)


def make_alpha_beta_from_cfg(corrector_cfg: Dict):
    """
    α_t = 1 + α * t^a * (1-t)^b
    β_t = α_t - 1    （如需单独β，可在cfg里设 beta_scale 覆盖）
    """
    alpha_scale = float(corrector_cfg.get("alpha", 12.0))
    a = float(corrector_cfg.get("a", 2.0))
    b = float(corrector_cfg.get("b", 0.25))
    def alpha_t(t: torch.Tensor) -> torch.Tensor:
        return 1.0 + alpha_scale * (t**a) * ((1 - t)**b)
    def beta_t(t: torch.Tensor) -> torch.Tensor:
        beta_scale = corrector_cfg.get("beta_scale")
        if beta_scale is None:
            return alpha_t(t) - 1.0
        return torch.as_tensor(beta_scale, dtype=t.dtype, device=t.device) * torch.ones_like(t)
    return alpha_t, beta_t


# ----------------------------- MeteorFlowMatcher (drop-in) -----------------------------
class MeteorFlowMatcher(nn.Module):
    def __init__(self, model: MeteorFlowModel, config: Dict):
        super().__init__()
        self.model = model
        self.config = config

        model_cfg = config.get('model', {})
        self.task = model_cfg.get('task', 'joint')
        self.vocab_size   = int(model_cfg['vocab_size'])
        self.pad_token_id = int(model_cfg['pad_token_id'])
        # 可选单独 mask token；没有就用 pad 充当 mask
        self.mask_token_id = int(model_cfg['mask_token_id'])

        self.loss_weights = config['model']['loss_weights']
        # m/z 归一化尺度：把原始 Da 的 m/z 压到 O(1)，让 (m/z, intensity) 两个通道量纲一致，
        # 从而流匹配噪声/插值良态。默认 1.0(=不归一化)以兼容旧 checkpoint；新配置显式设 1000。
        self.mz_scale = float(model_cfg.get('mz_norm_scale', 1.0))
        # 分箱余弦训练损失(0.01 Da / max 1005 / L2 / 可选剔除前体 bin)
        self.spectral_cos_mz_max = float(model_cfg.get('spectral_cos_mz_max', 1005.0))
        self.spectral_cos_bin_res = float(model_cfg.get('spectral_cos_bin_res', 0.01))
        self.spectral_cos_remove_precursor = bool(model_cfg.get('spectral_cos_remove_precursor', True))
        # 仅在 t > t_min 时计入多尺度 cos（低 t 时 pred 不可信，信号噪声大）
        self.spectral_cos_t_min = float(model_cfg.get('spectral_cos_t_min', 0.5))
        # Multi-resolution cosine can be generated on a log-spaced resolution
        # axis.  This covers the interval continuously in log space without
        # letting the number of scales increase the total loss weight.
        _default_multiscale = [
            {'bin_res': 1.0, 'soft_sigma_da': 0.5, 'loss_key': 'spectral_cos_1da'},
            {'bin_res': 0.1, 'soft_sigma_da': 0.1, 'loss_key': 'spectral_cos_0p1da'},
            {'bin_res': 0.01, 'soft_sigma_da': 0.05, 'loss_key': 'spectral_cos_0p01da'},
            {'bin_res': 0.001, 'soft_sigma_da': 0.005, 'loss_key': 'spectral_cos_0p001da'},
        ]
        log_cfg = model_cfg.get('spectral_cos_logspace', {}) or {}
        if bool(log_cfg.get('enabled', False)):
            max_res = float(log_cfg.get('max_bin_res', 5.0))
            min_res = float(log_cfg.get('min_bin_res', 0.01))
            n_scales = max(int(log_cfg.get('num_scales', 7)), 2)
            sigma_ratio = float(log_cfg.get('soft_sigma_ratio', 1.0))
            sigma_min = float(log_cfg.get('soft_sigma_min_da', 0.01))
            resolutions = [
                math.exp(
                    math.log(max_res)
                    + i * (math.log(min_res) - math.log(max_res)) / (n_scales - 1)
                )
                for i in range(n_scales)
            ]
            resolutions = [
                float(x) for x in log_cfg.get('extra_bin_res', [10.0])
            ] + resolutions
            self.spectral_cos_multiscale = []
            seen = set()
            for res in resolutions:
                rounded = round(res, 10)
                if rounded in seen:
                    continue
                seen.add(rounded)
                key_res = f'{res:.6g}'.replace('.', 'p').replace('-', 'm')
                self.spectral_cos_multiscale.append({
                    'bin_res': res,
                    'soft_sigma_da': max(res * sigma_ratio, sigma_min),
                    'loss_key': f'spectral_cos_{key_res}da',
                    'scale_weight': 1.0,
                })
        else:
            self.spectral_cos_multiscale = list(
                model_cfg.get('spectral_cos_multiscale', _default_multiscale)
            )
        self.spectral_cos_ms_weight = float(
            self.loss_weights.get('spectral_cos_multiscale', self.loss_weights.get('spectral_cos', 0.25))
        )
        scale_weight_sum = sum(
            max(float(entry.get('scale_weight', 1.0)), 0.0)
            for entry in self.spectral_cos_multiscale
        )
        scale_weight_sum = max(scale_weight_sum, 1e-8)
        self._cos_scale_runtime = []
        for entry in self.spectral_cos_multiscale:
            loss_key = str(entry['loss_key'])
            bin_res = float(entry['bin_res'])
            soft_sigma = float(entry.get('soft_sigma_da', bin_res * 0.5))
            buf_name = f'_bin_edges_{loss_key}'
            self.register_buffer(
                buf_name,
                make_bin_edges(self.spectral_cos_mz_max, bin_res),
                persistent=False,
            )
            self._cos_scale_runtime.append({
                'loss_key': loss_key,
                'bin_res': bin_res,
                'soft_sigma_da': soft_sigma,
                'bin_edges_attr': buf_name,
                'weight_fraction': max(
                    float(entry.get('scale_weight', 1.0)), 0.0
                ) / scale_weight_sum,
            })
        # 兼容旧日志 key
        self.register_buffer(
            'spectral_cos_bin_edges',
            make_bin_edges(self.spectral_cos_mz_max, self.spectral_cos_bin_res),
            persistent=False,
        )
        # --- Huber(smooth-L1) 带宽，单位=peak m/z 的原始尺度(mz_scale=1 时即 Da) ---
        # 为什么用 Huber 而非 MSE：Da 空间 m/z/质量 MSE 量纲 O(1e4)，直接调大权重会梯度爆炸。
        # Huber 在 |误差|>beta 处梯度被截断为常数(∝weight，不随误差增大)，故可安全调大权重；
        # 在 |误差|<beta 处退化为二次，保证接近目标时精细优化。beta 越小越接近 L1(更鲁棒/更强的稳定牵引)。
        self.mz_huber_beta = float(model_cfg.get('mz_huber_beta', 10.0))
        self.mass_huber_beta = float(model_cfg.get('mass_huber_beta', 10.0))
        # 流匹配 S_0 中 m/z 通道的高斯标准差(Da)。intensity 仍 ~ N(0,1)。
        # 典型 MS peak m/z std≈100–150 Da；若 S_0 m/z ~ N(0,1) 则插值路径几乎线性爬升，
        # 低 t 区域模型看到的是 O(1) Da 的伪谱，与推理从 randn 起步一致地偏离真实扩散。
        # Legacy checkpoints use standard Gaussian noise in normalized m/z
        # space and do not contain this key. Larger Da-like noise scales must
        # be enabled explicitly by newer configs.
        self.mz_flow_noise_scale = float(model_cfg.get('mz_flow_noise_scale', 1.0))
        # 谱 flow / m/z loss 在相对 precursor 的空间：Δm = precursor_mz - m/z_abs；encoder 仍吃绝对 Da。
        # Legacy checkpoints model absolute normalized m/z and do not contain
        # this key. Relative-gap flow must be enabled explicitly.
        self.mz_use_relative = bool(model_cfg.get('mz_use_relative', False))
        time_cfg = model_cfg.get('time_sampling', {}) or {}
        self.time_uniform_ratio = float(time_cfg.get('uniform_ratio', 0.7))
        self.time_endpoint_beta_a = float(time_cfg.get('endpoint_beta_a', 5.0))
        self.time_endpoint_beta_b = float(time_cfg.get('endpoint_beta_b', 1.0))
        if not 0.0 <= self.time_uniform_ratio <= 1.0:
            raise ValueError("time_sampling.uniform_ratio must be in [0, 1].")
        if self.time_endpoint_beta_a <= 0.0 or self.time_endpoint_beta_b <= 0.0:
            raise ValueError("Beta time-sampling parameters must be positive.")
        self.mz_da_t_min = float(model_cfg.get('mz_da_t_min', 0.5))
        self.mz_ppm_t_min = float(model_cfg.get('mz_ppm_t_min', 0.8))
        self.spectral_cos_fine_t_min = float(
            model_cfg.get('spectral_cos_fine_t_min', 0.8)
        )
        self.spectral_cos_ultrafine_t_min = float(
            model_cfg.get('spectral_cos_ultrafine_t_min', 0.9)
        )
        self.scheduler = CubicScheduler()

        # α_t, β_t
        corrector_cfg = config.get("corrector", {"alpha": 12.0, "a": 2.0, "b": 0.25})
        self.alpha_t, self.beta_t = make_alpha_beta_from_cfg(corrector_cfg)

        # 原子质量表
        atom_mass_tensor = torch.zeros(self.vocab_size, dtype=torch.float32)
        if atom_masses_list := model_cfg.get('atom_masses', []):
            n_masses = min(len(atom_masses_list), self.vocab_size)
            atom_mass_tensor[:n_masses] = torch.tensor(atom_masses_list[:n_masses], dtype=torch.float32)
        self.register_buffer("atom_mass_tensor", atom_mass_tensor)
        token_formula_tensor = torch.zeros((self.vocab_size, VECTOR_SIZE), dtype=torch.int)
        if formula_list := model_cfg.get('formula_list', []):
            n_formulas = min(len(formula_list), self.vocab_size)
            token_formula_tensor[:n_formulas, :] = torch.tensor(formula_list[:n_formulas], dtype=torch.int)
        self.register_buffer("token_formula_tensor", token_formula_tensor)

    def _precursor_b(self, condition: Dict, B: int, device, dtype) -> torch.Tensor:
        """(B,) precursor m/z in Da."""
        prec = condition['precursor_mz'].to(device=device, dtype=dtype)
        if prec.dim() > 1:
            prec = prec.view(B, -1)[:, 0]
        return prec

    def _peaks_abs_to_rel_mz(self, peaks: torch.Tensor, precursor: torch.Tensor) -> torch.Tensor:
        """Δm = precursor - m/z_abs（中性丢失距离；允许负值=isotope 高于 precursor）。"""
        out = peaks.clone()
        out[..., 0] = precursor.view(-1, 1) - peaks[..., 0]
        return out

    def _peaks_rel_to_abs_mz(self, peaks: torch.Tensor, precursor: torch.Tensor) -> torch.Tensor:
        """m/z_abs = precursor - Δm。"""
        out = peaks.clone()
        out[..., 0] = precursor.view(-1, 1) - peaks[..., 0]
        return out

    def _spec_evolves(self, task: str) -> bool:
        return task in ('smiles_to_spec', 'joint')

    def _peaks_for_model(self, peaks: torch.Tensor, precursor: torch.Tensor, task: str) -> torch.Tensor:
        """Encoder 始终使用绝对 Da m/z。"""
        if self.mz_use_relative and self._spec_evolves(task):
            return self._peaks_rel_to_abs_mz(peaks, precursor)
        return peaks

    def _pred_abs_to_flow_peaks(self, pred_spec: torch.Tensor, precursor: torch.Tensor) -> torch.Tensor:
        """模型输出绝对 m/z → 与 flow 状态同空间（相对 Δm）的 peaks。"""
        out = pred_spec.clone()
        if self.mz_use_relative:
            out[..., 0] = precursor.view(-1, 1) - pred_spec[..., 0]
        return out

    # ----------------------------- m/z 归一化辅助 -----------------------------
    def _norm_peaks_mz(self, peaks: torch.Tensor) -> torch.Tensor:
        """把 m/z 通道(索引0)缩放到 O(1)，intensity 通道(索引1)保持不变。返回副本。"""
        if self.mz_scale == 1.0:
            return peaks
        out = peaks.clone()
        out[..., 0] = out[..., 0] / self.mz_scale
        return out

    def _denorm_peaks_mz(self, peaks: torch.Tensor) -> torch.Tensor:
        """_norm_peaks_mz 的逆操作：把 m/z 通道还原回 Da。返回副本。"""
        if self.mz_scale == 1.0:
            return peaks
        out = peaks.clone()
        out[..., 0] = out[..., 0] * self.mz_scale
        return out

    def _sample_s0_peaks(self, S_1_peaks: torch.Tensor) -> torch.Tensor:
        """Rectified-flow 先验：intensity ~ N(0,1)；m/z 在 flow 空间(相对 Δm 或绝对 Da) ~ N(0, σ)。"""
        S_0 = torch.randn_like(S_1_peaks)
        scale = self.mz_flow_noise_scale
        if scale != 1.0:
            S_0[..., 0] = S_0[..., 0] * scale
        return S_0

    def _sample_initial_peaks(self, B: int, n_peaks: int, device: torch.device) -> torch.Tensor:
        """generate / generate_trajectory 的谱初值，与训练 S_0 同分布。"""
        peaks = torch.randn(B, n_peaks, 2, device=device)
        scale = self.mz_flow_noise_scale
        if scale != 1.0:
            peaks[..., 0] = peaks[..., 0] * scale
        return peaks

    def _binned_cosine_spec_loss(
        self,
        pred_peaks: torch.Tensor,
        true_peaks: torch.Tensor,
        prec_ion_mz: Optional[torch.Tensor] = None,
        bin_res: Optional[float] = None,
        soft_sigma_da: Optional[float] = None,
        bin_edges: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """MassSpecGym 对齐分箱 cos；不对 peak 做 spec_mask（靠 m/z>0 自然过滤 padding）。"""
        B, P, _ = pred_peaks.shape
        all_valid = torch.ones(B, P, dtype=torch.bool, device=pred_peaks.device)
        br = self.spectral_cos_bin_res if bin_res is None else bin_res
        sigma = 0.05 if soft_sigma_da is None else soft_sigma_da
        edges = self.spectral_cos_bin_edges if bin_edges is None else bin_edges
        return binned_spectrum_cosine_loss(
            pred_peaks,
            true_peaks,
            all_valid,
            prec_ion_mz=prec_ion_mz,
            mz_max=self.spectral_cos_mz_max,
            bin_res=br,
            remove_precursor=self.spectral_cos_remove_precursor,
            bin_edges=edges,
            soft_sigma_da=sigma,
        )

    # ----------------------------- Training (forward) -----------------------------
    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        S_1_smiles_tokens = batch['tokens']          # (BL, S)
        raw_peaks = batch['peaks']
        finite_spec_sample = torch.isfinite(raw_peaks).flatten(1).all(dim=1)
        safe_mz = torch.nan_to_num(
            raw_peaks[..., 0], nan=0.0, posinf=0.0, neginf=0.0,
        ).clamp(min=0.0, max=self.spectral_cos_mz_max)
        safe_intensity = torch.nan_to_num(
            raw_peaks[..., 1], nan=0.0, posinf=0.0, neginf=0.0,
        ).clamp(min=0.0, max=1.0)
        safe_peaks = torch.stack((safe_mz, safe_intensity), dim=-1)
        # Keep an explicit physical-Da copy for metric losses.  The flow state is
        # normalized, but Da/ppm/cosine losses must never consume that state.
        S_1_peaks_abs = safe_peaks
        # 归一化 m/z：让 (m/z, intensity) 两通道量纲一致，流匹配才良态。
        S_1_peaks = self._norm_peaks_mz(safe_peaks)   # (B, P, 2)
        raw_condition = batch['condition']
        condition = {
            key: (
                torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
                if torch.is_tensor(value) and value.is_floating_point()
                else value
            )
            for key, value in raw_condition.items()
        }
        smiles_mask       = batch['smiles_mask']     # (BL, S) True=valid token
        spec_mask         = batch['spec_mask']       # (B, P)

        smiles_is_dummy   = batch['smiles_is_dummy'] # (BL,)
        spec_is_dummy     = batch['spec_is_dummy']   # (B,)
        ecfp = torch.nan_to_num(
            batch['ecfp'], nan=0.0, posinf=0.0, neginf=0.0,
        )                                             # (B, 2048)

        device = S_1_smiles_tokens.device
        BL, S = S_1_smiles_tokens.shape
        B, P, _ = S_1_peaks.shape
        L = BL // B
        V = self.vocab_size
        assert BL % B == 0, "BL 与 B 不整除，multi-path 维度不一致"

        # 当前训练的 task（默认 joint）
        task = getattr(self, "task", "joint")
        raw_prec_b = self._precursor_b(
            raw_condition, B, device, S_1_peaks.dtype,
        )
        finite_precursor = torch.isfinite(raw_prec_b) & (raw_prec_b > 0.0)
        spec_is_dummy = spec_is_dummy | (~finite_spec_sample) | (~finite_precursor)
        prec_b = torch.nan_to_num(
            raw_prec_b, nan=1.0, posinf=self.spectral_cos_mz_max, neginf=1.0,
        ).clamp(min=1e-6, max=self.spectral_cos_mz_max)
        if self.mz_use_relative and self._spec_evolves(task):
            S_1_peaks = self._peaks_abs_to_rel_mz(S_1_peaks, prec_b)

        # ====================== 构造条件 & 采样 S_t ======================
        with torch.no_grad():
            # --- 1) total_mass from SMILES，一直用真值 S_1 ---
            token_mass = self.atom_mass_tensor[S_1_smiles_tokens]         # (BL,S)
            valid_smiles_mask = smiles_mask & (~smiles_is_dummy.unsqueeze(-1))  # (BL,S)
            total_mass_per_path = (token_mass * valid_smiles_mask).sum(dim=1)   # (BL,)
            total_mass_B = total_mass_per_path.view(B, L).mean(dim=1)           # (B,)
            condition['total_mass'] = total_mass_B

            # --- 2) 采样时间 t_B & κ(t) ---
            t_uniform = torch.rand(B, device=device, dtype=S_1_peaks.dtype)
            if self.time_uniform_ratio < 1.0:
                beta_dist = torch.distributions.Beta(
                    self.time_endpoint_beta_a, self.time_endpoint_beta_b
                )
                t_endpoint = beta_dist.sample((B,)).to(
                    device=device, dtype=S_1_peaks.dtype
                )
                use_endpoint = torch.rand(B, device=device) >= self.time_uniform_ratio
                t_B = torch.where(use_endpoint, t_endpoint, t_uniform)
            else:
                t_B = t_uniform
            t_B = t_B.clamp(min=1e-4, max=1.0 - 1e-4)
            kappa_B = self.scheduler(t_B)  # (B,)

            # ----------------- 3) 按 task 构造 S_t -----------------
            if task == 'spec_to_smiles':
                # 固定谱：S_t_peaks 一直等于 S_1_peaks
                S_t_peaks = S_1_peaks
                S_0_peaks = S_1_peaks  # 只是占位，不参与谱损失

                # 只演化 SMILES：从 mask 分布出发
                P_t_smiles = mix_with_mask_p0(
                    S_1_smiles_tokens,
                    kappa_B.repeat_interleave(L, dim=0),   # (BL,)
                    self.vocab_size,
                    self.mask_token_id,
                    smi0=batch.get('mol_prior') if self.config['model'].get('use_mol_prior', False) else None,
                )  # (BL,S,V)
                S_t_smiles_tokens = torch.multinomial(
                    P_t_smiles.flatten(0, 1), 1
                ).view(BL, S)
                fp_t = torch.zeros_like(ecfp)

            elif task == 'smiles_to_spec':
                # 固定 SMILES：S_t_smiles 一直等于 S_1_smiles
                S_t_smiles_tokens = S_1_smiles_tokens

                # 只演化谱：S_0 ~ (m/z: N(0,σ_mz), intensity: N(0,1))，线性插值到 S_1
                S_0_peaks = self._sample_s0_peaks(S_1_peaks)
                S_t_peaks = (1 - t_B.view(-1, 1, 1)) * S_0_peaks + t_B.view(-1, 1, 1) * S_1_peaks
                fp_t = ecfp

            elif task == 'joint':  # 'joint'
                # SMILES：κ*onehot(y) + (1-κ)*onehot(<mask>)，再采样
                P_t_smiles = mix_with_mask_p0(
                    S_1_smiles_tokens,
                    kappa_B.repeat_interleave(L, dim=0),   # (BL,)
                    self.vocab_size,
                    self.mask_token_id,
                    smi0=batch.get('mol_prior') if self.config['model'].get('use_mol_prior', False) else None,
                )  # (BL,S,V)
                S_t_smiles_tokens = torch.multinomial(
                    P_t_smiles.flatten(0, 1), 1
                ).view(BL, S)

                # 谱：S_0 与训练/推理一致的 m/z 尺度噪声，线性插值到 S_1
                S_0_peaks = self._sample_s0_peaks(S_1_peaks)
                S_t_peaks = (1 - t_B.view(-1, 1, 1)) * S_0_peaks + t_B.view(-1, 1, 1) * S_1_peaks
                
                fp_0 = torch.randn_like(ecfp)
                fp_1 = ecfp
                fp_t = (1 - t_B.view(-1, 1)) * fp_0 + t_B.view(-1, 1) * fp_1
                
            elif task == 'spec_to_ecfp':
                # 固定谱：S_t_peaks 一直等于 S_1_peaks
                S_t_peaks = S_1_peaks
                S_0_peaks = S_1_peaks  # 只是占位，不参与谱损失
                S_t_smiles_tokens = torch.full_like(S_1_smiles_tokens, self.mask_token_id, dtype=torch.long, device=device)
                fp_0 = torch.randn_like(ecfp)
                fp_1 = ecfp
                fp_t = (1 - t_B.view(-1, 1)) * fp_0 + t_B.view(-1, 1) * fp_1
                

        # ====================== 模型前向 ======================
        smiles_mask_in = torch.ones_like(S_1_smiles_tokens, dtype=torch.bool)
        spec_mask_in   = torch.ones_like(S_1_peaks[..., 0], dtype=torch.bool)
        S_t_token_masses = self.atom_mass_tensor[S_t_smiles_tokens]
        S_t_token_formular = self.token_formula_tensor[S_t_smiles_tokens].float()
        S_t_peaks_in = self._peaks_for_model(S_t_peaks, prec_b, task)

        pred_smiles_logits, pred_spec, pred_fp = self.model(
            S_t_smiles_tokens, S_t_token_masses,
            S_t_peaks_in, t_B, condition,
            smiles_mask_in, spec_mask_in,
            smiles_is_dummy, spec_is_dummy,
            S_t_fp=fp_t, fp_is_dummy=smiles_is_dummy,
            S_t_token_formular=S_t_token_formular,
        )

        # ====================== Losses ======================
        losses: Dict[str, torch.Tensor] = {}

        # -------- a) SMILES CE loss（只在需要演化 SMILES 的 task 上训练） --------
        ce_all = F.cross_entropy(
            pred_smiles_logits.reshape(-1, V),      # (BL*S, V)
            S_1_smiles_tokens.view(-1),             # (BL*S,)
            reduction='none',
        )  # (BL*S,)
        ce_all = ce_all.view(BL, S)                 # (BL,S)

        # 对 path 聚合，再按 non-dummy 平均
        per_path_ce = ce_all.mean(dim=-1)           # (BL,)
        per_path_ce = per_path_ce * (~smiles_is_dummy)  # (BL,)
        denom = (~smiles_is_dummy).sum().clamp_min(1e-8)
        smiles_loss = per_path_ce.sum() / denom

        losses['smiles_loss'] = smiles_loss * self.loss_weights.get('smiles', 1.0)
        if task == 'spec_to_ecfp':
            losses['smiles_loss'] = losses['smiles_loss'] * 0.0  # 该任务不训练 smiles loss 但是必须占用


        # -------- b) Spectrum loss（只在需要演化谱的 task 上训练） --------

        pred_flow = self._pred_abs_to_flow_peaks(pred_spec, prec_b)
        v_target_spec = (S_1_peaks - S_0_peaks)
        v_pred_spec   = (pred_flow   - S_0_peaks)

        valid_spec = (1 - spec_is_dummy).float()  # (B,)
        valid_peak = spec_mask.bool() & valid_spec.bool().unsqueeze(1)
        valid_peak_f = valid_peak.float()
        valid_peak_denom = valid_peak_f.sum().clamp_min(1.0)

        # Spectrum endpoint is predicted in normalized flow space.  Keep this
        # well-conditioned objective, then add physical Da and ppm anchors.
        w_mz_rel = float(self.loss_weights.get(
            'mz_flow', self.loss_weights.get('mz_rel', self.loss_weights.get('mz', 1.0))
        ))
        w_mz_abs = float(self.loss_weights.get('mz_abs', 1.0e-4))
        w_mz_ppm = float(self.loss_weights.get('mz_ppm', 1.0e-4))
        w_mz_total = float(self.loss_weights.get('mz_total', 1.0))
        flow_beta = max(self.mz_huber_beta / max(self.mz_scale, 1.0), 1e-8)
        loss_mz_rel_all = F.smooth_l1_loss(
            v_pred_spec[..., 0],
            v_target_spec[..., 0],
            reduction='none',
            beta=flow_beta,
        )
        loss_mz_rel = (loss_mz_rel_all * valid_peak_f).sum() / valid_peak_denom
        losses['mz_rel_loss'] = loss_mz_rel * w_mz_rel

        pred_spec_abs = pred_spec.clone()
        pred_spec_abs[..., 0] = pred_spec_abs[..., 0] * self.mz_scale
        pred_mz_abs = pred_spec_abs[..., 0]
        true_mz_abs = S_1_peaks_abs[..., 0]
        loss_mz_abs_all = F.smooth_l1_loss(
            pred_mz_abs,
            true_mz_abs,
            reduction='none',
            beta=self.mz_huber_beta,
        )
        da_gate = (t_B >= self.mz_da_t_min).float().unsqueeze(1)
        valid_peak_da = valid_peak_f * da_gate
        valid_peak_da_denom = valid_peak_da.sum().clamp_min(1.0)
        loss_mz_abs = (
            loss_mz_abs_all * valid_peak_da
        ).sum() / valid_peak_da_denom
        losses['mz_abs_loss'] = loss_mz_abs * w_mz_abs

        # Scale ppm to units of 10,000 ppm so the initial weighted objective is
        # O(1) rather than O(1e5), while retaining a 10 ppm Huber transition.
        ppm_scale = 1.0e4
        ppm_error_scaled = (
            1.0e6 * (pred_mz_abs - true_mz_abs)
            / true_mz_abs.clamp_min(1.0)
        ) / ppm_scale
        loss_mz_ppm_all = F.smooth_l1_loss(
            ppm_error_scaled,
            torch.zeros_like(ppm_error_scaled),
            reduction='none',
            beta=10.0 / ppm_scale,
        )
        ppm_gate = (t_B >= self.mz_ppm_t_min).float().unsqueeze(1)
        valid_peak_ppm = valid_peak_f * ppm_gate
        valid_peak_ppm_denom = valid_peak_ppm.sum().clamp_min(1.0)
        loss_mz_ppm = (
            loss_mz_ppm_all * valid_peak_ppm
        ).sum() / valid_peak_ppm_denom
        losses['mz_ppm_loss'] = loss_mz_ppm * w_mz_ppm

        # The component keys remain available for logging.  Only mz_loss is
        # optimized, so flow/Da/ppm terms are not counted twice.
        mz_weight_norm = max(abs(w_mz_rel) + abs(w_mz_abs) + abs(w_mz_ppm), 1e-8)
        losses['mz_loss'] = (
            losses['mz_rel_loss'] + losses['mz_abs_loss'] + losses['mz_ppm_loss']
        ) * (w_mz_total / mz_weight_norm)

        # intensity 已在 [0,1]，量纲正常，保留 MSE。
        loss_it_all = F.mse_loss(
            v_pred_spec[..., 1], v_target_spec[..., 1], reduction='none'
        )  # (B,P)

        # intensity
        loss_it_all = loss_it_all.mean(dim=-1)          # (B,)
        loss_it = (loss_it_all * valid_spec).sum() / valid_spec.sum().clamp_min(1e-8)
        losses['intensity_loss'] = loss_it * self.loss_weights.get('intensity', 1.0)
        if task in ['spec_to_smiles', 'spec_to_ecfp']:
            losses['mz_rel_loss']    = losses['mz_rel_loss']    * 0.0
            losses['mz_abs_loss']    = losses['mz_abs_loss']    * 0.0
            losses['mz_ppm_loss']    = losses['mz_ppm_loss']    * 0.0
            losses['mz_loss']        = losses['mz_loss']        * 0.0
            losses['intensity_loss'] = losses['intensity_loss'] * 0.0

        # -------- b2) 多尺度 soft-bin cosine（严格在物理 Da 空间） --------
        spec_supervise = task in ['smiles_to_spec', 'joint']
        cos_valid = spec_mask.bool()

        adduct_w = condition['adduct_weight'].to(device=device, dtype=pred_spec.dtype)
        if adduct_w.dim() > 1:
            adduct_w = adduct_w.view(B, -1)[:, 0]
        prec_ion = prec_b + adduct_w

        for scale in self._cos_scale_runtime:
            loss_key = scale['loss_key']
            w = self.spectral_cos_ms_weight * scale['weight_fraction']
            loss_name = f'{loss_key}_loss'
            if w > 0.0 and spec_supervise:
                scale_t_min = self.spectral_cos_t_min
                if scale['bin_res'] <= 0.1:
                    scale_t_min = max(
                        scale_t_min, self.spectral_cos_fine_t_min
                    )
                if scale['bin_res'] <= 0.05:
                    scale_t_min = max(
                        scale_t_min, self.spectral_cos_ultrafine_t_min
                    )
                t_gate = (t_B >= scale_t_min).float()
                bin_edges = getattr(self, scale['bin_edges_attr'])
                cos_vec = binned_spectrum_cosine_loss(
                    pred_spec_abs, S_1_peaks_abs, cos_valid, prec_ion_mz=prec_ion,
                    mz_max=self.spectral_cos_mz_max,
                    bin_res=scale['bin_res'],
                    remove_precursor=self.spectral_cos_remove_precursor,
                    bin_edges=bin_edges,
                    soft_sigma_da=scale['soft_sigma_da'],
                )
                loss_cos = (cos_vec * valid_spec * t_gate).sum() / (valid_spec * t_gate).sum().clamp_min(1e-8)
                losses[loss_name] = loss_cos * w
            else:
                losses[loss_name] = pred_spec.sum() * 0.0

        # 兼容旧日志 key（0.01 / 0.1 Da 档，权重与 multiscale 一致）
        w_ms = self.spectral_cos_ms_weight
        losses['spectral_cos_loss'] = losses.get(
            'spectral_cos_0p01da_loss', pred_spec.sum() * 0.0
        )
        losses['spectral_cos_coarse_loss'] = losses.get(
            'spectral_cos_0p1da_loss', pred_spec.sum() * 0.0
        )
        w_legacy = float(self.loss_weights.get('spectral_cos', 0.0))
        if w_legacy > 0.0 and w_legacy != w_ms:
            losses['spectral_cos_loss'] = losses['spectral_cos_loss'] * (w_legacy / max(w_ms, 1e-8))
        w_coarse_legacy = float(self.loss_weights.get('spectral_cos_coarse', 0.0))
        if w_coarse_legacy > 0.0 and w_coarse_legacy != w_ms:
            losses['spectral_cos_coarse_loss'] = losses['spectral_cos_coarse_loss'] * (
                w_coarse_legacy / max(w_ms, 1e-8)
            )

        if task in ['spec_to_smiles', 'spec_to_ecfp']:
            for scale in self._cos_scale_runtime:
                lk = f"{scale['loss_key']}_loss"
                if lk in losses:
                    losses[lk] = losses[lk] * 0.0
            losses['spectral_cos_coarse_loss'] = losses['spectral_cos_coarse_loss'] * 0.0
            losses['spectral_cos_loss'] = losses['spectral_cos_loss'] * 0.0
        
        fp_is_dummy = smiles_is_dummy.view(B, L)[:, 0]  # (B,)
        fp_mse = (pred_fp - ecfp).pow(2).sum(dim=-1) / 204.8
        fp_mse = (fp_mse * (~fp_is_dummy)).sum() / (~fp_is_dummy).float().sum().clamp_min(1e-8)
        losses['fp_loss'] = fp_mse
        if task == 'spec_to_smiles':
            losses['fp_loss'] = losses['fp_loss'] * 0.001  # 该任务不训练 fp loss 但是必须占用

        # -------- c) Mass Loss（可以在所有 task 上保留，作为正则） --------
        p1_hat = F.softmax(pred_smiles_logits, dim=-1)            # (BL,S,V)
        expected_mass = (p1_hat @ self.atom_mass_tensor)          # (BL,S)
        pred_total_mass = expected_mass.sum(dim=1).view(B, L).mean(dim=1)  # (B,)
        tgt_total_mass  = condition['total_mass']                  # (B,)

        # 总质量控制同样用 Huber(Da 空间量纲大)，便于调大权重而不爆炸。
        loss_mass_vec = F.smooth_l1_loss(pred_total_mass, tgt_total_mass, beta=self.mass_huber_beta, reduction='none')  # (B,)
        valid_mass_mask = (1 - spec_is_dummy).float()  # 这里也可以加 smiles_is_dummy 的逻辑

        loss_mass = (loss_mass_vec * valid_mass_mask).sum() / valid_mass_mask.sum().clamp_min(1e-8)
        loss_mass = loss_mass * self.loss_weights.get('mass', 1e-3)

        losses['mass_loss'] = loss_mass
        
        
        
        if task in ['spec_to_smiles', 'joint']:

            formula_vec = condition.get('formular_vector', None)  # [B,VECTOR_SIZE]
            if isinstance(formula_vec, torch.Tensor):
                formula_vec = formula_vec.to(device).float()

                formula_is_dummy = condition.get('formula_is_dummy', None)
                if not isinstance(formula_is_dummy, torch.Tensor):
                    formula_is_dummy = torch.zeros((B,), device=device, dtype=torch.bool)
                else:
                    formula_is_dummy = formula_is_dummy.to(device).view(-1).bool()

                # [B,VECTOR_SIZE] -> [BL,VECTOR_SIZE]
                gt_formula = formula_vec.repeat_interleave(L, dim=0)  # (BL,VECTOR_SIZE)
                valid_formula = (~formula_is_dummy).repeat_interleave(L)  # (BL,)

                # token mask：尽量别把 PAD 位置计入（你现在 CE 没mask，但 formula 约束最好 mask）
                token_mask = (S_1_smiles_tokens != self.pad_token_id) & (~smiles_is_dummy.unsqueeze(1))  # (BL,S)

                # E[formula] = sum_{pos,v} p(pos,v) * token_formula[v]
                token_formula = self.token_formula_tensor.to(device)  # (V,VECTOR_SIZE)
                p = torch.softmax(pred_smiles_logits, dim=-1)          # (BL,S,V)

                exp_formula = torch.einsum(
                    "bsv,ve->be",
                    p * token_mask.unsqueeze(-1).float(),
                    token_formula.float()
                )  # (BL,VECTOR_SIZE)

                # 归一化成比例（更稳）：除以总原子数
                # exp_sum = exp_formula.sum(dim=1, keepdim=True).clamp_min(1.0)
                # gt_sum  = gt_formula.sum(dim=1, keepdim=True).clamp_min(1.0)
                exp_norm = exp_formula
                gt_norm  = gt_formula

                # per-sample huber
                per = torch.nn.functional.smooth_l1_loss(exp_norm, gt_norm, reduction="none").mean(dim=1)  # (BL,)

                denom = valid_formula.float().sum().clamp_min(1.0)
                formula_loss = (per * valid_formula.float()).sum() / denom

                losses['formula_loss'] = formula_loss * float(self.loss_weights.get('formula', 1.0))
        else:
            # 没提供 formula_vec 就不加
            pass

        return losses

    # ----------------------------- Inference (generate with DFM) -----------------------------
    @torch.no_grad()
    def generate(
        self,
        batch: Dict[str, torch.Tensor],
        n_smiles: int, n_peaks: int,
        L: int = 1, T: int = 100,
        solver: str = 'euler', task: str = 'joint',
        rho: float = 0.2,
        shallow_fusion_lambda: float = 0.0,   # <<< 新增：0=关闭；>0 开启 shallow fusion
    ):
        """
        DFM 正式采样：
        - spec_to_smiles:  固定谱，SMILES 从 mask 全序列开始并演化
        - smiles_to_spec:  固定 SMILES（来自 batch），仅演化谱
        - joint:           SMILES 从 mask 开始 & 谱从 0 开始，同时演化

        shallow fusion（只对“生成 SMILES”的任务生效：spec_to_smiles / joint）：
        logp_fused = logp_cond + λ * logp_uncond
        """
        solver = str(solver).lower()
        if solver not in ('euler', 'heun'):
            raise ValueError(f"Unsupported solver: {solver!r}; expected 'euler' or 'heun'.")
        device = next(self.model.parameters()).device
        condition = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch['condition'].items()}
        B = condition['precursor_mz'].shape[0]
        prec_b = self._precursor_b(condition, B, device, torch.float32)

        # --------- 取 dummy flags（尽量兼容你的数据格式）---------
        # spec dummy：优先 batch['spec_is_dummy']，否则尝试 condition['is_dummy']，否则全 False
        spec_is_dummy = batch.get('spec_is_dummy')
        if spec_is_dummy is None:
            spec_is_dummy = condition.get('is_dummy')
        if spec_is_dummy is None:
            spec_is_dummy = torch.zeros(B, dtype=torch.bool, device=device)
        else:
            spec_is_dummy = spec_is_dummy.to(device).view(-1).bool()
            if spec_is_dummy.numel() != B:
                spec_is_dummy = spec_is_dummy.view(B, -1)[:, 0].bool()

        # SMILES dummy：推理时我们要生成 SMILES，所以这里必须全 False
        smiles_is_dummy = torch.zeros(B * L, dtype=torch.bool, device=device)

        # --------- 保障 total_mass 存在（避免 dummy tokens 覆盖掉真实 precursor 质量）---------
        if 'total_mass' not in condition:
            # fallback：用 precursor - adduct（或直接 precursor）
            if 'adduct_weight' in condition:
                total_mass_fallback = condition['precursor_mz'] - condition['adduct_weight']
            else:
                total_mass_fallback = condition['precursor_mz']

            tokens = batch.get("tokens", None)
            if tokens is not None:
                tokens = tokens.to(device)  # (BL,S)
                # tokens_is_dummy 可能是 [B,1] 或 [B]，你的 dataset 里是 tensor([0/1])
                tokens_is_dummy = batch.get("tokens_is_dummy", None)
                if tokens_is_dummy is not None:
                    tokens_is_dummy = tokens_is_dummy.to(device).view(-1)
                    # 兼容 [B,1]
                    if tokens_is_dummy.numel() != B:
                        tokens_is_dummy = tokens_is_dummy.view(B, -1)[:, 0]
                    tokens_is_dummy = tokens_is_dummy.bool()
                else:
                    tokens_is_dummy = None

                # 用 tokens 估 total_mass（只对非 dummy 的样本启用）
                token_mass = self.atom_mass_tensor[tokens]  # (BL,S)
                # 用 pad 来当 mask（如果你有更严格的 smiles_mask，也可以换成 batch['smiles_mask']）
                valid_mask = (tokens != self.pad_token_id).bool()
                total_mass_per_path = (token_mass * valid_mask.float()).sum(dim=1)         # (BL,)
                total_mass_B_tokens = total_mass_per_path.view(B, L).mean(dim=1)           # (B,)

                if tokens_is_dummy is None:
                    condition['total_mass'] = total_mass_B_tokens
                else:
                    condition['total_mass'] = torch.where(tokens_is_dummy, total_mass_fallback, total_mass_B_tokens)
            else:
                condition['total_mass'] = total_mass_fallback

        # -------- masks（推理时全 True；spec_mask 若 batch 提供则用 batch 的）---------
        n_smiles = max(n_smiles, self.config['data']['max_seq_len'])
        n_peaks = 20

        smiles_mask_inf = torch.ones(B * L, n_smiles, dtype=torch.bool, device=device)

        spec_mask_in = batch.get('spec_mask')
        if task in ('spec_to_smiles', 'spec_to_ecfp') and isinstance(
            spec_mask_in, torch.Tensor
        ):
            # Only a conditioning spectrum may provide its real validity mask.
            spec_mask_inf = spec_mask_in.to(device).bool()
        else:
            # Generated spectra cannot use the ground-truth peak-count mask.
            spec_mask_inf = torch.ones(B, n_peaks, dtype=torch.bool, device=device)

        # -------- 初值 --------
        if task == 'smiles_to_spec':
            S_1_smiles_tokens = batch['tokens'].to(device)  # (BL, S)
            assert S_1_smiles_tokens.shape[0] == B * L
            x_idx = S_1_smiles_tokens.clone()
        else:
            if 'mol_prior' in batch.keys() and self.config['model'].get('use_mol_prior', False):
                mol_prior = batch['mol_prior']
                x_idx = mol_prior
            else:
                x_idx = torch.full((B * L, n_smiles), self.mask_token_id, dtype=torch.long, device=device)

        # -------- peaks / fp 初值 --------
        if task in ['spec_to_smiles', 'spec_to_ecfp']:
            # 谱作为条件：归一化 m/z 后再喂给模型（与训练一致）
            peaks = self._norm_peaks_mz(batch['peaks'].to(device))  # (B,P,2)
            assert peaks.shape[1] == n_peaks
        else:
            peaks = self._sample_initial_peaks(B, n_peaks, device)

        # ecfp 可能不存在：兜底全 0
        if 'ecfp' in batch and isinstance(batch['ecfp'], torch.Tensor):
            ecfp_in = batch['ecfp'].to(device)
            if ecfp_in.dim() == 1:
                ecfp_in = ecfp_in.view(1, -1).repeat(B, 1)
        else:
            ecfp_in = torch.zeros(B, 2048, device=device)

        if task in ['spec_to_ecfp', "joint"]:
            fp_0 = torch.randn(B, 2048, device=device)
            fp_t = fp_0.clone()
        else:
            fp_t = ecfp_in

        # -------- DFM --------
        dfm = DFMSequence(
            vocab_size=self.vocab_size,
            mask_id=self.mask_token_id,
            kappa_fn=self.scheduler, kappa_prime_fn=self.scheduler.derivative,
            alpha_fn=self.alpha_t, beta_fn=self.beta_t
        )

        # -------- 时间推进 --------
        t_B = torch.full((B,), 1e-4, device=device)
        default_h = torch.full((B,), 1.0 / T, device=device)

        x0 = x_idx.clone()
        peak_0 = peaks.clone()

        token_scores = []

        # ===== shallow fusion 的“无条件”输入：无谱图（spec_mask=全 False + spec_is_dummy=1 + peaks=0）=====
        do_shallow_fusion = (shallow_fusion_lambda is not None) and (float(shallow_fusion_lambda) > 0.0) \
                            and (task in ['spec_to_smiles'])

        if do_shallow_fusion:
            peaks_uncond = torch.zeros_like(peaks)
            spec_mask_uncond = torch.zeros_like(spec_mask_inf)  # 全 False
            spec_is_dummy_uncond = torch.ones_like(spec_is_dummy, dtype=torch.bool)  # 全 dummy

        while t_B.max() < 1 - 1e-6:
            token_mass     = self.atom_mass_tensor[x_idx]
            token_formular = self.token_formula_tensor[x_idx].float()
            peaks_in = self._peaks_for_model(peaks, prec_b, task)

            # -------- 条件 forward --------
            pred_logits_cond, pred_s, pred_fp = self.model(
                x_idx, token_mass, peaks_in,
                t_B, condition,
                smiles_mask_inf, spec_mask_inf,
                smiles_is_dummy, spec_is_dummy,
                S_t_fp=fp_t, fp_is_dummy=smiles_is_dummy,
                S_t_token_formular=token_formular
            )

            # -------- shallow fusion（只融合 SMILES logits；谱/FP 不融合）--------
            pred_logits = pred_logits_cond
            if do_shallow_fusion:
                peaks_uncond_in = self._peaks_for_model(peaks_uncond, prec_b, task)
                pred_logits_uncond, _, _ = self.model(
                    x_idx, token_mass, peaks_uncond_in,
                    t_B, condition,
                    smiles_mask_inf, spec_mask_uncond,
                    smiles_is_dummy, spec_is_dummy_uncond,
                    S_t_fp=fp_t, fp_is_dummy=smiles_is_dummy,
                    S_t_token_formular=token_formular
                    
                )
                # logp_fused = logp_cond + λ logp_uncond
                logp_cond = F.log_softmax(pred_logits_cond, dim=-1)
                logp_uncond = F.log_softmax(pred_logits_uncond, dim=-1)
                pred_logits = logp_cond + float(shallow_fusion_lambda) * logp_uncond
                # 注意：DFMSequence.step 内部会 softmax(pred_logits)，这里 pred_logits 直接当 logits 用即可

            # -------- 步长 --------
            kappa = self.scheduler(t_B)
            kappa_p = self.scheduler.derivative(t_B)
            alpha = self.alpha_t(t_B)
            beta  = self.beta_t(t_B)
            if task == 'smiles_to_spec':
                # Continuous spectrum flow uses the same linear time path as
                # training. Do not distort it with the discrete DFM jump bound.
                h_B = default_h
            else:
                h_B = unified_step_size(
                    default_h, alpha, beta, kappa, kappa_p
                )
            h_B = torch.minimum(h_B, (1.0 - 1e-6) - t_B)

            # -------- 更新 SMILES --------
            if task in ['spec_to_smiles', 'joint']:
                x_idx, scores = dfm.step(t_B, h_B, x_idx, pred_logits, L=L, x0=x0)
                token_scores.append(scores.view(B, L).detach().cpu())

            # -------- 更新谱 --------
            if task in ['smiles_to_spec', 'joint']:
                pred_flow = self._pred_abs_to_flow_peaks(pred_s, prec_b)
                pred_v_spec = (pred_flow - peak_0)
                h_spec = h_B.view(B, 1, 1)
                if solver == 'euler':
                    peaks = peaks + h_spec * pred_v_spec
                else:
                    # Heun predictor-corrector for the continuous spectrum.
                    peaks_predictor = peaks + h_spec * pred_v_spec
                    t_next = (t_B + h_B).clamp(max=1.0 - 1e-6)
                    token_mass_next = self.atom_mass_tensor[x_idx]
                    token_formula_next = self.token_formula_tensor[x_idx].float()
                    peaks_predictor_in = self._peaks_for_model(
                        peaks_predictor, prec_b, task
                    )
                    _, pred_s_next, _ = self.model(
                        x_idx, token_mass_next, peaks_predictor_in,
                        t_next, condition,
                        smiles_mask_inf, spec_mask_inf,
                        smiles_is_dummy, spec_is_dummy,
                        S_t_fp=fp_t, fp_is_dummy=smiles_is_dummy,
                        S_t_token_formular=token_formula_next,
                    )
                    pred_flow_next = self._pred_abs_to_flow_peaks(
                        pred_s_next, prec_b
                    )
                    pred_v_spec_next = pred_flow_next - peak_0
                    peaks = peaks + 0.5 * h_spec * (
                        pred_v_spec + pred_v_spec_next
                    )

            # -------- 更新 ECFP --------
            if task in ['spec_to_ecfp', "joint"]:
                pred_v_ecfp = (pred_fp - fp_0)
                fp_t = fp_t + h_B.view(B, 1) * pred_v_ecfp

            t_B = (t_B + h_B).clamp(max=1.0 - 1e-6)

        # -------- 后处理谱：强度归一化 + 按强度降序重排（与 GT 数据一致） --------
        if task in ['smiles_to_spec', 'joint']:
            peaks[..., 1].clamp_(min=0.0)
            maxv = peaks[..., 1].amax(dim=1, keepdim=True).clamp_min(1e-8)
            peaks[..., 1] = peaks[..., 1] / maxv
            sort_idx = peaks[..., 1].argsort(dim=-1, descending=True)
            peaks = torch.gather(peaks, 1, sort_idx.unsqueeze(-1).expand(-1, -1, 2))
            final_formula_mz = None
            if self.mz_use_relative:
                peaks = self._peaks_rel_to_abs_mz(peaks, prec_b)
            if self.mz_scale != 1.0:
                peaks[..., 0] = (peaks[..., 0] * self.mz_scale).clamp_min(0.0)
            if final_formula_mz is not None:
                # Euler states stay continuous; only the returned spectrum is
                # snapped to the last hard integer-formula endpoint.
                peaks[..., 0] = final_formula_mz.clamp_min(0.0)
        elif self.mz_scale != 1.0:
            # 谱作为条件的任务：把归一化的条件谱还原回 Da 再返回
            peaks = self._denorm_peaks_mz(peaks)

        final_fp  = fp_t.round().clamp(min=0.0, max=1.0)
        fp_scores = (fp_t - final_fp).sum(dim=-1).abs()
        final_fp = final_fp.int()

        return {
            "smiles_tokens": x_idx,
            "peaks": peaks,
            "token_scores": token_scores,
            "ecfp": final_fp,
            "fp_scores": fp_scores
        }

    @torch.no_grad()
    def generate_trajectory(
        self,
        batch: Dict[str, torch.Tensor],
        n_smiles: int, n_peaks: int,
        L: int = 1, T: int = 100,
        solver: str = 'euler', task: str = 'joint',
        rho: float = 0.2,
        record_every: int = 1,
    ):
        """
        和 generate 类似，但额外返回 flow matching 的轨迹：
        - 使用 **ground truth 作为模型预测值**，避免异常数据；
        - 返回每个时间步的 SMILES token、谱图 peaks、ECFP 的轨迹。

        返回 dict:
            {
              "smiles_tokens": 最终的 x_idx,
              "peaks":         最终的 peaks,
              "ecfp":          最终的 fp_t,
              "trajectory": {
                  "t": [K],                 # 时间序列
                  "smiles_tokens": [K, BL, S],
                  "peaks":         [K, B, P, 2],
                  "ecfp":          [K, B, D],
              }
            }
        """
        device = next(self.model.parameters()).device
        condition = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch['condition'].items()}
        B = condition['precursor_mz'].shape[0]
        prec_b = self._precursor_b(condition, B, device, torch.float32)

        # total_mass
        if 'total_mass' not in condition:
            if "tokens" in batch.keys():
                token_mass = self.atom_mass_tensor[batch['tokens'].to(device)]         # (BL,S)
                valid_smiles_mask = torch.ones_like(batch['tokens'], dtype=torch.bool, device=device)  # (BL,S)
                total_mass_per_path = (token_mass * valid_smiles_mask).sum(dim=1)   # (BL,)
                total_mass_B = total_mass_per_path.view(B, L).mean(dim=1)           # (B,)
                condition['total_mass'] = total_mass_B 
            
            elif 'adduct_weight' in condition:
                condition['total_mass'] = condition['precursor_mz'] - condition['adduct_weight']
            else:
                condition['total_mass'] = condition['precursor_mz']

        # dummy flags
        spec_is_dummy   = batch.get('spec_is_dummy', torch.zeros(B, dtype=torch.bool, device=device))
        smiles_is_dummy = torch.zeros(B * L, dtype=torch.bool, device=device)

        # masks（推理时全 True）
        n_smiles = max(n_smiles, self.config['data']['max_seq_len'])
        n_peaks = n_peaks
        smiles_mask_inf = torch.ones(B * L, n_smiles, dtype=torch.bool, device=device)
        spec_mask_inf   = torch.ones(B, n_peaks, dtype=torch.bool, device=device)

        # ground truth 终点（x_1）
        gt_tokens = batch['tokens'].to(device)          # (BL, S)
        # 归一化 m/z（与训练/generate 一致）；返回前再反归一化
        gt_peaks  = self._norm_peaks_mz(batch['peaks'].to(device))   # (B, P, 2)
        gt_ecfp   = batch['ecfp'].to(device)            # (B, D)
        if self.mz_use_relative and self._spec_evolves(task):
            gt_peaks_flow = self._peaks_abs_to_rel_mz(gt_peaks, prec_b)
        else:
            gt_peaks_flow = gt_peaks

        # 初值
        if task == 'smiles_to_spec':
            # 必须给定真实 SMILES
            S_1_smiles_tokens = gt_tokens  # (BL, S)
            assert S_1_smiles_tokens.shape[0] == B * L
            x_idx = S_1_smiles_tokens.clone()
        else:
            # 从全 mask 开始
            x_idx = torch.full((B * L, n_smiles), self.mask_token_id, dtype=torch.long, device=device)

        if task in ['spec_to_smiles', 'spec_to_ecfp']:
            peaks = gt_peaks.clone()  # 固定输入谱
            assert peaks.shape[1] == n_peaks
        else:
            peaks = self._sample_initial_peaks(B, n_peaks, device)

        if task in ['spec_to_ecfp', "joint"]:
            fp_0 = torch.randn(B, gt_ecfp.shape[1], device=device)
            fp_t = fp_0.clone()
        else:
            fp_t = gt_ecfp.clone()

        # DFM 步进器
        dfm = DFMSequence(
            vocab_size=self.vocab_size,
            mask_id=self.mask_token_id,
            kappa_fn=self.scheduler, kappa_prime_fn=self.scheduler.derivative,
            alpha_fn=self.alpha_t, beta_fn=self.beta_t
        )

        # 时间推进
        t_B = torch.full((B,), 1e-4, device=device)
        default_h = torch.full((B,), 1.0 / T, device=device)
        x0 = x_idx.clone()
        peak_0 = peaks.clone()

        # === 关键：这里不用真正的 model，改用 ground truth 作为每次的“预测” ===
        # 这样生成的轨迹是朝向 ground truth 的理想 flow，避免奇怪的样本
        gt_onehot_logits = F.one_hot(gt_tokens, num_classes=self.vocab_size).float() * 1000.0  # (BL, S, V)

        # 轨迹缓存
        traj_t = []
        traj_x = []
        traj_peaks = []
        traj_fp = []

        # 记录初始状态
        traj_t.append(t_B[0].item())
        traj_x.append(x_idx.detach().cpu().clone())
        traj_peaks.append(peaks.detach().cpu().clone())
        traj_fp.append(fp_t.detach().cpu().clone())

        step_idx = 0
        while t_B.max() < 1 - 1e-6:
            # 这里不调用 self.model，直接用 ground truth 构造“预测”
            pred_logits = gt_onehot_logits  # (BL, S, V)
            pred_s = gt_peaks_flow          # (B, P, 2) flow 空间
            pred_fp = gt_ecfp               # (B, D)

            # 步长 h
            kappa = self.scheduler(t_B)
            kappa_p = self.scheduler.derivative(t_B)
            alpha = self.alpha_t(t_B)
            beta  = self.beta_t(t_B)
            h_B = unified_step_size(default_h, alpha, beta, kappa, kappa_p)
            h_B = torch.minimum(h_B, (1.0 - 1e-6) - t_B)

            # 更新 SMILES
            if task in ['spec_to_smiles', 'joint']:
                x_idx, _ = dfm.step(t_B, h_B, x_idx, pred_logits, L=L, x0=x0)

            # 更新谱
            if task in ['smiles_to_spec', 'joint']:
                pred_v_spec = (pred_s - peak_0)
                peaks = peaks + h_B.view(B, 1, 1) * pred_v_spec

            # 更新 ECFP
            if task in ['spec_to_ecfp', "joint"]:
                pred_v_ecfp = (pred_fp - fp_0)
                fp_t = fp_t + h_B.view(B, 1) * pred_v_ecfp

            # 时间前进
            t_B = (t_B + h_B).clamp(max=1.0 - 1e-6)
            step_idx += 1

            # 记录轨迹
            if (step_idx % record_every == 0) or (t_B.max() >= 1 - 1e-6):
                traj_t.append(t_B[0].item())
                traj_x.append(x_idx.detach().cpu().clone())
                traj_peaks.append(peaks.detach().cpu().clone())
                traj_fp.append(fp_t.detach().cpu().clone())

        # 后处理谱强度
        if task in ['smiles_to_spec', 'joint']:
            peaks[..., 1].clamp_(min=0.0)
            maxv = peaks[..., 1].amax(dim=1, keepdim=True).clamp_min(1e-8)
            peaks[..., 1] = peaks[..., 1] / maxv
            if self.mz_use_relative:
                peaks = self._peaks_rel_to_abs_mz(peaks, prec_b)
            if self.mz_scale != 1.0:
                peaks[..., 0] = (peaks[..., 0] * self.mz_scale).clamp_min(0.0)
        elif self.mz_scale != 1.0:
            peaks = self._denorm_peaks_mz(peaks)

        # 堆叠轨迹
        traj = {
            "t": torch.tensor(traj_t, dtype=torch.float32),
            "smiles_tokens": torch.stack(traj_x, dim=0),   # (K, BL, S)
            "peaks": torch.stack(traj_peaks, dim=0),       # (K, B, P, 2)
            "ecfp": torch.stack(traj_fp, dim=0),           # (K, B, D)
        }

        return {
            "smiles_tokens": x_idx,
            "peaks": peaks,
            "ecfp": fp_t,
            "trajectory": traj,
        }
