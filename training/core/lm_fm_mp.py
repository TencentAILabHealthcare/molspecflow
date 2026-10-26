import torch
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from typing import Dict, Optional, List, Tuple, Any
import os
import json
import time
# 导入新的 SMILES + Spectrum 模型
from molspecflow import MeteorFlowModel, MeteorFlowMatcher
import torch.nn as nn
class LitMeteorFlowMatcher(pl.LightningModule):
    """
    PyTorch Lightning wrapper for the Meteor (SMILES + Spectrum) 
    Conditional Flow Matching model.
    """
    def __init__(self, config: Dict):
        super().__init__()
        # 保存配置字典作为超参数
        self.save_hyperparameters()
        self.config = config
        self._tokenizer = None  # 惰性加载 Tokenizer

        # 实例化核心模型和 CFM 包装器
        # config 字典期望有一个 'model' 键用于存放模型特定的参数
        model_config = self.config.get('model', {}) 
        model = MeteorFlowModel(model_config)
        
        # Matcher 接收模型和 *整个* config
        self.cfm = MeteorFlowMatcher(model, self.config)
        
        test_cfg = self.config.get('test', {}) if isinstance(self.config, dict) else getattr(self.config, 'test', {})
        if isinstance(test_cfg, dict):
            self.test_num_samples_per_example: int = int(test_cfg.get('num_samples_per_example', 100))
            self.test_save_dir: str = test_cfg.get('save_dir', 'test_generation_results')
            self.test_tasks: List[str] = test_cfg.get(
                'tasks',
                ['joint', 'spec_to_smiles', 'smiles_to_spec']
            )
        else:
            # OmegaConf 或简单对象的 fallback
            self.test_num_samples_per_example = getattr(test_cfg, 'num_samples_per_example', 100)
            self.test_save_dir = getattr(test_cfg, 'save_dir', 'test_generation_results')
            self.test_tasks = getattr(test_cfg, 'tasks', ['joint', 'spec_to_smiles', 'smiles_to_spec'])

        # 每个 rank 内部的 sample 计数器（只用于给 sample 一个编号，方便 debug）
        self.test_sample_index: int = 0

        # rank 内 sample 计数器 + 输出目录
        self._test_sample_offset: int = 0
        self._test_output_dir: Optional[str] = None
        
        self.use_ema = bool(getattr(self.config.training, "use_ema", False)) if hasattr(self.config, "training") else False
        self.ema_decay = float(getattr(getattr(self.config, "training", {}), "ema_decay", 0.999)) if isinstance(getattr(self, "config", None), object) else 0.999
        # EMA 延迟启动步数：< 该步数不维护 EMA(验证用实时权重)；到达时以"当时的平稳权重"为起点初始化 EMA。
        # 默认 0 = 从头(step 0)启动 EMA，保持原有行为。
        self.ema_start_step = int(getattr(getattr(self.config, "training", {}), "ema_start_step", 0) or 0) if hasattr(self.config, "training") else 0
        self._ema_params = None
        self._ema_backup = None

    def _init_ema_from_current(self, tag: str = ""):
        self._ema_params = {}
        for n, p in self.named_parameters():
            if p.requires_grad:
                self._ema_params[n] = p.detach().clone().to("cpu")
        try:
            is_zero = (self.trainer is None) or bool(getattr(self.trainer, "is_global_zero", True))
        except Exception:
            is_zero = True
        if is_zero:
            print(f"[EMA] 以当前权重初始化 EMA {tag}")

    def on_fit_start(self):
        if not self.use_ema:
            return
        # 若设置了延迟启动，则先不初始化(等到达 ema_start_step 再以平稳权重为起点)
        if self.ema_start_step > 0:
            self._ema_params = None
            if (self.trainer is None) or bool(getattr(self.trainer, "is_global_zero", True)):
                print(f"[EMA] 延迟启动：global_step 到达 {self.ema_start_step} 前不维护 EMA(验证用实时权重)")
            return
        self._init_ema_from_current(tag="(step 0)")

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if not self.use_ema:
            return
        # 延迟启动：到达 ema_start_step 时，用当前(平稳)权重初始化 EMA
        if self._ema_params is None:
            gstep = self.trainer.global_step if self.trainer is not None else 0
            if gstep >= self.ema_start_step:
                self._init_ema_from_current(tag=f"(global_step={gstep})")
            return
        beta = self.ema_decay
        with torch.no_grad():
            for n, p in self.named_parameters():
                if n in self._ema_params:
                    # cpu buffer
                    ema = self._ema_params[n]
                    ema.mul_(beta).add_(p.detach().to("cpu"), alpha=(1.0 - beta))
                    
    def _swap_to_ema(self):
        if not self.use_ema or self._ema_params is None:
            return
        self._ema_backup = {}
        with torch.no_grad():
            for n, p in self.named_parameters():
                if n in self._ema_params:
                    self._ema_backup[n] = p.detach().clone()
                    p.copy_(self._ema_params[n].to(p.device, dtype=p.dtype))

    def _swap_back(self):
        if self._ema_backup is None:
            return
        with torch.no_grad():
            for n, p in self.named_parameters():
                if n in self._ema_backup:
                    p.copy_(self._ema_backup[n].to(p.device, dtype=p.dtype))
        self._ema_backup = None

    def on_validation_start(self):
        self._swap_to_ema()

    def on_validation_end(self):
        self._swap_back()

    def on_test_start(self):
        self._swap_to_ema()

    def on_test_end(self):
        self._swap_back()

    def _cfg_get(self, cfg: Any, key: str, default=None):
        if cfg is None:
            return default
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        return getattr(cfg, key, default)

    def _cfg_get_path(self, cfg: Any, keys: List[str], default=None):
        cur = cfg
        for k in keys:
            if cur is None:
                return default
            if isinstance(cur, dict):
                cur = cur.get(k, None)
            else:
                cur = getattr(cur, k, None)
        return default if cur is None else cur

    def _current_task(self) -> str:
        # MeteorFlowMatcher 里：self.task = config['model']['task']
        return str(getattr(self.cfm, "task", self._cfg_get_path(self.config, ["model", "task"], "joint")))

    def _is_finetune_mode(self) -> bool:
        task = self._current_task()
        enabled = self._cfg_get_path(self.config, ["training", "finetune", "enabled"], "auto")
        if isinstance(enabled, str) and enabled.lower() == "auto":
            return task != "joint"
        return bool(enabled)

    def _finetune_head_patterns(self, task: str) -> List[str]:
        # MeteorFlowModel 里 head 的命名（你 meteor_flow_model_arch.py 里就是这些）
        if task == "spec_to_smiles":
            return ["final_smiles_ln", "smiles_output_projection"]
        if task == "smiles_to_spec":
            return ["final_spec_ln", "spec_output_projection"]
        if task == "spec_to_ecfp":
            return ["fp_compress", "fp_ln", "final_fp_ln", "fp_output_projection"]
        # fallback
        return ["output_projection"]

    def _apply_finetune_freeze(self) -> None:
        """
        当 task != joint（或 training.finetune.enabled=True）时：
        - 冻结整个 MeteorFlowModel
        - 解冻：最后 K 个 fusion_layers + 对应 task head
        - 可选：LN/bias、condition encoders、embeddings
        """
        self._finetune_enabled = False
        self._ft_head_named_params = []
        self._ft_backbone_named_params = []

        if not self._is_finetune_mode():
            return

        self._finetune_enabled = True
        task = self._current_task()

        ft_cfg = self._cfg_get_path(self.config, ["training", "finetune"], {}) or {}
        k_last = int(self._cfg_get(ft_cfg, "train_last_fusion_layers", 2))
        train_ln = bool(self._cfg_get(ft_cfg, "train_layernorm", True))
        train_bias = bool(self._cfg_get(ft_cfg, "train_bias", True))
        train_cond = bool(self._cfg_get(ft_cfg, "train_condition_encoders", True))
        train_emb = bool(self._cfg_get(ft_cfg, "train_embeddings", False))

        m: nn.Module = self.cfm.model
        head_patterns = self._finetune_head_patterns(task)

        # 1) freeze everything
        for p in m.parameters():
            p.requires_grad_(False)

        # 2) unfreeze heads
        for n, p in m.named_parameters():
            if any(k in n for k in head_patterns):
                p.requires_grad_(True)

        # 3) unfreeze last K fusion layers
        if hasattr(m, "fusion_layers") and isinstance(m.fusion_layers, nn.ModuleList) and len(m.fusion_layers) > 0:
            kk = max(0, min(k_last, len(m.fusion_layers)))
            if kk > 0:
                for layer in m.fusion_layers[-kk:]:
                    for p in layer.parameters():
                        p.requires_grad_(True)

        # 4) optional: unfreeze condition encoders (小模块，通常安全)
        if train_cond:
            cond_patterns = [
                "precursor_mz_encoder", "total_mass_encoder", "adduct_embedding",
                "mz_encoder", "token_mass_embedding",
                "token_mass_peak_attn", "peak_token_mass_cross_attn", "formula_mass_head", "grid_mz_head",
                "sin_proj", "fourier_proj",
                "charge_embedding", "energy_embedding",
                "time_embed", "time_to_cond_mlp",
            ]
            for n, p in m.named_parameters():
                if any(k in n for k in cond_patterns):
                    p.requires_grad_(True)

        # 5) optional: LN/bias (更稳的轻量适配)
        if train_ln or train_bias:
            for n, p in m.named_parameters():
                nl = n.lower()
                if train_ln and (("layernorm" in nl) or (".ln" in nl) or ("_ln" in nl) or nl.endswith("norm.weight") or nl.endswith("norm.bias")):
                    p.requires_grad_(True)
                if train_bias and nl.endswith(".bias"):
                    p.requires_grad_(True)

        # 6) optional: embeddings (通常别开，容易过拟合)
        if train_emb:
            for n, p in m.named_parameters():
                if "embed" in n.lower() or "embedding" in n.lower():
                    p.requires_grad_(True)

        # 7) split trainable params into head/backbone for optimizer groups
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if any(k in n for k in head_patterns):
                self._ft_head_named_params.append((n, p))
            else:
                self._ft_backbone_named_params.append((n, p))

        # print summary
        try:
            is_zero = (self.trainer is None) or bool(getattr(self.trainer, "is_global_zero", True))
        except Exception:
            is_zero = True
        if is_zero:
            total = sum(pp.numel() for pp in self.parameters())
            trainable = sum(pp.numel() for pp in self.parameters() if pp.requires_grad)
            print(f"[Finetune] enabled | task={task} | trainable={trainable/1e6:.2f}M / total={total/1e6:.2f}M ({trainable/total*100:.2f}%)")
            print(f"[Finetune] trainable tensors: head={len(self._ft_head_named_params)}, backbone={len(self._ft_backbone_named_params)}")

    # ---------------- L2-SP (optional) ----------------
    def _maybe_init_l2sp(self) -> None:
        self._l2sp_lambda = 0.0
        self._l2sp_ref = None

        if not getattr(self, "_finetune_enabled", False):
            return
        ft_cfg = self._cfg_get_path(self.config, ["training", "finetune"], {}) or {}
        lam = float(self._cfg_get(ft_cfg, "l2sp_lambda", 0.0))
        if lam <= 0:
            return

        self._l2sp_lambda = lam
        ref_dtype = str(self._cfg_get(ft_cfg, "l2sp_ref_dtype", "float16")).lower()
        use_fp16 = ref_dtype in ["fp16", "float16", "half"]

        ref = {}
        for n, p in getattr(self, "_ft_backbone_named_params", []):
            t = p.detach().cpu()
            ref[n] = t.half() if use_fp16 else t.float()
        self._l2sp_ref = ref

        try:
            is_zero = (self.trainer is None) or bool(getattr(self.trainer, "is_global_zero", True))
        except Exception:
            is_zero = True
        if is_zero:
            numel = sum(v.numel() for v in ref.values())
            print(f"[Finetune] L2-SP enabled | lambda={lam} | cached {len(ref)} tensors, {numel/1e6:.2f}M elems")

    def _l2sp_penalty(self) -> torch.Tensor:
        lam = float(getattr(self, "_l2sp_lambda", 0.0))
        ref = getattr(self, "_l2sp_ref", None)
        if lam <= 0 or ref is None:
            return torch.zeros((), device=self.device)

        acc = torch.zeros((), device=self.device)
        denom = 0
        for n, p in getattr(self, "_ft_backbone_named_params", []):
            r = ref.get(n, None)
            if r is None:
                continue
            r = r.to(device=p.device, dtype=p.dtype)
            diff = p - r
            acc = acc + (diff * diff).sum()
            denom += p.numel()
        if denom == 0:
            return torch.zeros((), device=self.device)
        return (acc / float(denom)) * lam

    def setup(self, stage: Optional[str] = None) -> None:
        # Lightning 会在 configure_optimizers 前调用 setup("fit")
        if stage in (None, "fit"):
            if not hasattr(self, "_ft_setup_done"):
                self._ft_setup_done = True
                self._apply_finetune_freeze()
                self._maybe_init_l2sp()
        
    def forward(self, batch: Dict) -> Dict:
        """
        用于训练和验证的前向传播。
        直接调用 MeteorFlowMatcher 的 forward 方法。
        """
        return self.cfm(batch)

    @torch.no_grad()
    def generate(self, 
                 batch: Dict, 
                 n_smiles: int, 
                 n_peaks: int, 
                 L: int = 1, 
                 T: int = 100, 
                 solver: str = 'euler',
                 task: str = "joint") -> Dict:
        """
        面向用户的生成方法。
        从条件中生成 SMILES token 和谱图峰。

        Args:
            batch (Dict): 至少包含 `batch['condition']` 的批次数据。
            n_smiles (int): 要生成的SMILES长度。
            n_peaks (int): 要生成的谱图峰数量。
            L (int): 每个谱图要生成的SMILES路径数。
            T (int): 积分步数。
            solver (str): 'euler' or 'heun'.

        Returns:
            Dict: 一个包含 "smiles_tokens" [B*L, n_smiles] 和 "peaks" [B, n_peaks, 2] 的字典。
        """
        # 从批次中提取条件
        # condition = batch['condition']
        
        # 调用 CFM 的 generate 方法
        return self.cfm.generate(
            batch=batch,
            # condition=condition,
            n_smiles=n_smiles,
            n_peaks=n_peaks,
            L=L,
            T=T,
            solver=solver,
            task=task
        )

    def training_step(self, batch: Dict, batch_idx: int):
        losses = self.forward(batch)
        # m/z component losses are diagnostic values already combined
        # and normalized into mz_loss by MeteorFlowMatcher.
        total_loss = sum(
            value for key, value in losses.items()
            if key not in (
                'mz_rel_loss', 'mz_abs_loss', 'mz_ppm_loss',
                'spectral_cos_loss', 'spectral_cos_coarse_loss',
            )
        )

        # 可选 L2-SP（只在 finetune 且 l2sp_lambda>0 时生效）
        if getattr(self, "_finetune_enabled", False) and float(getattr(self, "_l2sp_lambda", 0.0)) > 0:
            l2sp = self._l2sp_penalty()
            total_loss = total_loss + l2sp
            self.log("train_l2sp", l2sp, on_step=True, on_epoch=True, sync_dist=True)

        self._log_losses(losses, total_loss, batch, step_name='train')
        return total_loss

    def validation_step(self, batch: Dict, batch_idx: int):
        """执行单个验证步骤。"""
        # 1. 前向传播 (有梯度)
        losses = self.forward(batch)
        
        # 2. 计算总损失 (有梯度)
        total_loss = sum(
            value for key, value in losses.items()
            if key not in (
                'mz_rel_loss', 'mz_abs_loss', 'mz_ppm_loss',
                'spectral_cos_loss', 'spectral_cos_coarse_loss',
            )
        )
        
        # 3. 记录日志 (在 @torch.no_grad() 内部执行)
        self._log_losses(losses, total_loss, batch, step_name='val')

        # 4. 返回带梯度的总损失 (用于 val_epoch_end 等)
        return total_loss

    @torch.no_grad()
    def on_before_optimizer_step(self, optimizer) -> None:
        """Prevent a non-finite batch gradient from contaminating Adam state."""
        for parameter in self.parameters():
            if parameter.grad is not None:
                parameter.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)

    def configure_optimizers(self):
        tr_cfg = self._cfg_get_path(self.config, ["training"], {}) or {}

        lr_joint = float(self._cfg_get(tr_cfg, "learning_rate", 2e-4))
        wd = float(self._cfg_get(tr_cfg, "weight_decay", 0.01))
        betas = self._cfg_get(tr_cfg, "betas", (0.9, 0.95))
        eps = float(self._cfg_get(tr_cfg, "adam_eps", 1e-8))

        def _split_decay(params: List[Tuple[str, torch.nn.Parameter]]):
            decay, nodecay = [], []
            for n, p in params:
                if not p.requires_grad:
                    continue
                nl = n.lower()
                # LN / bias / 1D params 不做 weight decay
                if p.ndim == 1 or nl.endswith(".bias") or ("norm" in nl) or (".ln" in nl) or ("_ln" in nl):
                    nodecay.append(p)
                else:
                    decay.append(p)
            return decay, nodecay

        param_groups = []

        if getattr(self, "_finetune_enabled", False):
            ft_cfg = self._cfg_get_path(self.config, ["training", "finetune"], {}) or {}
            lr_head = float(self._cfg_get(ft_cfg, "head_lr", 2e-5))
            lr_backbone = float(self._cfg_get(ft_cfg, "backbone_lr", 2e-6))

            # backbone group
            b_decay, b_nodecay = _split_decay(getattr(self, "_ft_backbone_named_params", []))
            if len(b_decay) > 0:
                param_groups.append({"params": b_decay, "lr": lr_backbone, "weight_decay": wd})
            if len(b_nodecay) > 0:
                param_groups.append({"params": b_nodecay, "lr": lr_backbone, "weight_decay": 0.0})

            # head group
            h_decay, h_nodecay = _split_decay(getattr(self, "_ft_head_named_params", []))
            if len(h_decay) > 0:
                param_groups.append({"params": h_decay, "lr": lr_head, "weight_decay": wd})
            if len(h_nodecay) > 0:
                param_groups.append({"params": h_nodecay, "lr": lr_head, "weight_decay": 0.0})

            try:
                is_zero = (self.trainer is None) or bool(getattr(self.trainer, "is_global_zero", True))
            except Exception:
                is_zero = True
            if is_zero:
                print(f"[Finetune] Optimizer groups={len(param_groups)} | lr_backbone={lr_backbone} lr_head={lr_head} wd={wd}")

            optimizer = torch.optim.AdamW(param_groups, lr=lr_head, betas=betas, eps=eps)
        else:
            # joint / 全参
            named = [(n, p) for n, p in self.named_parameters() if p.requires_grad]
            decay, nodecay = _split_decay(named)
            if len(decay) > 0:
                param_groups.append({"params": decay, "lr": lr_joint, "weight_decay": wd})
            if len(nodecay) > 0:
                param_groups.append({"params": nodecay, "lr": lr_joint, "weight_decay": 0.0})

            optimizer = torch.optim.AdamW(param_groups, lr=lr_joint, betas=betas, eps=eps)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=getattr(self.trainer, "max_epochs", 100)
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
        
    @torch.no_grad()
    def _log_losses(self, 
                    losses: Dict[str, torch.Tensor], 
                    total_loss: torch.Tensor, 
                    batch: Dict, 
                    step_name: str):
        """
        一个可重用的辅助函数，用于记录所有损失。
        """
        
        # --- 修改: 根据谱图数据计算 bsz ---
        try:
            # 'peaks' 的形状是 [B, P_len, 2], 所以 bsz 是 dim 0
            bsz = batch['peaks'].size(0)
        except (AttributeError, KeyError, TypeError):
            bsz = 1 
        # --- 结束修改 ---

        is_train = step_name == 'train'
        on_step_logging = is_train  
        
        if is_train and self.trainer and self.trainer.log_every_n_steps > 0:
            log_step_check = (self.trainer.global_step + 1) % self.trainer.log_every_n_steps == 0
            
            if log_step_check and self.trainer.is_global_zero:
                # 格式化所有 loss
                loss_items = [f"{k}: {v.item():.4f}" for k, v in losses.items()]
                loss_string = " | ".join(loss_items)
                
                # 打印格式化的字符串
                self.print(f"\n--- Step {self.trainer.global_step + 1} Detailed Losses ---")
                self.print(loss_string)
                self.print("-" * (len(loss_string) + 4)) # 打印分隔符

        # 记录所有损失
        for k, v in losses.items():
            self.log(f'{step_name}_{k}', v.item(), 
                     on_step=on_step_logging, on_epoch=True, 
                     logger=True, batch_size=bsz, sync_dist=True)
            
        # 记录总损失
        self.log(f'{step_name}_loss_total', total_loss.item(), 
                 on_step=on_step_logging, on_epoch=True, 
                 prog_bar=True, logger=True, 
                 batch_size=bsz, sync_dist=True)
        
    
    @torch.no_grad()
    def decode_tokens_to_smiles(self, generated_tokens: torch.Tensor) -> List[str]:
        """
        一个辅助函数，用于将生成的 [N, max_seq_len] 的 token 解码为
        可变长度的SMILES字符串, 它会在 [EOS] 处停止。
        (MODIFIED: 简化为使用 batch_decode)
        """
        # 1. 惰性加载 (Lazy load) Tokenizer
        if self._tokenizer is None:
            if (hasattr(self, 'trainer') and 
                hasattr(self.trainer, 'datamodule') and 
                hasattr(self.trainer.datamodule, 'tokenizer')):
                
                self._tokenizer = self.trainer.datamodule.tokenizer
                # self._eos_token_id = self._tokenizer.eos_token_id # <-- 不再需要
                
                # if self._eos_token_id is None: # <-- 不再需要
                #     print("Warning: Tokenizer does not have an EOS token defined.")
                #     self._eos_token_id = -1
            else:
                return ["Tokenizer not available in pl_module. Cannot decode."] * len(generated_tokens)

        # 2. 批量解码
        # batch_decode 会自动处理 [PAD]、[EOS] 等特殊 token
        # skip_special_tokens=True 会:
        #   a) 移除 [CLS], [SEP], [PAD]
        #   b) 在第一个 [EOS] 处停止解码
        return self._tokenizer.batch_decode(
            generated_tokens, # .cpu() is handled internally by batch_decode if needed
            skip_special_tokens=True
        )
            
    
    
    def _safe_json_convert(self, data: Any) -> Any:
        """
        递归地将 Tensors / numpy 等转换为 Python 原生类型，
        方便 json.dump 序列化。
        """
        import numpy as np

        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy().tolist()
        if isinstance(data, np.ndarray):
            return data.tolist()
        if isinstance(data, dict):
            return {k: self._safe_json_convert(v) for k, v in data.items()}
        if isinstance(data, (list, tuple)):
            return [self._safe_json_convert(v) for v in data]
        return data
    
    
    # ---------------- Test 生命周期回调 ----------------
    def on_test_start(self) -> None:
        """
        Test loop 开始前：
        - 统一 timestamp（多次 test 复用同一个 timestamp 文件）
        - 创建输出根目录 self._test_output_dir
        - 保存 config（只保存一次）
        - 仅 rank0 打印一次输出目录
        """
        # -------- 1) root_dir：优先 logger.experiment.dir，其次 trainer.default_root_dir --------
        root_dir = None
        try:
            if hasattr(self.trainer, "logger") and self.trainer.logger is not None:
                exp = getattr(self.trainer.logger, "experiment", None)
                if exp is not None:
                    d = getattr(exp, "dir", None)
                    if isinstance(d, str):
                        root_dir = d
                    elif callable(d):
                        root_dir = d()
        except Exception:
            root_dir = None

        if root_dir is None:
            try:
                d = self.trainer.default_root_dir
                root_dir = d() if callable(d) else d
            except Exception:
                root_dir = "."

        if not isinstance(root_dir, str):
            root_dir = "."

        # -------- 2) timestamp 文件：多次 test 复用同一个 timestamp --------
        timestamp_file = os.path.join(root_dir, ".test_timestamp.txt")
        if not os.path.exists(timestamp_file):
            if getattr(self.trainer, "is_global_zero", False):
                ts = time.strftime("%Y%m%d-%H%M%S")
                try:
                    with open(timestamp_file, "w", encoding="utf-8") as f:
                        f.write(ts)
                except Exception:
                    pass

        # barrier
        try:
            self.trainer.strategy.barrier()
        except Exception:
            pass

        try:
            with open(timestamp_file, "r", encoding="utf-8") as f:
                timestamp = f.read().strip()
        except Exception:
            timestamp = time.strftime("%Y%m%d-%H%M%S")

        self.test_timestamp = timestamp

        # -------- 3) 输出根目录（本次 test 复用同一个 stamped_dir）--------
        base_dir = self.test_save_dir if os.path.isabs(self.test_save_dir) else os.path.join(root_dir, self.test_save_dir)
        stamped_dir = f"{base_dir}_{timestamp}"
        self._test_output_dir = stamped_dir
        os.makedirs(self._test_output_dir, exist_ok=True)

        # -------- 4) 只保存一次 config --------
        if getattr(self.trainer, "is_global_zero", False):
            self.print(f"[Test] Output root: {self._test_output_dir}")

            cfg_obj = getattr(self, "config", None)
            if isinstance(cfg_obj, dict):
                cfg_dict = cfg_obj
            else:
                try:
                    cfg_dict = vars(cfg_obj)
                except Exception:
                    cfg_dict = {"raw_config_str": str(cfg_obj)}

            cfg_safe = self._safe_json_convert(cfg_dict)
            cfg_path = os.path.join(self._test_output_dir, "config_test.json")
            if not os.path.exists(cfg_path):
                try:
                    with open(cfg_path, "w", encoding="utf-8") as f:
                        json.dump(cfg_safe, f, indent=2, ensure_ascii=False)
                except Exception:
                    pass


    @torch.no_grad()
    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        - 按 b_idx 建目录：<root>/b000123/
        - 目录下写 gt.json（只写一次；存在就跳过；写成功不打印）
        - 生成结果按 repeat_idx + rank + task 写入，避免 DDP 冲突
        """
        device = self.device
        rank = int(getattr(self.trainer, "global_rank", 0))

        # -------- 基本字段 --------
        peaks = batch["peaks"].to(device)
        B = int(peaks.size(0))

        spec_mask = batch.get("spec_mask")
        if isinstance(spec_mask, torch.Tensor):
            spec_mask = spec_mask.to(device)

        condition = batch.get("condition", {})
        cond_on_device: Dict[str, Any] = {}
        for k, v in condition.items():
            cond_on_device[k] = v.to(device) if isinstance(v, torch.Tensor) else v

        smiles_gt = batch.get("smiles")
        if smiles_gt is None:
            smiles_gt = [""] * B

        tokens = batch.get("tokens")
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.to(device)

        smiles_mask = batch.get("smiles_mask")
        if isinstance(smiles_mask, torch.Tensor):
            smiles_mask = smiles_mask.to(device)

        # -------- RepeatDataset 注入的身份字段：必须存在 --------
        repeat_idx = batch.get("repeat_idx")
        b_idx = batch.get("b_idx")
        if not isinstance(repeat_idx, torch.Tensor) or not isinstance(b_idx, torch.Tensor):
            raise KeyError(
                "batch must contain 'repeat_idx' and 'b_idx' tensors. "
                "Please ensure RepeatDataset adds them and collate_fn keeps them."
            )
        repeat_idx_cpu = repeat_idx.detach().cpu()
        b_idx_cpu = b_idx.detach().cpu()

        # -------- config: L_paths / n_smiles / n_peaks_generate --------
        def _get_cfg_section(cfg, name, default=None):
            if isinstance(cfg, dict):
                return cfg.get(name, default)
            return getattr(cfg, name, default)

        data_cfg = _get_cfg_section(self.config, "data", None)
        model_cfg = _get_cfg_section(self.config, "model", None)

        L_paths = int(_get_cfg_section(data_cfg, "L_paths", 1))
        n_smiles = int(_get_cfg_section(data_cfg, "max_seq_len", 256))
        n_peaks_generate = int(_get_cfg_section(model_cfg, "n_peaks_generate", 20))

        # -------- generate 输入 batch --------
        gen_batch: Dict[str, Any] = {
            "condition": cond_on_device,
            "peaks": peaks,
            "spec_mask": spec_mask,
            "smiles": smiles_gt,
        }
        if tokens is not None:
            gen_batch["tokens"] = tokens
        if smiles_mask is not None:
            gen_batch["smiles_mask"] = smiles_mask

        # ============================================================
        # 0) 解码 GT tokens（可选；只用于写 gt.json 的 decoded_smiles_from_tokens）
        # ============================================================
        decoded_gt_flat: Optional[List[str]] = None
        if tokens is not None:
            decoded_gt_flat = self.decode_tokens_to_smiles(tokens)  # list[str]

        # ============================================================
        # 1) 逐样本写 GT（只在 repeat_idx == 0 时写；按 b_idx 进目录）
        # ============================================================
        for i in range(B):
            rid = int(repeat_idx_cpu[i].item())
            bid = int(b_idx_cpu[i].item())

            # 只让 repeat_idx==0 写 GT（每个 b_idx 只写一次）
            if rid != 0:
                continue

            sample_dir = os.path.join(self._test_output_dir, f"b{bid:06d}")
            os.makedirs(sample_dir, exist_ok=True)

            gt_path = os.path.join(sample_dir, "gt.json")
            if os.path.exists(gt_path):
                continue  # 已存在就跳过（不打印）

            cond_i = {}
            for k, v in cond_on_device.items():
                if isinstance(v, torch.Tensor) and v.size(0) == B:
                    cond_i[k] = v[i]
                else:
                    cond_i[k] = v

            decoded_paths_i: Optional[List[str]] = None
            if decoded_gt_flat is not None:
                # 兼容：len(decoded_gt_flat) 可能是 B*L 或 B
                if len(decoded_gt_flat) == B * L_paths:
                    decoded_paths_i = decoded_gt_flat[i * L_paths: (i + 1) * L_paths]
                elif len(decoded_gt_flat) == B:
                    decoded_paths_i = [decoded_gt_flat[i]]

            gt_obj = {
                "meta": {
                    "b_idx": bid,
                    "repeat_idx": rid,
                },
                "sample": {
                    "smiles": smiles_gt[i],
                    "decoded_smiles_from_tokens": decoded_paths_i,
                    "peaks": self._safe_json_convert(peaks[i]),
                    "condition": self._safe_json_convert(cond_i),
                },
            }

            # 原子写：先写 tmp 再 replace，降低 DDP 写冲突风险
            tmp_path = gt_path + f".tmp_rank{rank:02d}"
            try:
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(self._safe_json_convert(gt_obj), f, indent=2, ensure_ascii=False)
                os.replace(tmp_path, gt_path)
            except Exception:
                try:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except Exception:
                    pass

        # ============================================================
        # 2) 生成并落盘（按 b_idx 目录；文件名含 repeat_idx + rank + task）
        # ============================================================
        for task in self.test_tasks:
            current_L = L_paths

            gen_output = self.generate(
                batch=gen_batch,
                n_smiles=n_smiles,
                n_peaks=n_peaks_generate,
                L=current_L,
                task=task,
                T=10,
            )

            # ----------- 解码 SMILES -----------
            smiles_tokens = gen_output["smiles_tokens"]  # [B*L, S]（按你的 generate 约定）
            decoded_flat: List[str] = self.decode_tokens_to_smiles(smiles_tokens)

            grouped_smiles: List[List[str]] = []
            if len(decoded_flat) == B * current_L:
                for i in range(B):
                    grouped_smiles.append(decoded_flat[i * current_L: (i + 1) * current_L])
            else:
                for i in range(B):
                    grouped_smiles.append([decoded_flat[i]] if i < len(decoded_flat) else [""])

            # ----------- 谱图 / 指纹 -----------
            peaks_gen = gen_output.get("peaks")
            if isinstance(peaks_gen, torch.Tensor):
                peaks_gen = peaks_gen.detach().cpu()

            fp_gen = gen_output.get("ecfp")
            if isinstance(fp_gen, torch.Tensor):
                fp_gen = fp_gen.detach().cpu()

            fp_scores = gen_output.get("fp_scores")
            if isinstance(fp_scores, torch.Tensor):
                fp_scores = fp_scores.detach().cpu()

            # token_scores list[T] each [B,L]（按你 generate 已经 cpu 的约定；不确定就转）
            token_scores_list = gen_output.get("token_scores")
            if token_scores_list is not None:
                fixed = []
                for s in token_scores_list:
                    fixed.append(s.detach().cpu() if isinstance(s, torch.Tensor) else s)
                token_scores_list = fixed

            # ----------- 逐样本落盘 -----------
            for i in range(B):
                rid = int(repeat_idx_cpu[i].item())
                bid = int(b_idx_cpu[i].item())
                sample_dir = os.path.join(self._test_output_dir, f"b{bid:06d}")
                os.makedirs(sample_dir, exist_ok=True)

                out_obj: Dict[str, Any] = {
                    "meta": {
                        "b_idx": bid,
                        "repeat_idx": rid,
                        "rank": rank,
                        "task": task,
                        "L_paths": int(current_L),
                    },
                    "generated": {
                        "smiles_paths": grouped_smiles[i],
                        "peaks": self._safe_json_convert(peaks_gen[i] if peaks_gen is not None else None),
                        "ecfp": self._safe_json_convert(
                            fp_to_onbit_indices(fp_gen[i]) if fp_gen is not None else None
                        ) if ("fp_to_onbit_indices" in globals() or hasattr(self, "fp_to_onbit_indices")) else self._safe_json_convert(fp_gen[i] if fp_gen is not None else None),
                        "fp_scores": self._safe_json_convert(fp_scores[i] if fp_scores is not None else None),
                    },
                }

                if token_scores_list is not None and len(token_scores_list) > 0 and isinstance(token_scores_list[0], torch.Tensor):
                    per_traj_scores: List[List[float]] = []
                    for step_scores in token_scores_list:
                        if isinstance(step_scores, torch.Tensor) and step_scores.dim() == 2 and step_scores.size(0) == B:
                            per_traj_scores.append(step_scores[i].tolist())
                    if len(per_traj_scores) > 0:
                        out_obj["generated"]["token_scores"] = per_traj_scores

                # 文件名：repeat_idx + rank + task（避免 DDP 同名覆盖）
                fname = f"gen_task-{task}_r{rid:04d}_rank{rank:02d}.json"
                out_path = os.path.join(sample_dir, fname)

                try:
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(self._safe_json_convert(out_obj), f, indent=2, ensure_ascii=False)
                except Exception as e:
                    if getattr(self.trainer, "is_global_zero", False):
                        self.print(f"[Test] Failed to save: {out_path} ({e})")

        return {}

def fp_to_onbit_indices(fp_row, thr: float = 0.5):
    """
    fp_row: Tensor[2048] or list
    Return: List[int]  (indices where bit==1)
    """
    if fp_row is None:
        return None

    # torch tensor
    if isinstance(fp_row, torch.Tensor):
        x = fp_row
        if x.numel() == 0:
            return []
        if x.dtype.is_floating_point:
            idx = (x > thr).nonzero(as_tuple=False).view(-1).tolist()
        else:
            idx = (x != 0).nonzero(as_tuple=False).view(-1).tolist()
        return [int(i) for i in idx]

    # python list (dense or already-sparse)
    if isinstance(fp_row, list):
        if len(fp_row) == 0:
            return []
        # already sparse indices
        if all(isinstance(v, int) for v in fp_row) and (max(fp_row) >= 2):
            return sorted(set(int(i) for i in fp_row))
        # dense 0/1 or float list
        return [i for i, v in enumerate(fp_row) if float(v) > thr]

    return None
