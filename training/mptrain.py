# mp_train.py
# 专用于 "unified_mp" (多路径SMILES) 任务的训练脚本

import os

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from pytorch_lightning.strategies import DDPStrategy
import wandb
import torch
import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from .callbacks.validation import SMILESGenerationLogger, FlowTrajectoryLogger
from easydict import EasyDict
torch.serialization.add_safe_globals([EasyDict])

# --- 导入 "unified_mp" 任务专属的模块 ---
from .core.lm_fm_mp import LitMeteorFlowMatcher
from .data_utils.dm_fp import METEORDataModule
from .utils.config_loader import load_config

# 设置 PyTorch 精度
torch.set_float32_matmul_precision('medium')

def train(config, args):
    """'unified_mp' 任务的主训练函数"""

    is_debug_mode = args.debug

    finetune_cfg = config.training.get('finetune', {})
    if finetune_cfg.get('enabled') and not (
        config.training.get('pretrained_path') or config.training.get('resume_path')
    ):
        raise ValueError('Fine-tuning requires --pretrained or training.resume_path.')
    
    # --- 调试模式重载 ---
    if is_debug_mode:
        print("="*50)
        print("           RUNNING IN DEBUG MODE")
        print("="*50)
        config.training.accelerator = 'auto'
        config.training.devices = 1
        config.training.strategy = 'auto'
        config.data.batch_size = 10 # 使用小批次
        config.data.num_workers_dataloader = 0 # 调试时必须为 0
        config.logging.offline_mode = True # 禁用 W&B
        config.data.L_paths = 1 # 调试时减少路径

    # 1. 初始化 W&B Logger
    group_name = os.path.splitext(os.path.basename(args.config))[0]
    shanghai_tz = ZoneInfo("Asia/Shanghai")
    now_in_shanghai = datetime.now(shanghai_tz)
    timestamp_str = now_in_shanghai.strftime("%Y%m%d%H%M%S")
    unique_run_name = f"{timestamp_str}"

    wandb_logger = WandbLogger(
        project=config.logging.project_name,
        name=unique_run_name,
        group=group_name,
        config=config, # 记录完整的配置
        offline=config.logging.offline_mode,
        save_dir=config.logging.save_dir,
    )
    
    # 创建检查点目录
    if config.logging and config.logging.checkpoint_dir:
        config.logging.checkpoint_dir = os.path.join(
            config.logging.checkpoint_dir, unique_run_name)
        os.makedirs(config.logging.checkpoint_dir, exist_ok=True)

    # 2. 初始化 DataModule (使用 data_module_mp)
    config.data.training_task = config.model.get('task', 'joint')  # 传递任务类型到 DataModule
    data_module = METEORDataModule(config.data)
    
    # --- 关键步骤 ---
    # 运行 setup() 来初始化 tokenizer 和 index2weight
    print("Setting up DataModule to calculate atom masses...")
    data_module.setup(stage='fit')


    
    # 将动态计算的 atom_masses 列表注入到模型配置中
    if hasattr(data_module, 'index2weight') and data_module.index2weight is not None:
        print("Injecting dynamically calculated atom_masses into model config.")
        config.model.atom_masses = data_module.index2weight.tolist()
        config.model.formula_list = data_module.index2formula.tolist()
        # 顺便检查和设置 vocab_size 和 pad_token_id
        config.model.vocab_size = data_module.tokenizer.vocab_size
        config.model.pad_token_id = data_module.tokenizer.pad_token_id
        config.model.mask_token_id = data_module.tokenizer.mask_token_id
    else:
        print("Warning: Could not find index2weight map in DataModule. Model will use static list from config.")

    # 3. 初始化 LightningModule (使用 LitMeteorFlowMatcher)
    pretrained_path = getattr(config.training, "pretrained_path", None)
    resume_path = getattr(config.training, "resume_path", None)

    if pretrained_path:
        # ================== 预训练初始化（新功能） ==================
        # 用当前 config 初始化新模型，然后把旧 ckpt 中能对得上的参数 load 进来
        print(f"[Init] Initializing new LitMeteorFlowMatcher model, then loading pretrained weights from: {pretrained_path}")
        lightning_model = LitMeteorFlowMatcher(config)

        # 1) 读取 ckpt
        ckpt = torch.load(pretrained_path, map_location="cpu")
        ckpt_state = ckpt.get("state_dict", ckpt)

        model_state = lightning_model.state_dict()
        remapped_state = {}
        remapped_legacy_gaussian = []
        gaussian_suffixes = (
            "centers",
            "log_sigma",
            "projection.weight",
            "projection.bias",
        )

        # HybridMzEncoder can remap legacy Gaussian keys in _load_from_state_dict,
        # but the exact-key filter below used to discard those keys before that
        # hook could see them. Remap first so warm-starting a Hybrid encoder really
        # reuses the learned Gaussian branch instead of silently randomizing it.
        for k, v in ckpt_state.items():
            target_key = k
            if target_key not in model_state:
                for suffix in gaussian_suffixes:
                    marker = f".{suffix}"
                    if k.endswith(marker):
                        candidate = f"{k[:-len(marker)]}.gaussian{marker}"
                        if candidate in model_state:
                            target_key = candidate
                            remapped_legacy_gaussian.append((k, candidate))
                        break
            remapped_state[target_key] = v

        filtered_state = {}
        skipped_shape_mismatch = []
        skipped_not_in_model = []

        for k, v in remapped_state.items():
            if k in model_state:
                if v.shape == model_state[k].shape:
                    filtered_state[k] = v
                else:
                    skipped_shape_mismatch.append((k, tuple(v.shape), tuple(model_state[k].shape)))
            else:
                skipped_not_in_model.append(k)

        print(f"[Init] Pretrained state_dict keys: {len(ckpt_state)}")
        print(f"[Init] Matched & loaded keys    : {len(filtered_state)}")
        print(f"[Init] Remapped legacy Gaussian: {len(remapped_legacy_gaussian)}")
        print(f"[Init] Skipped (not in model)   : {len(skipped_not_in_model)}")
        print(f"[Init] Skipped (shape mismatch) : {len(skipped_shape_mismatch)}")
        for old_name, new_name in remapped_legacy_gaussian[:20]:
            print(f"        - {old_name} -> {new_name}")

        # 2) 加载过滤后的权重
        missing_before = set(model_state.keys()) - set(filtered_state.keys())
        lightning_model.load_state_dict(filtered_state, strict=False)
        missing_after = set(lightning_model.state_dict().keys()) - set(filtered_state.keys())

        if missing_after:
            print(f"[Init] Remaining missing keys in model (initialized with default/random): {len(missing_after)}")
            # 你想看的话也可以 print 前几条:
            for name in list(missing_after)[:20]:
                print(f"        - {name}")
        if skipped_shape_mismatch:
            print("[Init] Example of shape-mismatch keys (old_shape -> new_shape):")
            for name, old_shape, new_shape in skipped_shape_mismatch[:20]:
                print(f"        - {name}: {old_shape} -> {new_shape}")

    elif resume_path:
        # ================== 原生 resume（严格恢复训练） ==================
        print(f"[Init] Resuming training from checkpoint: {resume_path}")
        lightning_model = LitMeteorFlowMatcher.load_from_checkpoint(
            checkpoint_path=resume_path,
        )
    else:
        # ================== 从零初始化 ==================
        print("[Init] Initializing new LitMeteorFlowMatcher model from scratch.")
        lightning_model = LitMeteorFlowMatcher(config)
    
    # 4. 配置回调
    if is_debug_mode:
        callbacks_list = [
            SMILESGenerationLogger(num_samples_to_log=2, ckpt_path=config.logging.checkpoint_dir),
            FlowTrajectoryLogger(num_samples_to_log=10),
        ]
    else:
        callbacks_list = [
            LearningRateMonitor(logging_interval='step'),
            ModelCheckpoint(
                dirpath=config.logging.checkpoint_dir,
                filename=f'{{step:07d}}-{{val_loss_total:.4f}}',
                save_top_k=config.logging.save_top_k,
                verbose=True,
                monitor=config.logging.monitor,
                mode=config.logging.mode,
                save_last=config.logging.save_last,
            ),
            EarlyStopping(
                monitor=config.logging.monitor,
                patience=config.logging.early_stopping_patience,
                verbose=True,
                mode=config.logging.mode,
            ),
            SMILESGenerationLogger(num_samples_to_log=2, ckpt_path=config.logging.checkpoint_dir),
        ]

    # 5. 初始化 Trainer
    trainer_args = {
        'accelerator': config.training.accelerator,
        'devices': config.training.devices,
        'logger': wandb_logger,
        'callbacks': callbacks_list,
    }

    if is_debug_mode:
        print("Running debug mode with 10 epochs, 2 train/val batches to trigger callbacks.")
        trainer_args.update({
            'max_epochs': 10,
            'strategy': 'auto',
            'log_every_n_steps': 1,
            'limit_train_batches': 2, # 运行 2 个训练批次
            'limit_val_batches': 2,   # 运行 2 个验证批次
            'val_check_interval': 1.0, # 每个 epoch 结束时验证
        })
    else:
        trainer_args.update({
            'max_epochs': config.training.num_epochs,
            'strategy': DDPStrategy(find_unused_parameters=False), # DDP
            'log_every_n_steps': config.training.log_every_n_steps,
            'accumulate_grad_batches': int(
                getattr(config.training, 'accumulate_grad_batches', 1) or 1
            ),
            'val_check_interval': config.training.check_val_every_n_steps if config.training.check_val_every_n_steps is not None else 1.0,
            'check_val_every_n_epoch': config.training.check_val_every_n_epochs,
            # 'detect_anomaly': config.training.get('detect_anomaly', True),
        })
        # 梯度裁剪(可选)：热启动/归一化迁移初期谱头梯度较大，裁剪可保护 backbone。
        # 未在配置中设置则为 0(=不裁剪)，保持对既有配置的向后兼容。
        grad_clip = float(getattr(config.training, 'gradient_clip_val', 0.0) or 0.0)
        if grad_clip > 0.0:
            trainer_args['gradient_clip_val'] = grad_clip

        # 配置了 category_ratio(按谱/分子/paired 配比采样) 时，禁用 Lightning 自动 DDP sampler，
        # 改用 DataModule 里 DDP 感知的 RatioSampler(否则会被替换成 DistributedSampler 丢掉配比)。
        if getattr(config.data, 'category_ratio', None):
            trainer_args['use_distributed_sampler'] = False
            print("[Trainer] 检测到 data.category_ratio → use_distributed_sampler=False (使用 RatioSampler)")
    
    trainer = pl.Trainer(**trainer_args)
    
    
    if args.test:
        print("\n[TEST MODE] Skipping training and running test on the test set...")
        trainer.test(model=lightning_model, datamodule=data_module)
        print("Test finished. JSON shards are saved under test_save_dir.")
        if not is_debug_mode:
            wandb.finish()
        return

    print("\nStarting training for 'unified_mp' task...")
    trainer.fit(model=lightning_model, datamodule=data_module)
    print("Training finished.")
    if not is_debug_mode:
        wandb.finish()



if __name__ == "__main__":
    default_config_path = os.path.join(
        os.path.dirname(__file__),
        'configs',
        'mptrain.yaml',
    )
    parser = argparse.ArgumentParser(description="Train / Test the METEOR Multi-Path (MP) model.")
    parser.add_argument(
        '--config',
        type=str,
        default=default_config_path,
        help="Path to the YAML configuration file for the 'unified_mp' task."
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help="Run in debug mode: CPU, small batches, no DDP, no online logging."
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help="If set, skip training and only run test() on the test set to generate JSON results."
    )
    parser.add_argument('--data-root', help='Directory containing the raw and processed datasets.')
    parser.add_argument('--tokenizer', help='Path to the tokenizer directory.')
    parser.add_argument('--output-dir', help='Directory for logs, checkpoints, and generated files.')
    parser.add_argument('--pretrained', help='Checkpoint used to initialize model weights.')
    args = parser.parse_args()

    config = load_config(args.config)
    working_root = Path.cwd()

    def resolve_path(value):
        path = Path(value).expanduser()
        return str(path if path.is_absolute() else working_root / path)

    for key in ('zinc_root', 'graph_massgym_root', 'spectrum_root'):
        config.data[key] = resolve_path(args.data_root or config.data[key])
    config.data.tokenizer_path = resolve_path(args.tokenizer or config.data.tokenizer_path)
    config.model.tokenizer_path = config.data.tokenizer_path
    output_dir = resolve_path(args.output_dir or 'runs')
    config.logging.save_dir = output_dir
    config.logging.checkpoint_dir = str(Path(output_dir) / 'checkpoints' / config.model.task)
    config.test.save_dir = str(Path(output_dir) / 'test' / config.model.task)
    if args.pretrained:
        config.training.pretrained_path = resolve_path(args.pretrained)
    elif config.training.pretrained_path:
        config.training.pretrained_path = resolve_path(config.training.pretrained_path)
    if config.training.resume_path:
        config.training.resume_path = resolve_path(config.training.resume_path)
    train(config, args)
