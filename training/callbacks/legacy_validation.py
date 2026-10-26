import json
import os
import random
from typing import Any, Dict, Optional, Tuple

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback


def _safe_json_convert(data: Any) -> Any:
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy().tolist()
    if isinstance(data, dict):
        return {k: _safe_json_convert(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_safe_json_convert(v) for v in data]
    return data


def _wandb_run_dir(trainer: pl.Trainer) -> Optional[str]:
    logger = getattr(trainer, "logger", None)
    exp = getattr(logger, "experiment", None)
    return getattr(exp, "dir", None)


def _first_val_batch(trainer: pl.Trainer):
    val_loader = None
    if getattr(trainer, "val_dataloaders", None):
        vls = trainer.val_dataloaders
        if isinstance(vls, (list, tuple)) and len(vls) > 0:
            val_loader = vls[0]
        else:
            val_loader = vls
    if val_loader is None and getattr(trainer, "datamodule", None) is not None:
        dm = trainer.datamodule
        if hasattr(dm, "val_dataloader"):
            val_loader = dm.val_dataloader()
    if not val_loader:
        return None
    try:
        return next(iter(val_loader))
    except StopIteration:
        return None


def _move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        elif isinstance(v, dict):
            out[k] = {kv: (vv.to(device) if isinstance(vv, torch.Tensor) else vv) for kv, vv in v.items()}
        else:
            out[k] = v
    return out


def _prepare_mp_log_batch(
    val_batch_device: Dict[str, Any],
    indices_to_log: torch.Tensor,
    l_paths: int,
) -> Tuple[Dict[str, Any], list, int]:
    batch_size = int(val_batch_device["peaks"].size(0))
    gt_smiles_list = [val_batch_device["smiles"][i] for i in indices_to_log.detach().cpu().tolist()]

    log_batch: Dict[str, Any] = {
        "condition": {k: v[indices_to_log] for k, v in val_batch_device["condition"].items()},
        "peaks": val_batch_device["peaks"][indices_to_log],
        "smiles": gt_smiles_list,
        "ecfp": val_batch_device["ecfp"][indices_to_log],
    }

    spec_mask = val_batch_device.get("spec_mask", None)
    log_batch["spec_mask"] = spec_mask[indices_to_log] if spec_mask is not None else None

    for k, v in val_batch_device.items():
        if isinstance(v, torch.Tensor) and k not in log_batch:
            log_batch[k] = v[indices_to_log]

    _, seq_len = val_batch_device["tokens"].shape
    smiles_tokens = val_batch_device["tokens"].view(batch_size, l_paths, seq_len)
    log_batch["tokens"] = smiles_tokens[indices_to_log].reshape(-1, seq_len)

    smiles_mask = val_batch_device.get("smiles_mask", None)
    if smiles_mask is not None:
        smiles_mask = smiles_mask.view(batch_size, l_paths, seq_len)
        log_batch["smiles_mask"] = smiles_mask[indices_to_log].reshape(-1, seq_len)
    else:
        log_batch["smiles_mask"] = None

    return log_batch, gt_smiles_list, int(indices_to_log.numel())


class WandbGenerationLogger(Callback):
    def __init__(self, num_samples_to_log: int = 3, save_dir: str = "validation_results"):
        super().__init__()
        self.num_samples_to_log = num_samples_to_log
        self.save_dir = save_dir

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if getattr(trainer, "global_rank", 0) != 0:
            return

        run_dir = _wandb_run_dir(trainer)
        if not run_dir:
            print("WandbGenerationLogger requires the WandbLogger.")
            return

        val_batch = _first_val_batch(trainer)
        if val_batch is None:
            print("Validation dataloader is empty or unavailable.")
            return

        try:
            val_batch = {k: v.to(pl_module.device) for k, v in val_batch.items()}
        except Exception as e:
            print(f"Error moving batch to device: {e}")
            return

        batch_size = int(val_batch["tokens"].size(0))
        num_to_select = min(self.num_samples_to_log, batch_size)
        indices_to_log = random.sample(range(batch_size), num_to_select)
        log_batch = {k: v[indices_to_log] for k, v in val_batch.items()}

        results = pl_module.predict_step(log_batch, batch_idx=0, num_samples=2)

        output_dir = os.path.join(run_dir, "smiles_generation")
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, f"epoch_{trainer.current_epoch}_step_{trainer.global_step}.json")

        try:
            with open(filepath, "w") as f:
                json.dump(results, f, indent=4)
            print(f"Saved validation generation results to {filepath}")
            print(f"CheckPointDir:{trainer.checkpoint_callback.best_model_path}")
        except Exception as e:
            print(f"Failed to save generation results to {filepath}: {e}")


class SMILESGenerationLogger(Callback):
    def __init__(self, num_samples_to_log: int = 4, save_dir: str = "generation_results", ckpt_path: str = ""):
        super().__init__()
        self.num_samples_to_log = num_samples_to_log
        self.save_dir = save_dir
        self.ckpt_path = ckpt_path
        self.tasks_to_run = ["joint", "spec_to_smiles", "smiles_to_spec", "spec_to_ecfp"]

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if getattr(trainer, "global_rank", 0) != 0:
            return

        run_dir = _wandb_run_dir(trainer)
        if not run_dir:
            print("SMILESGenerationLogger requires WandbLogger to work.")
            return

        val_batch = _first_val_batch(trainer)
        if val_batch is None:
            print("Validation dataloader is empty or unavailable.")
            return

        try:
            val_batch_device = _move_batch_to_device(val_batch, pl_module.device)
            batch_size = int(val_batch_device["peaks"].size(0))
        except Exception as e:
            print(f"Error preparing validation batch: {e}")
            return

        num_to_select = min(self.num_samples_to_log, batch_size)
        indices_to_log = torch.tensor(random.sample(range(batch_size), num_to_select), device=pl_module.device)
        l_paths = int(pl_module.hparams.config.data.L_paths)
        log_batch, gt_smiles_list, num_selected = _prepare_mp_log_batch(val_batch_device, indices_to_log, l_paths)

        all_results_json = {
            "ground_truth": {
                "smiles": gt_smiles_list,
                "condition": _safe_json_convert(log_batch["condition"]),
                "peaks": _safe_json_convert(log_batch["peaks"]),
            }
        }

        n_smiles = pl_module.hparams.config.data.max_seq_len
        n_peaks = pl_module.hparams.config.model.get("n_peaks_generate", 20)
        tokenizer = pl_module.trainer.datamodule.tokenizer

        print(f"\n[Rank {trainer.global_rank}] Logging {num_selected} samples for tasks: {self.tasks_to_run}...")
        for task in self.tasks_to_run:
            print(f"  - Running task: {task}...")
            gen_output = pl_module.generate(
                batch=log_batch,
                n_smiles=n_smiles,
                n_peaks=n_peaks,
                L=l_paths,
                task=task,
            )
            decoded_smiles = pl_module.decode_tokens_to_smiles(gen_output["smiles_tokens"])
            preview = tokenizer.batch_decode(gen_output["smiles_tokens"])
            if preview:
                print(f"    - Sample decoded SMILES (first 1): {preview[0]}")
            all_results_json[task] = {
                "smiles": decoded_smiles,
                "peaks": _safe_json_convert(gen_output["peaks"]),
            }

        output_dir = os.path.join(run_dir, self.save_dir)
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, f"epoch_{trainer.current_epoch}_step_{trainer.global_step}.json")

        try:
            with open(filepath, "w") as f:
                json.dump(all_results_json, f, indent=2)
            print(f"Saved validation generation results to {filepath}")
            print(f"CheckPointDir:{self.ckpt_path}")
        except Exception as e:
            print(f"Failed to save generation results to {filepath}: {e}")


class FlowTrajectoryLogger(Callback):
    def __init__(self, num_samples_to_log: int = 2, save_dir: str = "trajectories", record_every: int = 1):
        super().__init__()
        self.num_samples_to_log = num_samples_to_log
        self.save_dir = save_dir
        self.record_every = record_every

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if getattr(trainer, "global_rank", 0) != 0:
            return

        run_dir = _wandb_run_dir(trainer)
        if not run_dir:
            print("FlowTrajectoryLogger requires WandbLogger-like logger with .experiment.dir")
            return

        val_batch = _first_val_batch(trainer)
        if val_batch is None:
            print("Validation dataloader is empty or unavailable.")
            return

        try:
            val_batch_device = _move_batch_to_device(val_batch, pl_module.device)
            batch_size = int(val_batch_device["peaks"].size(0))
        except Exception as e:
            print(f"Error preparing validation batch: {e}")
            return

        num_to_select = min(self.num_samples_to_log, batch_size)
        indices_to_log = torch.tensor(random.sample(range(batch_size), num_to_select), device=pl_module.device)
        l_paths = int(pl_module.hparams.config.data.L_paths)
        log_batch, gt_smiles_list, num_selected = _prepare_mp_log_batch(val_batch_device, indices_to_log, l_paths)

        n_smiles = pl_module.hparams.config.data.max_seq_len
        n_peaks = pl_module.hparams.config.model.get("n_peaks_generate", 20)
        print(f"[FlowTrajectoryLogger] Logging {num_selected} samples (joint trajectory)...")

        traj_output = pl_module.cfm.generate_trajectory(
            batch=log_batch,
            n_smiles=n_smiles,
            n_peaks=n_peaks,
            L=l_paths,
            task="joint",
            record_every=self.record_every,
        )

        all_results_json = {
            "ground_truth": {
                "smiles": gt_smiles_list,
                "condition": _safe_json_convert(log_batch["condition"]),
                "tokens": _safe_json_convert(log_batch["tokens"]),
                "peaks": _safe_json_convert(log_batch["peaks"]),
                "ecfp": _safe_json_convert(log_batch["ecfp"]),
            },
            "joint": _safe_json_convert(traj_output),
        }

        output_dir = os.path.join(run_dir, self.save_dir)
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, f"trajectory_epoch_{trainer.current_epoch}_step_{trainer.global_step}.json")

        try:
            with open(filepath, "w") as f:
                json.dump(all_results_json, f, indent=2)
            print(f"[FlowTrajectoryLogger] Saved trajectory to {filepath}")
        except Exception as e:
            print(f"[FlowTrajectoryLogger] Failed to save trajectory: {e}")
