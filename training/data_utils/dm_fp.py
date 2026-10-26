# -*- coding: utf-8 -*-
# data_module_mp.py
"""
PyTorch Lightning DataModule for the "unified_mp" (Unified Multi-Path) task.

This module loads data from the multi-path compatible datasets:
- MSMolDataset (from graph_massgym_multipath)
- ZINC_PyG_LMDB_Dataset (from zinc100m_multipath)
- MassSpecLmdbDataset (from spectrum_multipath)

It uses a custom collate function to batch:
- SMILES tokens into [B*L, S_len]
- Spectrum peaks into [B, P_len, 2]
- Conditions into a dictionary of tensors [B, ...]
"""

import os
from pathlib import Path
import token
from typing import Dict, List, Optional, Any
import pytorch_lightning as pl
from sympy import im
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from transformers import AutoTokenizer
from rdkit import Chem
from rdkit.Chem import Descriptors
from tqdm import tqdm

# --- 导入新的 "multipath" (mp) 数据集 ---
from .datasets.ms_mp import MassSpecLmdbDataset
from .datasets.massgym_mp import MSMolDataset
from .datasets.zinc100m_mp import ZINC_LMDB_Dataset
from molspecflow.formula_utils import smiles_to_compact_vector, VECTOR_SIZE
import numpy as np

# =============================================================================
# 1. 辅助函数 (Token-Mass Map)
# =============================================================================

def _calculate_token_mass(token_str: str) -> float:
    """Calculates the monoisotopic mass of a single SMILES token string using RDKit."""
    try:
        # 尝试将 token 视为一个 SMILES 片段
        mol = Chem.MolFromSmiles(token_str)
        if mol is None:
            # 某些 token (如 '[H+]') 不是有效的 Mol，但有意义
            if token_str == '[H+]':
                return 1.007276
            elif token_str == '[Na+]':
                return 22.989769
            else:
                # 其他特殊情况
                return 0.0
        return Descriptors.ExactMolWt(mol)
    except:
        return 0.0

def create_index2weight_map(tokenizer):
    """Dynamically creates a token-ID-to-mass map for a given tokenizer."""
    vocab = tokenizer.get_vocab()
    vocab_size = len(vocab)
    weights_list = [0.0] * vocab_size
    print("Creating index-to-weight map from tokenizer vocabulary...")
    
    # 过滤掉特殊 tokens，它们不应该有质量
    special_tokens_set = set(tokenizer.all_special_tokens)
    
    filtered_vocab = {t: i for t, i in vocab.items() if t not in special_tokens_set}
    
    for token, token_id in tqdm(filtered_vocab.items(), desc="Calculating Token Masses"):
        weights_list[token_id] = _calculate_token_mass(token)
        
    return torch.tensor(weights_list, dtype=torch.float32)


def create_index2formula_map(tokenizer):
    """
    动态为给定的 tokenizer 创建 Token-ID 到原子计数向量的映射表。
    
    Returns:
        torch.Tensor: 形状为 [vocab_size, 28] 的 LongTensor。
    """
    vocab = tokenizer.get_vocab()
    vocab_size = len(vocab)
    
    # 使用 np.int64 初始化
    formula_matrix = np.zeros((vocab_size, VECTOR_SIZE), dtype=np.int64)
    
    print(f"Creating index-to-formula map (Dtype: torch.long, Dim: {vocab_size}x{VECTOR_SIZE})...")
    
    special_tokens_set = set(tokenizer.all_special_tokens)
    
    for token, token_id in tqdm(vocab.items(), desc="Processing Chemical Tokens"):
        if token in special_tokens_set:
            continue
            
        formula_matrix[token_id] = smiles_to_compact_vector(token)
        
    # 转换为 torch.long (即 torch.int64)
    return torch.tensor(formula_matrix, dtype=torch.long)
# =============================================================================
# 2. PyTorch Lightning DataModule
# =============================================================================

class METEORDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        # config 应该包含所有参数，例如:
        # hparams = {
        #   "task": "unified_mp",
        #   "data_root": "/path/to/data",
        #   "zinc_root": "/path/to/zinc",
        #   "graph_massgym_root": "/path/to/graph_massgym",
        #   "spectrum_root": "/path/to/spectrum",
        #   "tokenizer_path": "/path/to/tokenizer",
        #   "batch_size": 32,
        #   "num_workers_dataloader": 8,
        #   "num_workers_preprocess": 16,
        #   "L_paths": 4,
        #   "max_seq_len": 256
        # }
        self.save_hyperparameters(config)
        
        self.tokenizer = None
        self.index2weight = None
        self.index2formula = None
        self.training_task = config.training_task
        # 从 hparams 获取 dataloader worker 数量
        self.num_workers_dataloader = self.hparams.get("num_workers_dataloader", os.cpu_count() // 2)
        
        # 确认 task
        self.task = self.hparams.get("task", "unified_mp")
        if self.task != "unified_mp":
            raise ValueError(f"This DataModule is configured only for 'unified_mp' task, but got {self.task}")

    def setup(self, stage: str):
        """
        设置 'unified_mp' 任务所需的数据集。
        """
        
        # 1. 初始化 Tokenizer 和 mass map (所有进程都需要)
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.hparams.tokenizer_path, trust_remote_code=True)
            self.pad_token_id = self.tokenizer.pad_token_id
            
        if self.index2weight is None:
            # index2weight map 对于 MeteorFlowModel 是必需的
            # 我们在这里创建它，并将其传递给 collate_fn
            self.index2weight = create_index2weight_map(self.tokenizer).to('cpu')
        if self.index2formula is None:
            self.index2formula = create_index2formula_map(self.tokenizer).to('cpu')
        

        # 2. 获取预处理和数据集参数
        num_workers_preprocess = self.hparams.get("num_workers_preprocess", os.cpu_count())
        
        # 这是 'unified_mp' 任务独有的参数
        dataset_args = {
            "L": self.hparams.L_paths,
            "max_seq_len": self.hparams.max_seq_len,
            "tokenizer_path": self.hparams.tokenizer_path,
            # "pad_token_id": self.pad_token_id
        }

        print(f"--- [DataModule] Setting up stage: {stage} for task '{self.task}' ---")

        if stage in ('fit', None):
            # --- 实例化训练数据集 ---
            print(f"Loading Graph-MassGym (multipath) from: {self.hparams.graph_massgym_root}")
            graph_gym_train = MSMolDataset(
                root=self.hparams.graph_massgym_root,
                data_split='train',
                num_workers=num_workers_preprocess,
                **dataset_args
            )
            
            val_graph_gym = MSMolDataset(
                root=self.hparams.graph_massgym_root,
                data_split='val', # 验证集使用 'val' split
                num_workers=num_workers_preprocess,
                **dataset_args
            )

            
            
            if self.training_task == 'joint':
                print(f"Loading ZINC (multipath) from: {self.hparams.zinc_root}")
                zinc_train = ZINC_LMDB_Dataset(
                    root=self.hparams.zinc_root,
                    split='train',
                    num_workers=num_workers_preprocess,
                    **dataset_args
                )
                print(f"Loading Spectrum (multipath) from: {self.hparams.spectrum_root}")
                spec_train = MassSpecLmdbDataset(
                    root=self.hparams.spectrum_root,
                    split='train',
                    num_workers=num_workers_preprocess,
                    **dataset_args
                )
                self.train_datasets_list = [graph_gym_train, spec_train, zinc_train]
            elif self.training_task in ['spec_to_smiles', 'spec_to_ecfp', 'smiles_to_spec']:
                self.train_datasets_list = [graph_gym_train]
            else:
                raise ValueError(f"Unsupported training task: {self.training_task}")
            self.train_dataset = ConcatDataset(self.train_datasets_list)

            # --- 按“谱图 / 分子 / paired”三类给每个子数据集打标签，供 category_ratio 平衡采样 ---
            #   paired  : 分子+谱图配对(graph_massgym train)
            #   spectrum: 有谱图(spec LMDB、graph_massgym 的 'ms' 辅助 split)
            #   molecule: 仅分子(ZINC、graph_massgym 的 'mol' 辅助 split)
            _cat_of = {
                id(graph_gym_train): 'paired',
                **({id(spec_train): 'spectrum', id(zinc_train): 'molecule'}
                   if self.training_task == 'joint' else {}),
            }
            self.category_ranges_map = {}
            _off = 0
            for _ds in self.train_datasets_list:
                _tag = _cat_of.get(id(_ds), 'paired')
                _n = len(_ds)
                self.category_ranges_map.setdefault(_tag, []).append((_off, _off + _n))
                _off += _n
            _sizes = {c: sum(e - s for (s, e) in r) for c, r in self.category_ranges_map.items()}
            print(f"[DataModule] 训练集类别规模: {_sizes}  (category_ratio={self.hparams.get('category_ratio', None)})")

            self.val_dataset = val_graph_gym
            


        
        elif stage in ('test', None):
            # --- 实例化测试数据集 ---
            test_graph_gym = MSMolDataset(
                root=self.hparams.graph_massgym_root,
                data_split='test', # 测试集使用 'test' split
                num_workers=num_workers_preprocess,
                **dataset_args
            )
            self.test_dataset = test_graph_gym
            self.test_dataset = RepeatDataset(self.test_dataset, repeats=1000)

    def _ddp_info(self):
        """从 trainer 拿 (world_size, global_rank)；未在 DDP 下则 (1,0)。"""
        try:
            if self.trainer is not None:
                return int(getattr(self.trainer, 'world_size', 1) or 1), int(getattr(self.trainer, 'global_rank', 0) or 0)
        except Exception:
            pass
        return 1, 0

    def _resolve_samples_per_epoch(self, cats, ranges) -> int:
        """决定 RatioSampler 每个 epoch 的总采样数。

        - 未配置 / null / 'auto' / 'full' → 用 ConcatDataset 全集规模(Σ各类别样本数)。
          仍按 category_ratio 做有放回配比采样(如 1:1:1)；epoch 边界只影响 val/ckpt 节奏。
        - 显式整数 → 人为缩短 epoch(调试或快速迭代)。
        """
        spe = self.hparams.get('samples_per_epoch', None)
        cat_sizes = {c: sum(e - s for (s, e) in r) for c, r in zip(cats, ranges)}
        full = sum(cat_sizes.values())
        if spe is None or spe == '' or str(spe).lower() in ('auto', 'full'):
            print(
                f"[DataModule] samples_per_epoch=全集 {full:,} "
                f"(Σ类别 {cat_sizes})；RatioSampler 仍按 category_ratio 有放回采样"
            )
            return full
        total = int(spe)
        if total <= 0:
            print(f"[DataModule] samples_per_epoch={spe!r} 无效，回退全集 {full:,}")
            return full
        print(
            f"[DataModule] samples_per_epoch={total:,} (人为上限；全集={full:,}，Σ类别 {cat_sizes})"
        )
        return total

    def train_dataloader(self):
        ratio_cfg = self.hparams.get('category_ratio', None)
        # 未配置 category_ratio → 保持原有 shuffle 行为(按各数据集自然规模配比)
        if not ratio_cfg or not getattr(self, 'category_ranges_map', None):
            return DataLoader(
                self.train_dataset,
                batch_size=self.hparams.batch_size,
                shuffle=True,
                collate_fn=self.collate_fn,
                num_workers=self.num_workers_dataloader,
                persistent_workers=self.num_workers_dataloader > 0,
                pin_memory=True
            )
        # 配置了 category_ratio → 用 DDP 感知的按类别配比采样器(需 trainer use_distributed_sampler=False)
        cats = [c for c in ['spectrum', 'molecule', 'paired'] if c in self.category_ranges_map]
        ranges = [self.category_ranges_map[c] for c in cats]
        weights = [float(ratio_cfg.get(c, 0.0)) for c in cats]
        nr, rk = self._ddp_info()
        total = self._resolve_samples_per_epoch(cats, ranges)
        self._train_sampler = RatioSampler(
            ranges, weights, total, num_replicas=nr, rank=rk,
            seed=int(self.hparams.get('sampler_seed', 0) or 0),
        )
        print(f"[DataModule] 按配比采样: {dict(zip(cats, weights))} | rank={rk}/{nr} | 每卡每epoch={self._train_sampler.num_samples}")
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            sampler=self._train_sampler,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers_dataloader,
            persistent_workers=self.num_workers_dataloader > 0,
            pin_memory=True
        )

    def val_dataloader(self):
        # 关闭了 use_distributed_sampler 时(配了 category_ratio)，val 需手动做 DDP 分片，避免各卡重复评测。
        sampler = None
        shuffle = True
        if self.hparams.get('category_ratio', None):
            nr, rk = self._ddp_info()
            if nr > 1:
                from torch.utils.data.distributed import DistributedSampler
                sampler = DistributedSampler(self.val_dataset, num_replicas=nr, rank=rk, shuffle=True)
                shuffle = False
        return DataLoader(
            self.val_dataset,
            shuffle=shuffle,
            sampler=sampler,
            batch_size=self.hparams.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers_dataloader,
            persistent_workers=self.num_workers_dataloader > 0,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers_dataloader,
            persistent_workers=self.num_workers_dataloader > 0,
            pin_memory=True,
            shuffle=False,
        )
            
    def predict_dataloader(self):
        return self.test_dataloader()

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """        
        Input batch: List[Dict], 每个样本包含:
            - 'tokens': Tensor [L, S_len] (L 通常为增强/重复次数)
            - 'ms': Dict {
                'precursor_mz', 'adduct_weight', 'is_dummy', 'charge', 'energy': Tensor[1]
                'mz', 'intensity': Tensor[P_len_i]
            }
            - 'ecfp': Tensor [2048]
            - 'smiles': str
            - 'formula_vector': Tensor [V] (Optional)
            - 'mol_prior': Tensor [L, D] (Optional)
            - 'tokens_is_dummy': bool (Optional)
            
        Output batch: Dict:
            - 'tokens': Tensor [B*L, S_len]
            - 'smiles_mask': Tensor [B*L, S_len] (bool)
            - 'ecfp': Tensor [B, 2048]
            - 'peaks': Tensor [B, P_len_max, 2] (mz, intensity)
            - 'spec_mask': Tensor [B, P_len_max] (bool)
            - 'condition': Dict {
                'precursor_mz', 'adduct_weight', 'is_dummy', 'charge', 'energy': Tensor[B]
                'formula_vector': Tensor[B, V] (Optional)
            }
            - 'smiles': List[str] (length B)
            - 'smiles_is_dummy': Tensor [B*L] (bool)
            - 'spec_is_dummy': Tensor [B] (bool)
        """
        if not batch: 
            return None
        
        B = len(batch)
        L = batch[0]['tokens'].shape[0]
        output_batch = {}

        # --- 1. 数据解包 (一次遍历提高效率) ---
        tokens_list = []
        ecfp_list = []
        ms_list = []
        smiles_list = []
        smiles_dummy_list = []
        
        for s in batch:
            tokens_list.append(s['tokens'])
            ecfp_list.append(s['ecfp'])
            ms_list.append(s['ms'])
            smiles_list.append(s.get('smiles', ""))
            # 默认 tokens_is_dummy 为 True 如果不存在
            smiles_dummy_list.append(s.get('tokens_is_dummy', True))

        # --- 2. 处理 Tokens & ECFP ---
        # 直接 stack 并 reshape 为 [B*L, S_len]
        output_batch['tokens'] = torch.stack(tokens_list, dim=0).view(B * L, -1)
        output_batch['ecfp'] = torch.stack(ecfp_list, dim=0)
        
        # 根据业务逻辑，目前设为全 True 掩码
        output_batch['smiles_mask'] = torch.ones_like(output_batch['tokens'], dtype=torch.bool)
        
        # 处理重复/增强后的 dummy 标记: [B] -> [B*L]
        output_batch['smiles_is_dummy'] = torch.tensor(smiles_dummy_list, dtype=torch.bool).repeat_interleave(L)

        # --- 3. 处理质谱数据 (Peaks) ---
        # 填充 mz 和 intensity 到最大长度 [B, P_len_max]
        padded_mz = torch.nn.utils.rnn.pad_sequence(
            [ms['mz'] for ms in ms_list], batch_first=True, padding_value=0.0
        )
        padded_intensity = torch.nn.utils.rnn.pad_sequence(
            [ms['intensity'] for ms in ms_list], batch_first=True, padding_value=0.0
        )
        
        # 合并为 [B, P_len_max, 2] 并生成掩码
        output_batch['peaks'] = torch.stack([padded_mz, padded_intensity], dim=-1)
        output_batch['spec_mask'] = (padded_mz != 0.0)

        # --- 4. 处理条件字段 (Conditions) ---
        cond_keys = ['precursor_mz', 'adduct_weight', 'is_dummy', 'charge', 'energy']
        condition_batch = {
            k: torch.stack([ms[k] for ms in ms_list]).squeeze(-1) 
            for k in cond_keys
        }
        
        # 修正 energy 中的 nan 值并同步 dummy 状态
        condition_batch['energy'] = torch.nan_to_num(condition_batch['energy'], nan=0.0)
        output_batch['condition'] = condition_batch
        output_batch['spec_is_dummy'] = condition_batch['is_dummy']

        # --- 5. 处理可选字段 (Optional Fields) ---
        if 'smiles' in batch[0].keys():
            output_batch['smiles'] = smiles_list

        if 'formular_vector' in batch[0].keys():
            output_batch['condition']['formular_vector'] = torch.stack(
                [s['formular_vector'] for s in batch], dim=0
            )

        if 'mol_prior' in batch[0].keys():
            # [B, L, D] -> [B*L, D]
            output_batch['mol_prior'] = torch.stack(
                [s['mol_prior'] for s in batch], dim=0
            ).view(B * L, -1)

        if 'repeat_idx' in batch[0].keys():
            output_batch['repeat_idx'] = torch.tensor([s['repeat_idx'] for s in batch], dtype=torch.long)
            output_batch['b_idx'] = torch.tensor([s['b_idx'] for s in batch], dtype=torch.long)

        return output_batch



class RatioSampler(torch.utils.data.Sampler):
    """按类别配比(with replacement)采样，DDP 感知、O(1) 内存(分块生成)。

    category_ranges  : List[List[(start,end)]]，每个类别在 ConcatDataset 中占用的若干区间。
    category_weights : List[float]，各类别的采样配比(会归一化)。1:1:1 即三类等概率。
    每次 __iter__ 用 (seed + epoch*大素数 + rank) 播种，保证不同 epoch/不同卡取到不同样本。
    """
    def __init__(self, category_ranges, category_weights, num_samples,
                 num_replicas=1, rank=0, seed=0):
        self.category_ranges = category_ranges
        w = torch.tensor(category_weights, dtype=torch.double)
        assert float(w.sum()) > 0, "category_weights 之和必须 > 0"
        self.category_probs = w / w.sum()
        self.num_replicas = max(1, int(num_replicas))
        self.rank = int(rank)
        self.total = int(num_samples)
        self.num_samples = self.total // self.num_replicas   # 每卡样本数
        self.seed = int(seed)
        self.epoch = 0
        # 预计算每类的区间起点/前缀，用于把“类内偏移”向量化映射回全局索引
        self._starts, self._prev, self._bounds, self._size = [], [], [], []
        for ranges in category_ranges:
            sizes = torch.tensor([e - s for (s, e) in ranges], dtype=torch.long)
            self._size.append(int(sizes.sum()))
            self._starts.append(torch.tensor([s for (s, e) in ranges], dtype=torch.long))
            self._bounds.append(sizes.cumsum(0))
            self._prev.append(torch.cat([torch.zeros(1, dtype=torch.long), sizes.cumsum(0)[:-1]]))

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch * 1000003 + self.rank)
        self.epoch += 1  # 下个 epoch 自动换种子
        remaining = self.num_samples
        chunk = 131072
        while remaining > 0:
            n = min(chunk, remaining)
            cats = torch.multinomial(self.category_probs, n, replacement=True, generator=g)
            out = torch.empty(n, dtype=torch.long)
            for ci in range(len(self.category_ranges)):
                mask = cats == ci
                k = int(mask.sum())
                if k == 0:
                    continue
                offs = torch.randint(0, self._size[ci], (k,), generator=g)
                ridx = torch.bucketize(offs, self._bounds[ci], right=True)
                out[mask] = self._starts[ci][ridx] + offs - self._prev[ci][ridx]
            for v in out.tolist():
                yield v
            remaining -= n


class RepeatDataset(Dataset):
    def __init__(self, base: Dataset, repeats: int):
        self.base = base
        self.repeats = int(repeats)
        self.n = len(base)

    def __len__(self):
        return self.n * self.repeats

    def __getitem__(self, idx: int):
        ridx = idx // self.n          # 0..repeats-1
        bidx = idx % self.n
        batch = self.base[bidx]
        # 关键：把 repeat_idx 带出来
        if isinstance(batch, dict):
            batch = dict(batch)
            batch["repeat_idx"] = ridx
            batch['b_idx']      = bidx
        return batch
