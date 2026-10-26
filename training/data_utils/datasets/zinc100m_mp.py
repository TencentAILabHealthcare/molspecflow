import os
import pickle
from pathlib import Path
from typing import List, Optional, Dict

import lmdb
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm
from multiprocessing import cpu_count
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem
import argparse
import platform
import deepsmiles
from transformers import AutoTokenizer
import shutil
from molspecflow.formula_utils import smiles_to_compact_vector
from rdkit.Chem.Scaffolds import MurckoScaffold

# Disable RDKit console logging to keep the output clean
RDLogger.DisableLog('rdApp.*')

def smiles_to_deepsmiles(smi: str) -> str:
    conv = deepsmiles.Converter(rings=True, branches=True)
    return conv.encode(smi)

def get_scaffold_deepsmiles(mol: Chem.Mol) -> str:
    scaf = MurckoScaffold.GetScaffoldForMol(mol)
    if scaf is None or scaf.GetNumAtoms() == 0:
        return ""
    scaf_smi = Chem.MolToSmiles(scaf, canonical=True, isomericSmiles=False, kekuleSmiles=True, allHsExplicit=True)
    return smiles_to_deepsmiles(scaf_smi)

class ZINC_LMDB_Dataset(Dataset):
    """
    ZINC 100M 预处理 Dataset（train/valid/test 已分好 CSV）：

    预处理阶段：
      - 从 <root>/raw/zinc100m/{train,valid,test}.csv 流式读取 'smiles' 列
      - 不预先统计总行数，不做 RDKit 解析，不 canonical
      - 对每个 split 单独构建 LMDB：
          key   = idx(10位零填充，0..N-1，无空洞)
          value = 原始 SMILES UTF-8 编码
      - 若设定 sample_size，则只对 train split 截断前 sample_size 条

    取样阶段 (__getitem__)：
      - 从 LMDB 读 SMILES
      - RDKit 解析 + on-the-fly 生成 L 条 DeepSMILES 多路径 SMILES
      - tokenizer 编码成 [L, max_seq_len] tokens
      - 生成 dummy MS + 2048 维 ECFP

    外部接口保持不变：
      __init__(root, split='train', tokenizer_path=None, L=1,
               max_seq_len=128, num_workers=None, sample_size=-1)

      __getitem__ 返回:
        {
          'ms': dummy_ms,
          'smiles': canonical_smi,
          'tokens': tokens_tensor,        # [L, max_seq_len]
          'tokens_is_dummy': tensor([0]),
          'ecfp': ecfp_tensor             # [2048]
        }
    """

    @property
    def processed_dir(self) -> str:
        return str(self.processed_dir_base / self.split)

    @property
    def processed_file_names(self) -> List[str]:
        return ['data.lmdb']

    def __init__(
        self,
        root: str,
        split: str = 'train',
        tokenizer_path: str = None,
        L: int = 1,
        max_seq_len: int = 128,
        num_workers: Optional[int] = None,
        sample_size: Optional[int] = -1
    ):
        super().__init__()

        root_path = Path(root).resolve()
        csv_data_path = root_path / 'raw' / 'zinc100m'
        self.csv_data_path = csv_data_path.resolve()

        self.num_workers = num_workers if num_workers is not None else cpu_count()
        # sample_size <=0 或 None => 全量
        self.sample_size = sample_size if (sample_size is not None and sample_size > 0) else None

        if tokenizer_path is None:
            raise ValueError("A tokenizer_path must be provided.")
        self.tokenizer_path = tokenizer_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        self.L = L
        self.max_seq_len = max_seq_len

        # DeepSMILES 转换器放在 __init__，避免每次 __getitem__ 初始化
        self.deepsmiles_converter = deepsmiles.Converter(rings=True, branches=True)

        # 根据 sample_size 区分不同处理目录
        if self.sample_size is None:
            processed_folder_name = 'zinc100m_multipath_smiles_full'
        else:
            sample_in_millions = self.sample_size / 1_000_000
            if float(sample_in_millions).is_integer():
                folder_suffix = f'{int(sample_in_millions)}m'
            else:
                folder_suffix = f'{sample_in_millions:.3f}m'
            processed_folder_name = f'zinc100m_multipath_smiles_{folder_suffix}'

        self.processed_dir_base = root_path / 'processed' / processed_folder_name
        self.split = split
        self.lmdb_path = self.processed_dir_base / self.split / 'data.lmdb'

        # 如果当前 split 的 LMDB 不存在，则整体预处理（train/validation/test 一起）
        required_splits = ['train', 'validation', 'test']
        any_missing = any(
            not (self.processed_dir_base / s / 'data.lmdb').exists()
            for s in required_splits
        )

        if any_missing:
            print(f"Some LMDB splits missing under {self.processed_dir_base}, start preprocessing...")
            self.process()

        self._env: Optional[lmdb.Environment] = None
        self._len = self._get_db_length()

    def _check_exists(self) -> bool:
        return self.lmdb_path.exists()

    # ---------- 预处理主流程（不再数行，不再 RDKit，多进程） ----------

    def process(self):
        print(f"--- Starting ZINC Preprocessing from CSV files in {self.csv_data_path} ---")
        self.processed_dir_base.mkdir(parents=True, exist_ok=True)
        split_map = {'train': 'train', 'validation': 'valid', 'test': 'test'}

        for split_name, file_prefix in split_map.items():
            csv_file = self.csv_data_path / f"{file_prefix}.csv"
            if csv_file.exists():
                self._process_csv_split(split_name, csv_file)
            else:
                print(f"Warning: CSV file not found for '{split_name}' split at {csv_file}. Skipping.")

    def _process_csv_split(self, split_name: str, csv_path: Path):
        """
        流式读取某个 split 的 CSV：
          - 不预先 count 总行数
          - 不调用 RDKit，不做 canonical
          - 逐行写入 LMDB，key 从 0 递增，保证无空洞
          - 若是 train 且指定了 sample_size，则截断到 sample_size 条
        """
        split_dir = self.processed_dir_base / split_name
        lmdb_path = split_dir / 'data.lmdb'
        split_dir.mkdir(parents=True, exist_ok=True)

        if lmdb_path.exists():
            print(f"Multipath SMILES LMDB for '{split_name}' already exists. Skipping.")
            return

        print(f"Processing split '{split_name}' from {csv_path.name} (streaming, no line-count)...")

        use_writemap = platform.system() in ["Linux", "Darwin"]
        env = lmdb.open(
            str(lmdb_path),
            map_size=1 << 40,  # ~1TB 地址空间，足够 100M 级别
            writemap=use_writemap,
            subdir=False,
        )

        commit_interval = 100_000  # 每 10 万条 commit 一次
        num_processed = 0
        idx_counter = 0
        stop_flag = False

        try:
            txn = env.begin(write=True)
            reader = pd.read_csv(
                csv_path,
                usecols=['smiles'],
                chunksize=50_000,
                dtype=str,
            )

            pbar = tqdm(reader, desc=f"Streaming {split_name} (SMILES only)")
            for chunk in pbar:
                if stop_flag:
                    break

                smiles_series = chunk['smiles'].dropna()
                if smiles_series.empty:
                    continue

                for smi in smiles_series:
                    # train split 上支持 sample_size 截断
                    if self.sample_size is not None and split_name == 'train':
                        if idx_counter >= self.sample_size:
                            stop_flag = True
                            break

                    smi_str = smi.strip()
                    key = f'{idx_counter:010d}'.encode('ascii')
                    value = smi_str.encode('utf-8')
                    txn.put(key, value)

                    idx_counter += 1
                    num_processed += 1

                    if num_processed % commit_interval == 0:
                        txn.put(b'__len__', pickle.dumps(num_processed, protocol=pickle.HIGHEST_PROTOCOL))
                        txn.commit()
                        txn = env.begin(write=True)

                pbar.set_postfix({'n': num_processed})

            # 最后一次 commit
            txn.put(b'__len__', pickle.dumps(num_processed, protocol=pickle.HIGHEST_PROTOCOL))
            txn.commit()
            env.close()
            print(f"✅ Successfully finished processing '{split_name}'. Wrote {num_processed:,} entries.")

        except (KeyboardInterrupt, Exception) as e:
            print(f"\nError during processing of '{split_name}': {e}")
            try:
                txn.abort()
            except Exception:
                pass
            env.close()
            if lmdb_path.exists():
                try:
                    os.remove(lmdb_path)
                except IsADirectoryError:
                    shutil.rmtree(lmdb_path)
            print("Removed incomplete LMDB. Please re-run the script.")
            raise

    # ---------- LMDB 读辅助函数 ----------

    def _get_db_length(self) -> int:
        if not self.lmdb_path.exists():
            return 0
        try:
            env = lmdb.open(str(self.lmdb_path), readonly=True, lock=False, subdir=False)
            with env.begin() as txn:
                len_bytes = txn.get(b'__len__')
            env.close()
            return pickle.loads(len_bytes) if len_bytes else 0
        except (lmdb.Error, pickle.UnpicklingError):
            return 0

    def _open_env(self):
        if self._env is None:
            self._env = lmdb.open(
                str(self.lmdb_path),
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                subdir=False,
            )

    # ---------- PyTorch Dataset API ----------

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> Dict:
        """
        取样时：
        1. 从 LMDB 读 SMILES (UTF-8 bytes)
        2. RDKit 解析 + 生成 L 条 DeepSMILES 多路径
        3. Tokenize -> [L, max_seq_len]
        4. 生成 dummy MS + ECFP
        """
        if idx >= self._len:
            raise IndexError("Index out of range")
        self._open_env()
        key = f'{idx:010d}'.encode('ascii')
        with self._env.begin() as txn:
            byteflow = txn.get(key)

        if byteflow is None:
            raise KeyError(f"Data for index {idx} not found.")

        canonical_smi = byteflow.decode('utf-8')
        mol = Chem.MolFromSmiles(canonical_smi) if canonical_smi else None

        # 1. 生成 L 条 DeepSMILES 路径
        smiles_list: List[str] = []
        if mol is not None:
            # 第一条用 canonical + 显式 H
            base_smi = Chem.MolToSmiles(
                mol,
                canonical=True,
                isomericSmiles=False,
                kekuleSmiles=True,
                allHsExplicit=True
            )
            base_smi = self.deepsmiles_converter.encode(base_smi)
            smiles_list.append(base_smi)

            for _ in range(self.L - 1):
                rand_smi = Chem.MolToSmiles(
                    mol,
                    doRandom=True,
                    canonical=False,
                    isomericSmiles=False,
                    kekuleSmiles=True,
                    allHsExplicit=True
                )
                rand_smi = self.deepsmiles_converter.encode(rand_smi)
                smiles_list.append(rand_smi)
        else:
            # fallback
            smiles_list = [canonical_smi or ""] * self.L

        # 2. Tokenize L 条 SMILES
        tokenized = self.tokenizer(
            smiles_list,
            padding=True,
            return_tensors='pt'
        )
        seq_len = min(self.max_seq_len, tokenized['input_ids'].size(1))
        tokens_tensor = torch.full(
            (self.L, self.max_seq_len),
            self.tokenizer.pad_token_id,
            dtype=torch.long
        )
        tokens_tensor[:, :seq_len] = tokenized['input_ids'][:, :seq_len]

        # 3. Dummy spectrum
        mol_mass = 0.0
        if mol is not None:
            try:
                mol_mass = AllChem.CalcExactMolWt(mol)
            except Exception:
                mol_mass = 0.0
        precursor_mz_val = mol_mass + 1.007276  # Assume [M+H]+

        dummy_ms = {
            'precursor_mz': torch.tensor([precursor_mz_val], dtype=torch.float32),
            'mz': torch.tensor([precursor_mz_val], dtype=torch.float32),
            'intensity': torch.tensor([1.0], dtype=torch.float32),
            'adduct_weight': torch.tensor([1.007276], dtype=torch.float32),
            'charge': torch.tensor([1], dtype=torch.int32),
            'energy': torch.tensor([0.0], dtype=torch.float32),
            'is_dummy': torch.tensor([1], dtype=torch.long),  # Mark as dummy
        }

        # 4. ECFP fingerprint
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            arr = np.zeros((2048,), dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp, arr)
            ecfp_tensor = torch.from_numpy(arr.astype(np.float32))
            formular_vector = smiles_to_compact_vector(canonical_smi)
            formular_vector = torch.from_numpy(formular_vector.astype(np.long))
            
            mol_prior = get_scaffold_deepsmiles(mol)
            mol_prior_padded = torch.zeros((self.L, self.max_seq_len), dtype=torch.long) + self.tokenizer.pad_token_id
            tokenizd  = self.tokenizer.encode(mol_prior, padding=True, return_tensors='pt')
            token_size = min(self.max_seq_len, tokenizd.size(1))
            mol_prior_padded[:, :token_size] = tokenizd[:, :token_size]
        else:
            ecfp_tensor = torch.zeros(2048, dtype=torch.float32)
            formular_vector = torch.zeros((28,), dtype=torch.long)  # Dummy formula vector
            mol_prior_padded = torch.zeros((self.L, self.max_seq_len), dtype=torch.long) + self.tokenizer.pad_token_id
        
        return {
            'ms': dummy_ms,
            'smiles': canonical_smi,
            'tokens': tokens_tensor,
            'tokens_is_dummy': torch.tensor([0], dtype=torch.long),
            'ecfp': ecfp_tensor,
            'formular_vector': formular_vector,
            'mol_prior': mol_prior_padded,
        }


# ---------- CLI ----------

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Preprocess ZINC SMILES from CSV into LMDB databases (On-the-fly SMILES).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--root', type=str,
                        default="data",
                        help="Root directory to save the 'processed' LMDB databases.")
    parser.add_argument('--csv_path', type=str,
                        default="data/raw/zinc100m",
                        help="(仅用于打印) path to directory containing train.csv/valid.csv/test.csv.")
    parser.add_argument('--tokenizer_path', type=str,
                        default="tokenizer",
                        help="Path to the pretrained tokenizer (e.g., 'seyonec/ChemBERTa-zinc-base-v1').")
    parser.add_argument('--L', type=int, default=1,
                        help="Number of random SMILES paths to generate per molecule (for testing __getitem__).")
    parser.add_argument('--max_seq_len', type=int, default=200,
                        help="Max sequence length for SMILES tokenization (for testing __getitem__).")
    parser.add_argument('--num_workers', type=int, default=cpu_count(),
                        help="Number of worker processes for preprocessing (currently unused).")
    parser.add_argument('--sample_size', type=int, default=-1,
                        help="Process only the first N samples from the training CSV. Set to <=0 for full dataset.")

    args = parser.parse_args()

    print("\n--- ZINC Preprocessing Configuration (On-the-fly SMILES) ---")
    print(f"  Source CSV Directory: {args.csv_path}")
    print(f"  Output Root Directory: {args.root}")
    print(f"  Tokenizer Path: {args.tokenizer_path}")
    print(f"  Worker Processes (unused in preprocessing): {args.num_workers}")
    if args.sample_size and args.sample_size > 0:
        print(f"  Training Sample Size: {args.sample_size:,}")
    else:
        print("  Training Sample Size: full dataset")
    print("-----------------------------------------")

    # ⚠️ 建议先手动删掉旧的 processed/zinc100m_multipath_smiles* 目录，避免读到旧 LMDB
    dataset = ZINC_LMDB_Dataset(
        root=args.root,
        split='train',
        tokenizer_path=args.tokenizer_path,
        L=args.L,
        max_seq_len=args.max_seq_len,
        num_workers=args.num_workers,
        sample_size=args.sample_size,
    )

    if len(dataset) > 0:
        print("Testing __getitem__...")
        sample = dataset[0]
        print(f"Sample 0 keys: {sample.keys()}")
        print(f"Sample 0 tokens shape: {sample['tokens'].shape}")
        print(f"Sample 0 MS 'is_dummy': {sample['ms']['is_dummy']}")

    print("\nPreprocessing complete. LMDB databases are ready.")
