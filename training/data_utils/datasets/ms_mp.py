import os
import pickle
from pathlib import Path
from typing import List, Optional, Dict, Union, Tuple
import torch
from torch.utils.data import Dataset
import lmdb
import pandas as pd
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
import argparse
import ast


# --- Constants ---
PRECURSOR_MZ = 'precursor_mz'
CHARGE = 'charge'
ADDUCT = 'adduct'       # CSV 里没有也没关系，用默认值
SPECTRUM = 'spectrum'

ADDUCTS_WEIGHT = {
    '[M+H]+':   {'mass': 1.007276},
    '[M+Na]+':  {'mass': 22.989769},
    '[M+K]+':   {'mass': 38.963158},
    '[M+NH4]+': {'mass': 18.033823},
}



def _fast_count_rows(filepath: Path) -> int:
    """Efficiently counts rows in a large text file by reading in chunks."""
    buffer_size = 1024 * 1024 * 16  # 16MB buffer
    with open(filepath, 'rb') as f:
        count = 0
        while True:
            chunk = f.read(buffer_size)
            if not chunk:
                break
            count += chunk.count(b'\n')
    # Assuming there's a header line, otherwise return count
    return count - 1 if count > 0 else 0


def _process_csv_row(job: Dict) -> Optional[Tuple[bytes, str]]:
    """
    Worker Function: 只负责把一行 CSV 解析成 value 的 numpy dict，
    不再分配 key，由主进程在写 LMDB 时顺序生成 key。

    输入: job 是 row._asdict() 加上一列 'split_name'
    返回: (value_bytes, split_name)
    """
    try:
        split_name: str = job['split_name']

        # 解析 spectrum 字符串 -> 两个 list
        spectrum_str = job.get(SPECTRUM, None)
        if spectrum_str is None or (isinstance(spectrum_str, float) and np.isnan(spectrum_str)):
            return None

        try:
            mz_array, intensity_array = ast.literal_eval(spectrum_str)
            mz_array = np.asarray(mz_array, dtype=np.float32)
            intensity_array = np.asarray(intensity_array, dtype=np.float32)
        except Exception:
            # 解析失败直接丢弃
            return None

        if mz_array.size > 0:
            # 按强度排序，取 top 200
            sorted_indices = np.argsort(intensity_array)[::-1][:200]
            mz_sorted = mz_array[sorted_indices]
            intensity_sorted = intensity_array[sorted_indices]
        else:
            mz_sorted = np.empty(0, dtype=np.float32)
            intensity_sorted = np.empty(0, dtype=np.float32)

        # precursor_mz
        precursor_mz_raw = job.get(PRECURSOR_MZ, None)
        if precursor_mz_raw is None or (isinstance(precursor_mz_raw, float) and np.isnan(precursor_mz_raw)):
            return None
        precursor_mz = float(precursor_mz_raw)

        # charge
        charge_raw = job.get(CHARGE, 1)
        if isinstance(charge_raw, float) and np.isnan(charge_raw):
            charge_val = 1
        else:
            try:
                charge_val = int(charge_raw)
            except Exception:
                charge_val = 1

        # adduct（CSV 里没有的话就用默认 '[M+H]+'）
        adduct_str = job.get(ADDUCT, None)
        if adduct_str is None or (isinstance(adduct_str, float) and np.isnan(adduct_str)):
            adduct_str = '[M+H]+'
        adduct_weight = ADDUCTS_WEIGHT.get(adduct_str, ADDUCTS_WEIGHT['[M+H]+'])['mass']

        item_np = {
            'precursor_mz': np.array([precursor_mz], dtype=np.float32),
            'mz': mz_sorted,
            'intensity': intensity_sorted,
            'adduct_weight': np.array([adduct_weight], dtype=np.float32),
            'charge': np.array([charge_val], dtype=np.int32),
        }

        value = pickle.dumps({'ms': item_np}, protocol=pickle.HIGHEST_PROTOCOL)
        return value, split_name

    except Exception as e:
        print(f"Worker error processing job: {e}")
        return None


class MassSpecLmdbDataset(Dataset):
    """
    A PyTorch Dataset that preprocesses from a large CSV file to LMDB databases.
    只包含 spectrum；__getitem__ 返回 dummy token，以保持和其它任务统一。
    接口保持与你之前的版本一致。

    关键改动：
    - 预处理阶段只在真正写入 LMDB 时递增计数器并分配 key (0000000000, 0000000001, ...)
    - Dataset 加载阶段不再遍历 LMDB 获取所有 key，而是直接用 idx -> f"{idx:010d}".encode() 构造 key
    """

    def __init__(
        self,
        root: Union[str, Path],
        split: str,
        L: int,                # 用于 dummy tokens 的第一个维度
        max_seq_len: int,      # dummy tokens 的长度
        pad_token_id: int = 0,
        num_workers: Optional[int] = None,
        csv_filename: str = 'GeMS_A.csv',
        val_split_ratio: float = 0.025,
        test_split_ratio: float = 0.025,
        seed: int = 42,
        total_size: Optional[int] = None,
        sample_size: Optional[int] = -1,
        **kwargs
    ):
        super().__init__()
        self.root = Path(root).resolve()
        self.raw_path = self.root / 'raw' / csv_filename
        split = "validation" if split == "val" else split
        self.split = split

        self.num_workers = num_workers if num_workers is not None else os.cpu_count()
        self.val_split_ratio = val_split_ratio
        self.test_split_ratio = test_split_ratio
        self.seed = seed
        self.total_size = total_size
        self.sample_size = sample_size if sample_size and sample_size > 0 else None

        self.L = L
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id

        # === 目录根据 sample_size 区分 ===
        if self.sample_size is None:
            sample_tag = "full"
        else:
            sample_tag = f"sample_{self.sample_size}"
        self.processed_dir = self.root / 'processed' / f'spectra_only_{sample_tag}'
        self.lmdb_path = self.processed_dir / self.split / 'data.lmdb'

        required_splits = ['train', 'validation', 'test']
        is_any_split_missing = any(
            not (self.processed_dir / s / 'data.lmdb').exists()
            or (self.processed_dir / s / '_incomplete').exists()
            for s in required_splits
        )

        if is_any_split_missing:
            print("One or more data splits are missing or incomplete. Starting full preprocessing...")
            self.process()

        self._env: Optional[lmdb.Environment] = None
        self._len = self._get_len_from_db()

    # ---------- Preprocess ----------

    def process(self):
        if not self.raw_path.exists():
            raise FileNotFoundError(f"CSV source file not found at: {self.raw_path}")

        print("--- Starting CSV to LMDB Preprocessing for Spectra-Only Data (Dynamic Split) ---")

        train_ratio = 1.0 - self.val_split_ratio - self.test_split_ratio
        print(f"Using split ratios: {train_ratio:.1%} train / {self.val_split_ratio:.1%} val / {self.test_split_ratio:.1%} test")
        np.random.seed(self.seed)

        self._process_split_dynamically(train_ratio)

    def _process_split_dynamically(self, train_ratio: float):
        """
        Iterates through the CSV once, dynamically assigning rows to train/val/test splits
        and writing them to separate LMDBs.

        关键点：只有在真正写入 LMDB 时才递增 split_counters，并生成连续 key。
        """

        split_names = ['train', 'validation', 'test']
        envs: Dict[str, lmdb.Environment] = {}
        txns: Dict[str, lmdb.Transaction] = {}
        incomplete_markers: Dict[str, Path] = {}
        split_counters = {split: 0 for split in split_names}

        try:
            # --- 1. 打开各个 split 的 LMDB ---
            for split_name in split_names:
                split_dir = self.processed_dir / split_name
                lmdb_path = split_dir / 'data.lmdb'
                incomplete_marker = split_dir / '_incomplete'
                incomplete_markers[split_name] = incomplete_marker

                if lmdb_path.exists() and not incomplete_marker.exists():
                    print(f"LMDB for '{split_name}' is already complete. Skipping.")
                    continue

                print(f"Processing '{split_name}' split...")
                split_dir.mkdir(parents=True, exist_ok=True)

                # 清理旧数据
                if lmdb_path.exists():
                    if lmdb_path.is_dir():
                        # 目录形式的 LMDB（subdir=True）
                        for fname in os.listdir(lmdb_path):
                            (lmdb_path / fname).unlink()
                        lmdb_path.rmdir()
                    else:
                        lmdb_path.unlink()

                if incomplete_marker.exists():
                    incomplete_marker.unlink()

                incomplete_marker.touch()
                env = lmdb.open(
                    str(lmdb_path),
                    map_size=1024**4,  # 保持你原先的巨大 map_size
                    subdir=True,
                )
                envs[split_name] = env
                txns[split_name] = env.begin(write=True)

            if not txns:
                print("All splits are already processed and complete.")
                return

            # --- 2. 多进程遍历 CSV ---
            chunk_size = 10_000
            current_rows_read = 0

            with Pool(processes=self.num_workers) as pool:
                csv_iterator = pd.read_csv(
                    self.raw_path,
                    chunksize=chunk_size,
                    iterator=True,
                    low_memory=False,
                )

                pbar = tqdm(csv_iterator, desc="Processing CSV chunks")
                for chunk_df in pbar:
                    if self.sample_size is not None and current_rows_read >= self.sample_size:
                        print(f"\nReached sample size limit of {self.sample_size}. Stopping.")
                        break

                    # 控制 sample_size 截断
                    if self.sample_size is not None:
                        remaining = self.sample_size - current_rows_read
                        if len(chunk_df) > remaining:
                            chunk_df = chunk_df.iloc[:remaining]

                    current_rows_read += len(chunk_df)

                    # 动态分配 split
                    rand_vals = np.random.rand(len(chunk_df))
                    split_col = np.where(
                        rand_vals < train_ratio, 'train',
                        np.where(
                            rand_vals < (train_ratio + self.val_split_ratio),
                            'validation',
                            'test'
                        )
                    )
                    chunk_df['split_name'] = split_col

                    jobs: List[Dict] = []
                    for row in chunk_df.itertuples(index=False):
                        job_dict = row._asdict()  # 包含所有列（包括 split_name）
                        split_name = job_dict['split_name']
                        if split_name not in txns:
                            continue
                        # 不在这里分配 key / 递增 counter，只把 row 信息交给 worker
                        jobs.append(job_dict)

                    # 多进程处理这个 chunk 的所有 jobs
                    for result in pool.imap_unordered(_process_csv_row, jobs, chunksize=512):
                        if result is None:
                            continue
                        value, split_name = result
                        if split_name in txns:
                            # 只有真正写入时才分配 key，保证 key 连续
                            idx = split_counters[split_name]
                            key = f"{idx:010d}".encode()
                            split_counters[split_name] += 1
                            txns[split_name].put(key, value)

                    pbar.set_postfix({split: f"{count:,}" for split, count in split_counters.items()})

            # --- 3. 提交 & 关闭 ---
            for split_name, txn in txns.items():
                print(f"Committing data for '{split_name}'...")
                final_len = split_counters[split_name]
                txn.put(b'__len__', pickle.dumps(final_len))
                txn.commit()
                envs[split_name].close()
                incomplete_markers[split_name].unlink()
                print(f"✅ Successfully finished processing for '{split_name}' split. Total: {final_len:,}")

        except (KeyboardInterrupt, Exception) as e:
            print(f"\nProcess interrupted by {type(e).__name__}. LMDBs may be incomplete.")
            for split_name, txn in txns.items():
                try:
                    txn.abort()
                except Exception:
                    pass
            for split_name, env in envs.items():
                try:
                    env.close()
                except Exception:
                    pass
            print(f"Error details: {e}")
            import traceback
            traceback.print_exc()
            raise

    # ---------- LMDB helpers ----------

    def _get_len_from_db(self) -> int:
        if not self.lmdb_path.exists():
            return 0
        env = lmdb.open(str(self.lmdb_path), readonly=True, lock=False, subdir=True)
        with env.begin() as txn:
            len_bytes = txn.get(b'__len__')
        env.close()
        return pickle.loads(len_bytes) if len_bytes else 0

    def _open_env(self):
        if self._env is None:
            self._env = lmdb.open(
                str(self.lmdb_path),
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                subdir=True,
            )

    # ---------- PyTorch Dataset API ----------

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> Dict:
        """
        Retrieves a spectrum and pairs it with dummy tokens
        for API consistency.

        关键：不再从 DB 拉取所有 key，而是直接用 idx -> "0000000000" 形式构造 key。
        """
        if idx < 0 or idx >= self._len:
            raise IndexError(f"Index {idx} out of range for dataset of length {self._len}.")

        self._open_env()

        key = f"{idx:010d}".encode()

        with self._env.begin() as txn:
            byteflow = txn.get(key)
        if byteflow is None:
            raise IndexError(f"Data for index {idx} (key {key}) not found in LMDB.")

        item_np = pickle.loads(byteflow)

        # 1. spectrum 转成 torch
        ms_np = item_np['ms']
        ms_torch = {k.lower(): torch.from_numpy(v) for k, v in ms_np.items()}
        ms_torch['is_dummy'] = torch.tensor([0], dtype=torch.long)  # Mark as real

        # Truncate for consistency: 取前 20 个 peak
        max_peaks = 20
        int_padded = torch.zeros(max_peaks, dtype=torch.float32)
        mz_padded = torch.zeros(max_peaks, dtype=torch.float32)
        n_peaks = min(ms_torch['intensity'].size(0), max_peaks)
        int_padded[:n_peaks] = ms_torch['intensity'][:max_peaks]
        mz_padded[:n_peaks] = ms_torch['mz'][:max_peaks]
        ms_torch['intensity'] = int_padded
        ms_torch['mz'] = mz_padded
        ms_torch['energy'] = torch.tensor([0.0], dtype=torch.float32)  # Dummy energy

        # 2. dummy tokens (shape [L, max_seq_len])
        dummy_tokens = torch.full(
            (self.L, self.max_seq_len),
            self.pad_token_id,
            dtype=torch.long
        )

        return {
            'ms': ms_torch,
            'smiles': "",  # No SMILES info available
            'tokens': dummy_tokens,
            'tokens_is_dummy': torch.tensor([1], dtype=torch.long),
            'ecfp': torch.zeros((2048,), dtype=torch.float32),  # Dummy ECFP
            'formular_vector': torch.zeros((28,), dtype=torch.long),  # Dummy formula vector
            'mol_prior': torch.zeros((self.L, self.max_seq_len), dtype=torch.long) + self.pad_token_id
        }


# ---------- CLI ----------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Preprocess a large CSV file into train/validation/test LMDB databases.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--root', type=str, default='data',
                        help="Root directory with a 'raw' folder for the CSV and to save 'processed' LMDBs.")
    parser.add_argument('--csv_filename', type=str, default='GeMS_A.csv',
                        help="Name of the CSV file, expected inside '<root>/raw/'.")
    parser.add_argument('--L', type=int, default=1,
                        help="Number of paths (for dummy token shape).")
    parser.add_argument('--max_seq_len', type=int, default=128,
                        help="Max sequence length (for dummy token shape).")
    parser.add_argument('--pad_token_id', type=int, default=0,
                        help="PAD token ID (for dummy tokens).")
    parser.add_argument('--num_workers', type=int, default=os.cpu_count(),
                        help="Number of worker processes for preprocessing.")
    parser.add_argument('--val_split', type=float, default=0.025,
                        help="Fraction of data for the validation set.")
    parser.add_argument('--test_split', type=float, default=0.025,
                        help="Fraction of data for the test set.")
    parser.add_argument('--seed', type=int, default=42,
                        help="Random seed for data splitting.")
    parser.add_argument('--total_size', type=int, default=None,
                        help="Optional: Provide the exact number of rows (minus header) to skip the initial file scan.")
    parser.add_argument('--sample_size', type=int, default=-1,
                        help="Optional: Process only the first N samples of the dataset for quick testing.")

    args = parser.parse_args()

    raw_path = Path(args.root) / 'raw' / args.csv_filename
    print(f"Starting preprocessing. Raw CSV: {raw_path}")

    if args.sample_size and args.sample_size > 0:
        sample_tag = f"sample_{args.sample_size}"
    else:
        sample_tag = "full"
    out_dir = Path(args.root) / 'processed' / f'spectra_only_{sample_tag}'
    print(f"Output LMDBs will be saved in: {out_dir}")

    try:
        dataset = MassSpecLmdbDataset(
            root=args.root,
            csv_filename=args.csv_filename,
            split='train',
            L=args.L,
            max_seq_len=args.max_seq_len,
            pad_token_id=args.pad_token_id,
            num_workers=args.num_workers,
            val_split_ratio=args.val_split,
            test_split_ratio=args.test_split,
            seed=args.seed,
            total_size=args.total_size,
            sample_size=args.sample_size
        )
        print("\nPreprocessing complete. LMDB databases for 'train', 'validation', and 'test' are ready.")

        # --- Test __getitem__ ---
        if len(dataset) > 0:
            print("Testing __getitem__...")
            sample = dataset[0]
            print(f"Sample 0 keys: {sample.keys()}")
            print(f"Sample 0 tokens shape: {sample['tokens'].shape}")
            print(f"Sample 0 MS 'is_dummy': {sample['ms']['is_dummy']}")
        # --- End Test ---

    except FileNotFoundError as e:
        print(f"\nError: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred during preprocessing: {e}")
        import traceback
        traceback.print_exc()
