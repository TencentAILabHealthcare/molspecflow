import os
import shutil
import pickle
from pathlib import Path
from typing import List, Optional, Dict
import numpy as np
import torch
# --- 修改: 不再需要 torch_geometric ---
from torch.utils.data import Dataset
# --- 结束修改 ---
from tqdm import tqdm
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from multiprocessing import Pool, cpu_count
import lmdb
import argparse
# --- 新增 ---
from transformers import AutoTokenizer
import deepsmiles
# --- 结束新增 ---

import numpy as np
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
import re
import ast
from molspecflow.formula_utils import POSSIBLE_ELEMENTS, ELEMENT_TO_IDX, VECTOR_SIZE, smiles_to_compact_vector
from rdkit.Chem.Scaffolds import MurckoScaffold
# Disable RDKit console logging
RDLogger.DisableLog('rdApp.*')

ADDUCTS_WEIGHT = {
    '[M+H]+': {'smi': "[H+]", "mass": 1.007276},
    '[M+Na]+': {'smi': "[Na+]", "mass": 22.989769},
    '[M+K]+': {'smi': "[K+]", "mass": 38.963158},
    '[M+NH4]+': {'smi': "[NH4+]", "mass": 18.033823},
}

# --- 修改: 不再需要 worker tokenizer ---
# def init_worker(tokenizer_path: str): ...
# --- 结束修改 ---


def smiles_to_deepsmiles(smi: str) -> str:
    conv = deepsmiles.Converter(rings=True, branches=True)
    return conv.encode(smi)

def get_scaffold_deepsmiles(mol: Chem.Mol) -> str:
    scaf = MurckoScaffold.GetScaffoldForMol(mol)
    if scaf is None or scaf.GetNumAtoms() == 0:
        return ""
    scaf_smi = Chem.MolToSmiles(scaf, canonical=True, isomericSmiles=False, kekuleSmiles=True, allHsExplicit=True)
    return smiles_to_deepsmiles(scaf_smi)

# --- Helper function to convert a single sample ---
def _process_row(row_data: Dict) -> Optional[Dict]:
    """
    Worker function to process a single row from the DataFrame into a
    dictionary of NumPy arrays, which is safe for pickling across processes.
    
    MODIFIED: Only saves canonical SMILES and MS data. No graph data.
    """
    
    smi = row_data['smiles']
    mol = Chem.MolFromSmiles(smi)
    if not mol:
        return None

    # --- 修改: 只保留 canonical_smi ---
    smi = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
    # --- 结束修改 ---

    # --- 修改: 移除图特征提取 ---
    # Chem.Kekulize(mol, clearAromaticFlags=True)
    # AllChem.ComputeGasteigerCharges(mol)
    # xs = [] ...
    # x = np.array(xs, dtype=np.float32)
    # edge_indices, edge_attrs = [], [] ...
    # edge_index = ...
    # edge_attr = ...
    # --- 结束修改 ---

    spectrum_str = row_data['spectrum']
    mz, intensity = ast.literal_eval(spectrum_str) if isinstance(spectrum_str, str) else spectrum_str
    
    spectrum_sorted = sorted(zip(mz, intensity), key=lambda item: item[1], reverse=True)
    mz_sorted, intensity_sorted = zip(*spectrum_sorted) if spectrum_sorted else ([], [])
    
    precursor_mass = row_data['PARENT_MASS']
    adduct_weight = ADDUCTS_WEIGHT.get(row_data.get('ADDUCT', '[M+H]+'), ADDUCTS_WEIGHT['[M+H]+'])['mass']
    
    # --- 结束修改 ---
    return {
        'ms': {
            'precursor_mz': np.array([precursor_mass], dtype=np.float32),
            'mz': np.array(list(mz_sorted), dtype=np.float32),
            'intensity': np.array(list(intensity_sorted), dtype=np.float32),
            'adduct_weight': np.array([adduct_weight], dtype=np.float32),
            'charge': np.array([1], dtype=np.int32),
            'energy': np.array([row_data.get('COLLISION_ENERGY', 0.0)], dtype=np.float32)
        },
        'smiles': smi,
        # 'tokens': np.array(tokens, dtype=np.int64) 
    }



class MSMolDataset(Dataset): # --- 修改: 基类改为 torch.utils.data.Dataset ---
    """
    A PyTorch Dataset that uses LMDB for storage.
    MODIFIED: Generates L random SMILES paths on-the-fly in __getitem__.
    """
    
    @property
    def processed_dir_path(self) -> Path:
        # --- 修改: 更改路径以避免与旧数据冲突 ---
        return Path(self.root) / 'processed_multipath_smiles' / "mass_gym_coe" / self.data_split
        # --- 结束修改 ---
    
    @property
    def processed_dir(self) -> str:
        return str(self.processed_dir_path)
    
    @property
    def raw_file_names(self) -> List[str]:
        return [self.raw_filename]

    @property
    def processed_file_names(self) -> str:
        return 'data.lmdb'
    
    
    def __init__(self, root: str, 
                 data_split: str, 
                 tokenizer_path: str, 
                 L: int, 
                 max_seq_len: int, 
                 pad_token_id: int = 0, 
                 num_workers: Optional[int] = None):
        super().__init__() # --- 修改: super() 不再需要 root ---
        self.root = root
        self.data_split = data_split
        self.raw_filename = 'mass_spec_gym.csv'
        self.num_workers = num_workers if num_workers is not None else os.cpu_count() // 2
        
        # --- 新增: On-the-fly generation parameters ---
        if tokenizer_path is None:
            raise ValueError("A tokenizer_path must be provided.")
        self.tokenizer_path = tokenizer_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        self.L = L
        self.max_seq_len = max_seq_len
        # self.pad_token_id = pad_token_id
        self.lmdb_path = self.processed_dir_path / self.processed_file_names
        # --- 结束新增 ---
        
        # --- 修改: super().__init__(root) 移到这里 ---
        # super().__init__(root)
        # --- 结束修改 ---
        
        self._env: Optional[lmdb.Environment] = None
        self._len: Optional[int] = None

        # --- 新增: 手动调用 process ---
        if not self._check_exists():
            print(f"LMDB not found at {self.lmdb_path}. Processing...")
            self.process()
        # --- 结束新增 ---
        
        self._len = self._get_len_from_db()


    def download(self):
        pass
    
    # --- 新增: 检查是否存在 ---
    def _check_exists(self) -> bool:
        return (self.processed_dir_path / self.processed_file_names).exists()
    # --- 结束新增 ---

    def process(self):
        lmdb_path = self.processed_dir_path / self.processed_file_names
        os.makedirs(self.processed_dir_path, exist_ok=True)
        if os.path.exists(lmdb_path):
            print(f"Multipath SMILES LMDB already exists at {lmdb_path}. Skipping processing.")
            return

        # --- 修改: raw_paths[0] -> raw_dir / raw_filename ---
        raw_path = Path(self.root) / 'raw' / self.raw_filename
        if not raw_path.exists():
             raise FileNotFoundError(f"Raw file not found: {raw_path}")
        # --- 结束修改 ---
             
        df = pd.read_csv(raw_path)
        if self.data_split in ['train', 'val', 'test']:
            split_df = df[df['FOLD'] == self.data_split].copy()
            data_rows = split_df.to_dict('records')
            print(f"Processing {len(data_rows)} samples for split '{self.data_split}' using NumPy backend...")
        else:
            split_df = df.copy()
            data_rows = split_df.to_dict('records')
        
        
        write_env = lmdb.open(str(lmdb_path), map_size=1024**4) # 1TB
        
        with write_env.begin(write=True) as txn:
            idx = 0
            # --- 修改: 移除 worker pool 初始化器 ---
            with Pool(self.num_workers) as pool:
            # --- 结束修改 ---
                results_iterator = pool.imap(_process_row, data_rows)
                for result in tqdm(results_iterator, total=len(data_rows)):
                    if result:
                        key = f'{idx:010d}'.encode()
                        txn.put(key, pickle.dumps(result))
                        idx += 1
            
            txn.put(b'__len__', pickle.dumps(idx))

        write_env.close()
        print(f"Saved {idx} processed SMILES/MS entries to LMDB at {lmdb_path}")

    def _get_env(self):
        if self._env is None:
            lmdb_path = self.processed_dir_path / self.processed_file_names
            self._env = lmdb.open(str(lmdb_path), readonly=True, lock=False, readahead=False, meminit=False)
        return self._env
    
    def _get_len_from_db(self) -> int:
        lmdb_path = self.processed_dir_path / self.processed_file_names
        if not lmdb_path.exists(): return 0
        with lmdb.open(str(lmdb_path), readonly=True, lock=False) as env:
            with env.begin() as txn:
                len_bytes = txn.get(b'__len__')
                return pickle.loads(len_bytes) if len_bytes else 0
            


    def __getitem__(self, idx: int) -> Dict:
        """
        Retrieves a single item, creates dummy placeholders if necessary,
        and generates L random SMILES paths on-the-fly.
        """
        with self._get_env().begin() as txn:
            byteflow = txn.get(f'{idx:010d}'.encode())
        
        if byteflow is None:
            raise IndexError(f"Index {idx} out of range.")
        
        item_np = pickle.loads(byteflow)
        
        # --- MODIFICATION START ---
        final_item = {}

        # --- Handle SMILES / Tokens ---
        if self.data_split == 'ms1':
            # Create dummy tokens for 'ms' split (no SMILES)
            final_item['tokens'] = torch.full(
                (self.L, self.max_seq_len), self.tokenizer.pad_token_id, dtype=torch.long
            )
            final_item['smiles'] = ""
            final_item['tokens_is_dummy'] = torch.tensor([1], dtype=torch.long)
            final_item['ecfp'] = torch.zeros((2048,), dtype=torch.float32)
            final_item['formular_vector'] = torch.zeros((VECTOR_SIZE,), dtype=torch.int)
            final_item['mol_prior'] = get_scaffold_deepsmiles(Chem.MolFromSmiles(item_np.get('smiles', '')))
        else:
            # Generate L random SMILES paths on-the-fly
            canonical_smi = item_np.get('smiles', '')
            mol = Chem.MolFromSmiles(canonical_smi)
            deepsmiles_encoder = deepsmiles.Converter(rings=True, branches=True)
            smiles_list = [Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False, kekuleSmiles=True, allHsExplicit=True)]
            smiles_list[0] = deepsmiles_encoder.encode(smiles_list[0])
            final_item['formular_vector'] = torch.from_numpy(smiles_to_compact_vector(canonical_smi)).long()
            final_item['mol_prior'] = get_scaffold_deepsmiles(mol)
            if mol:
                for _ in range(self.L-1):
                    random_smi = Chem.MolToSmiles(mol, doRandom=True, canonical=False, isomericSmiles=False, kekuleSmiles=True, allHsExplicit=True)
                    random_smi = deepsmiles_encoder.encode(random_smi)
                    # try:
                    # random_smi = safe.encode(inp=canonical_smi, randomize=True, canonical=True, ignore_stereo=True, slicer='mmpa')
                    
                    # except Exception as e:
                    #     print(f"Error generating random SMILES for {canonical_smi}: {e}")
                    #     random_smi = canonical_smi
                    
                        
                    smiles_list.append(random_smi)
            else:
                # Fallback for invalid SMILES
                smiles_list = [canonical_smi] * self.L
            
            # Tokenize the list of L SMILES
            tokenized = self.tokenizer(
                smiles_list,
                padding=True,
                # truncation=True,
                # max_length=self.max_seq_len,
                return_tensors='pt'
            )
            final_item['tokens'] = torch.zeros((self.L, self.max_seq_len), dtype=torch.long) + self.tokenizer.pad_token_id
            final_item['tokens'][:, :tokenized['input_ids'].size(1)] = tokenized['input_ids'] # Shape [L, max_seq_len]
            final_item['smiles'] = canonical_smi # Return canonical for info
            final_item['tokens_is_dummy'] = torch.tensor([0], dtype=torch.long)
            ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            arr = np.zeros((1,), dtype=np.int8)
            AllChem.DataStructs.ConvertToNumpyArray(ecfp, arr)
            final_item['ecfp'] = torch.from_numpy(arr).float()
            mol_prior = torch.zeros((self.L, self.max_seq_len), dtype=torch.long) + self.tokenizer.pad_token_id
            tokenizd  = self.tokenizer.encode(final_item['mol_prior'], padding=True, return_tensors='pt')
            mol_prior[:, :tokenizd.size(1)] = tokenizd
            final_item['mol_prior'] = mol_prior
            
        # --- End SMILES / Tokens ---

        # --- Handle 'ms' data ---
        if self.data_split == 'mol1':
            # Create a dummy spectrum for 'mol' split
            mol_mass = 0.0
            precursor_mz_val = mol_mass + 1.007276 # Assume [M+H]+
            ms_torch = {
                'precursor_mz': torch.tensor([precursor_mz_val], dtype=torch.float32),
                'mz': torch.tensor([precursor_mz_val]+[0.0]*19, dtype=torch.float32),
                'intensity': torch.tensor([1.0]+[0.0]*19, dtype=torch.float32),
                'adduct_weight': torch.tensor([1.007276], dtype=torch.float32),
                'charge': torch.tensor([1], dtype=torch.int),
                'is_dummy': torch.tensor([1], dtype=torch.long),
                'energy': torch.tensor([0.0], dtype=torch.float32)
            }
        else:
            # Create the real spectrum for 'train', 'val', 'test', 'ms' splits
            ms_np = item_np['ms']
            ms_torch = {k: torch.from_numpy(v) for k, v in ms_np.items()}
            ms_torch['is_dummy'] = torch.tensor([0], dtype=torch.long)
            ms_padded = torch.zeros(20, dtype=torch.float32)
            int_padded = torch.zeros(20, dtype=torch.float32)     
            mz_len = min(ms_torch['mz'].shape[0], 20)
            int_len = min(ms_torch['intensity'].shape[0], 20)
            ms_padded[:mz_len] = ms_torch['mz'][:mz_len]
            int_padded[:int_len] = ms_torch['intensity'][:int_len]
            ms_torch['intensity'] = int_padded
            ms_torch['mz'] = ms_padded
            
        
        final_item['ms'] = ms_torch
        
        # --- 移除 'graph' data ---
        # final_item['graph'] = ...
        
        return final_item
    
    def __len__(self):
        if self._len is None:
            self._len = self._get_len_from_db()
        return self._len
    # --- 结束修改 ---
    
    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Preprocess Mass Spectrometry data from a CSV file into LMDB databases.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--root', type=str, default="data",
                        help="Root directory containing raw/ and for saving processed/ data.")
    # --- 新增 ---
    parser.add_argument('--tokenizer_path', type=str, required=True,
                        help="Path to the pretrained tokenizer (e.g., 'seyonec/ChemBERTa-zinc-base-v1').")
    parser.add_argument('--L', type=int, default=1,
                        help="Number of random SMILES paths to generate per molecule (for testing __getitem__).")
    parser.add_argument('--max_seq_len', type=int, default=128,
                        help="Max sequence length for SMILES tokenization (for testing __getitem__).")
    # --- 结束新增 ---
    parser.add_argument('--num_workers', type=int, default=cpu_count(), 
                        help="Number of worker processes to use for preprocessing.")
    
    args = parser.parse_args()

    splits_to_process = ['train', 'val', 'test', 'mol', 'ms']
    
    print("--- Starting Dataset Preprocessing (On-the-fly SMILES) ---")
    print(f"Root Directory: {args.root}")
    print(f"Tokenizer Path: {args.tokenizer_path}")
    print(f"Number of Workers: {args.num_workers}")
    print("--------------------------------------")

    for split in splits_to_process:
        print(f"\n--- Processing '{split}' split ---")
        try:
            dataset = MSMolDataset(
                root=args.root,
                data_split=split,
                tokenizer_path=args.tokenizer_path,
                L=args.L,
                max_seq_len=args.max_seq_len,
                num_workers=args.num_workers
            )
            print(f"'{split}' split contains {len(dataset)} samples.")
            
            # --- Test __getitem__ ---
            if len(dataset) > 0:
                print("Testing __getitem__...")
                sample = dataset.get(0)
                print(f"Sample 0 keys: {sample.keys()}")
                print(f"Sample 0 tokens shape: {sample['tokens'].shape}")
                print(f"Sample 0 MS 'is_dummy': {sample['ms']['is_dummy']}")
            # --- End Test ---

            dataset.close()
        except Exception as e:
            print(f"An error occurred while processing the '{split}' split: {e}")
            import traceback
            traceback.print_exc()

    print("\n--- Preprocessing complete for all splits. ---")
