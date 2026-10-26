import numpy as np
import re
import torch
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

POSSIBLE_ELEMENTS = [
    'H', 'Li', 'B', 'C', 'N', 'O', 'F', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 
    'K', 'Ca', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'As', 'Se', 'Br', 'Mo', 'Sn', 'I'
]

# 建立 符号 -> 索引 的快速映射
ELEMENT_TO_IDX = {el: i for i, el in enumerate(POSSIBLE_ELEMENTS)}
VECTOR_SIZE = len(POSSIBLE_ELEMENTS)

# 与 POSSIBLE_ELEMENTS 顺序对齐的单同位素质量 (Da)，用于 formula → m/z
ELEMENT_MONO_MASSES = [
    1.007825, 7.016004, 11.009305, 12.000000, 14.003074, 15.994915, 18.998403,
    22.989769, 24.985837, 26.981538, 27.976927, 30.973762, 31.972071, 34.968853,
    38.963707, 39.962591, 54.938045, 55.934938, 58.933195, 59.930786, 62.929597,
    63.929142, 74.921595, 79.916521, 78.918336, 97.905406, 119.902199, 126.904473,
]


def get_element_mono_masses_tensor(dtype=torch.float32):
    return torch.tensor(ELEMENT_MONO_MASSES, dtype=dtype)


def smiles_to_compact_vector(smiles):
    """
    将 SMILES 转换为 28 维紧凑型常用元素向量
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return np.zeros(VECTOR_SIZE, dtype=int)
    
    # 1. 获取分子式 (C++ 效率)
    formula_str = CalcMolFormula(mol)
    
    # 2. 正则解析 (处理 Br, Mg 等双字母)
    pattern = re.compile(r'([A-Z][a-z]?)(\d*)')
    matches = pattern.findall(formula_str)
    
    # 3. 填充向量
    vector = np.zeros(VECTOR_SIZE, dtype=int)
    
    for symbol, count in matches:
        if symbol in ELEMENT_TO_IDX:
            idx = ELEMENT_TO_IDX[symbol]
            vector[idx] = int(count) if count else 1
        else:
            # 如果出现了列表之外的稀有元素 (如 Au, Pt, Gd)
            # 可以在此处打印警告或忽略
            pass
            
    return vector
  
  
