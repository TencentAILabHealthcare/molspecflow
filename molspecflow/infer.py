"""Run MolSpecFlow inference from JSONL records and a trusted checkpoint.

Run ``python -m molspecflow.infer --help`` for the command-line interface.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import torch
from rdkit import Chem
from rdkit.Chem import Descriptors

from . import MeteorFlowMatcher, MeteorFlowModel
from .formula_utils import ELEMENT_TO_IDX, VECTOR_SIZE, smiles_to_compact_vector


ADDUCT_MASSES = {
    "[M+H]+": 1.007276,
    "[M+Na]+": 22.989769,
    "[M+K]+": 38.963158,
    "[M+NH4]+": 18.033823,
}
TASKS = ("spec_to_smiles", "smiles_to_spec", "spec_to_ecfp")


def _number(value: object, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _formula_vector(formula: str | None) -> list[int]:
    counts = [0] * VECTOR_SIZE
    if formula is None:
        return counts
    matches = list(re.finditer(r"([A-Z][a-z]?)(\d*)", formula))
    if not matches or "".join(match.group() for match in matches) != formula:
        raise ValueError(f"Invalid molecular formula: {formula!r}")
    for match in matches:
        element = match.group(1)
        if element not in ELEMENT_TO_IDX:
            raise ValueError(f"Unsupported element in formula: {element}")
        counts[ELEMENT_TO_IDX[element]] += int(match.group(2) or 1)
    return counts


def _adduct_mass(record: dict) -> float:
    if "adduct_weight" in record:
        return _number(record["adduct_weight"], "adduct_weight")
    adduct = record.get("adduct", "[M+H]+")
    if adduct not in ADDUCT_MASSES:
        raise ValueError(f"Unsupported adduct {adduct!r}; provide adduct_weight")
    return ADDUCT_MASSES[adduct]


def _spectrum_peaks(record: dict) -> tuple[list[list[float]], list[bool]]:
    if "peaks" in record:
        pairs = record["peaks"]
    else:
        mz, intensity = record.get("mz"), record.get("intensity")
        if not isinstance(mz, list) or not isinstance(intensity, list) or len(mz) != len(intensity):
            raise ValueError("Provide equal-length mz and intensity arrays, or peaks pairs")
        pairs = list(zip(mz, intensity))
    if not isinstance(pairs, (list, tuple)):
        raise ValueError("peaks must be a list of [mz, intensity] pairs")
    clean = []
    for pair in pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError("Each peak must be [mz, intensity]")
        mz = _number(pair[0], "peak mz")
        intensity = _number(pair[1], "peak intensity")
        if mz > 0 and intensity > 0:
            clean.append([mz, intensity])
    if not clean:
        raise ValueError("Spectrum has no positive peaks")
    clean.sort(key=lambda pair: pair[1], reverse=True)
    clean = clean[:20]
    mask = [True] * len(clean) + [False] * (20 - len(clean))
    return clean + [[0.0, 0.0]] * (20 - len(clean)), mask


def prepare_batch(record: dict, task: str, tokenizer, max_seq_len: int) -> dict:
    """Convert one public JSON record into the matcher's batch contract."""
    if task not in TASKS:
        raise ValueError(f"Unknown task: {task}")
    adduct_mass = _adduct_mass(record)
    condition_formula = _formula_vector(record.get("formula"))
    tokens = None
    if task == "smiles_to_spec":
        import deepsmiles

        smiles = record.get("smiles")
        mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
        if mol is None:
            raise ValueError("smiles_to_spec requires a valid SMILES")
        canonical = Chem.MolToSmiles(
            mol, canonical=True, isomericSmiles=False,
            kekuleSmiles=True, allHsExplicit=True,
        )
        deep = deepsmiles.Converter(rings=True, branches=True).encode(canonical)
        ids = tokenizer(deep, add_special_tokens=True)["input_ids"]
        if len(ids) > max_seq_len:
            raise ValueError(f"Encoded SMILES has {len(ids)} tokens; max_seq_len={max_seq_len}")
        tokens = torch.full((1, max_seq_len), tokenizer.pad_token_id, dtype=torch.long)
        tokens[0, :len(ids)] = torch.as_tensor(ids, dtype=torch.long)
        neutral_mass = Descriptors.ExactMolWt(mol)
        condition_formula = smiles_to_compact_vector(smiles).tolist()
        precursor_mz = _number(record.get("precursor_mz", neutral_mass + adduct_mass),
                               "precursor_mz")
        peaks = [[0.0, 0.0]] * 20
        mask = [False] * 20
    else:
        precursor_mz = _number(record.get("precursor_mz"), "precursor_mz")
        peaks, mask = _spectrum_peaks(record)
        neutral_mass = precursor_mz - adduct_mass
    if precursor_mz <= 0 or neutral_mass <= 0:
        raise ValueError("precursor_mz and neutral mass must be positive")
    condition = {
        "precursor_mz": torch.tensor([precursor_mz], dtype=torch.float32),
        "adduct_weight": torch.tensor([adduct_mass], dtype=torch.float32),
        "charge": torch.tensor([_number(record.get("charge", 1), "charge")], dtype=torch.float32),
        "energy": torch.tensor([_number(record.get("collision_energy", record.get("energy", 0.0)), "collision_energy")], dtype=torch.float32),
        "is_dummy": torch.zeros(1, dtype=torch.bool),
        "total_mass": torch.tensor([neutral_mass], dtype=torch.float32),
        "formular_vector": torch.tensor([condition_formula], dtype=torch.float32),
    }
    batch = {
        "condition": condition,
        "peaks": torch.tensor([peaks], dtype=torch.float32),
        "spec_mask": torch.tensor([mask], dtype=torch.bool),
        "spec_is_dummy": torch.zeros(1, dtype=torch.bool),
    }
    if tokens is not None:
        batch["tokens"] = tokens
    return batch


def _config_from_checkpoint(checkpoint: dict, config_path: Path | None) -> dict:
    if config_path is not None:
        from training.utils.config_loader import load_config
        return load_config(str(config_path))
    config = checkpoint.get("hyper_parameters", {}).get("config")
    if config is None:
        raise ValueError("Checkpoint has no embedded config; pass --config")
    return config


def load_inference_model(checkpoint_path: Path, tokenizer_path: Path,
                         config_path: Path | None, device: torch.device):
    """Load a trusted Lightning checkpoint without constructing a training dataset."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a state-dict or Lightning checkpoint")
    config = _config_from_checkpoint(checkpoint, config_path)
    config["model"]["vocab_size"] = tokenizer.vocab_size
    config["model"]["pad_token_id"] = tokenizer.pad_token_id
    config["model"]["mask_token_id"] = tokenizer.mask_token_id
    state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, dict) or not state:
        raise ValueError("Checkpoint contains no model state_dict")
    if any(key.startswith("cfm.") for key in state):
        state = {key.removeprefix("cfm."): value for key, value in state.items()
                 if key.startswith("cfm.")}
    matcher = MeteorFlowMatcher(MeteorFlowModel(config["model"]), config)
    matcher.load_state_dict(state, strict=True)
    matcher.to(device).eval()
    return matcher, tokenizer, int(config["data"]["max_seq_len"])


def _decode_smiles(token_ids: torch.Tensor, tokenizer) -> tuple[list[str | None], list[str]]:
    import deepsmiles

    raw = tokenizer.batch_decode(token_ids.detach().cpu(), skip_special_tokens=True)
    converter = deepsmiles.Converter(rings=True, branches=True)
    smiles = []
    for text in raw:
        try:
            decoded = converter.decode(text)
            mol = Chem.MolFromSmiles(decoded)
            smiles.append(Chem.MolToSmiles(mol, isomericSmiles=False) if mol else None)
        except (ValueError, RuntimeError):
            smiles.append(None)
    return smiles, raw


def predict_one(matcher: MeteorFlowMatcher, tokenizer, record: dict,
                task: str, max_seq_len: int, steps: int, num_samples: int) -> dict:
    batch = prepare_batch(record, task, tokenizer, max_seq_len)
    output = matcher.generate(
        batch, n_smiles=max_seq_len, n_peaks=20,
        L=num_samples if task == "spec_to_smiles" else 1,
        T=steps, solver="euler", task=task,
    )
    result = {"identifier": record.get("identifier"), "task": task}
    if task == "spec_to_smiles":
        smiles, raw = _decode_smiles(output["smiles_tokens"], tokenizer)
        result.update(predictions=smiles, raw_deepsmiles=raw)
    elif task == "smiles_to_spec":
        peaks = output["peaks"][0].detach().cpu().tolist()
        result.update(pred_mz=[pair[0] for pair in peaks],
                      pred_int=[pair[1] for pair in peaks])
    else:
        result["ecfp_bits"] = output["ecfp"][0].detach().cpu().int().tolist()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Trusted MolSpecFlow Lightning checkpoint")
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="YAML config if absent from checkpoint")
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL")
    parser.add_argument("--output", type=Path, required=True, help="Prediction JSONL")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--num-samples", type=int, default=1,
                        help="SMILES candidates per spectrum")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.steps < 1 or args.num_samples < 1:
        parser.error("--steps and --num-samples must be positive")
    torch.manual_seed(args.seed)
    matcher, tokenizer, max_seq_len = load_inference_model(
        args.checkpoint, args.tokenizer, args.config, torch.device(args.device))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open(encoding="utf-8") as source, args.output.open("w", encoding="utf-8") as target:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Input line {line_number} must be a JSON object")
            prediction = predict_one(matcher, tokenizer, record, args.task,
                                     max_seq_len, args.steps, args.num_samples)
            if prediction["identifier"] is None:
                prediction["identifier"] = f"sample_{line_number:07d}"
            target.write(json.dumps(prediction, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
