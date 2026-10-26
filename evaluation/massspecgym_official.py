"""Score MolSpecFlow predictions with MassSpecGym's own metric functions.

Prediction records must carry official ``identifier`` values. Older outputs can
be joined by test-row position with ``--legacy-index-order`` after a SMILES
identity check. This is a scoring adapter, not a replacement for inference.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import pandas as pd
import torch
from rdkit import Chem


def _denovo_evaluator(with_mces: bool):
    from massspecgym.models.base import Stage
    from massspecgym.models.de_novo.base import DeNovoMassSpecGymModel

    class OfflineDeNovo(DeNovoMassSpecGymModel):
        def step(self, batch, stage=Stage.NONE):
            raise NotImplementedError("Offline scoring only")

        def _update_metric(self, *args, **kwargs):
            pass

    return OfflineDeNovo(
        top_ks=(1, 10),
        no_mces_metrics_at_stages=() if with_mces else (Stage.TEST,),
    )


def score_denovo(truth: list[str], predictions: list[list[str | None]],
                 with_mces: bool = True) -> dict[str, float]:
    """Use the official InChIKey, Morgan Tanimoto, and optional MCES metrics."""
    from massspecgym.models.base import Stage

    if len(truth) != len(predictions) or not truth:
        raise ValueError("Truth and prediction lists must be nonempty and aligned")
    predictions = [row if row else [None] for row in predictions]
    evaluator = _denovo_evaluator(with_mces)
    values = evaluator.evaluate_de_novo_step(predictions, truth, Stage.TEST)
    metrics = {
        name.removeprefix("test_"): float(tensor.float().mean().item())
        for name, tensor in values.items()
    }
    metrics["num_valid_mols"] = sum(
        sum(Chem.MolFromSmiles(s) is not None for s in row if s is not None)
        for row in predictions
    ) / len(predictions)
    return metrics


def _positive_peaks(spectrum: tuple[list[float], list[float]], mz_max: float):
    mz, intensity = spectrum
    if len(mz) != len(intensity):
        raise ValueError("m/z and intensity lengths differ")
    pairs = [(float(m), float(i)) for m, i in zip(mz, intensity)
             if 0.0 < float(m) < mz_max and float(i) > 0.0]
    if not pairs:
        return None
    mz_t = torch.tensor([x[0] for x in pairs], dtype=torch.float32)
    ints_t = torch.tensor([x[1] for x in pairs], dtype=torch.float32)
    return mz_t, ints_t


def _official_reference_peaks(
    spectrum: tuple[list[float], list[float]], mz_max: float, bin_res: float
) -> tuple[list[float], list[float]]:
    """Apply the benchmark's SpecToMzsInts transform to a reference spectrum."""
    import numpy as np
    from matchms import Spectrum
    from massspecgym.data.transforms import SpecToMzsInts

    mz, intensity = spectrum
    if len(mz) != len(intensity):
        raise ValueError("Reference m/z and intensity lengths differ")
    order = np.argsort(np.asarray(mz, dtype=float))
    spec = Spectrum(
        mz=np.asarray(mz, dtype=float)[order],
        intensities=np.asarray(intensity, dtype=float)[order],
        metadata={},
        metadata_harmonization=False,
    )
    transformed = SpecToMzsInts(n_peaks=None, mz_from=10.0,
                                mz_to=mz_max, mz_bin_res=bin_res)(spec)
    return transformed["spec_mzs"].tolist(), transformed["spec_ints"].tolist()


def score_simulation_spectrum(
    truth: tuple[list[float], list[float]],
    prediction: tuple[list[float], list[float]],
    mz_max: float = 1000.0,
    bin_res: float = 0.01,
) -> dict[str, float]:
    """Call official sparse cosine and Jensen-Shannon metric functions."""
    from massspecgym.simulation_utils.spec_utils import (
        sparse_cosine_distance,
        sparse_jensen_shannon_similarity,
    )

    true_peaks = _positive_peaks(_official_reference_peaks(truth, mz_max, bin_res), mz_max)
    pred_peaks = _positive_peaks(prediction, mz_max)
    if true_peaks is None or pred_peaks is None:
        return {"cos_sim": 0.0, "cos_sim_sqrt": 0.0, "js_sim": 0.0}
    true_mz, true_int = true_peaks
    pred_mz, pred_int = pred_peaks
    true_idx = torch.zeros(len(true_mz), dtype=torch.long)
    pred_idx = torch.zeros(len(pred_mz), dtype=torch.long)

    def logprob(ints: torch.Tensor, sqrt: bool = False) -> torch.Tensor:
        transformed = ints.sqrt() if sqrt else ints
        return (transformed / transformed.sum()).log()

    common = dict(
        true_mzs=true_mz, true_batch_idxs=true_idx,
        pred_mzs=pred_mz, pred_batch_idxs=pred_idx,
        mz_max=mz_max, mz_bin_res=bin_res,
    )
    cosine = 1.0 - sparse_cosine_distance(
        true_logprobs=logprob(true_int),
        pred_logprobs=logprob(pred_int), **common,
    )
    cosine_sqrt = 1.0 - sparse_cosine_distance(
        true_logprobs=logprob(true_int, sqrt=True),
        pred_logprobs=logprob(pred_int, sqrt=True), **common,
    )
    js = sparse_jensen_shannon_similarity(
        true_logprobs=logprob(true_int),
        pred_logprobs=logprob(pred_int), **common,
    )
    return {
        "cos_sim": float(cosine[0].item()),
        "cos_sim_sqrt": float(cosine_sqrt[0].item()),
        "js_sim": float(js[0].item()),
    }


def _canonical(smiles: str | None) -> str | None:
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    return Chem.MolToSmiles(mol, isomericSmiles=False) if mol else None


def load_reference(path: Path) -> pd.DataFrame:
    """Read the official TSV or a column-equivalent local CSV export."""
    frame = pd.read_csv(path, sep="\t" if path.suffix == ".tsv" else ",")
    frame.columns = [col.lower() for col in frame.columns]
    required = {"identifier", "fold", "smiles", "simulation_challenge"}
    if not required <= set(frame):
        raise ValueError(f"Reference is missing columns: {sorted(required - set(frame))}")
    if frame["identifier"].duplicated().any():
        raise ValueError("Duplicate reference identifiers")
    return frame


def _read_jsonl(path: Path) -> list[dict]:
    paths = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    if not paths:
        raise ValueError(f"No JSONL files found in {path}")
    records = []
    for item in paths:
        with item.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
    return records


def _attach_ids(records: list[dict], reference: pd.DataFrame,
                legacy_index_order: bool) -> list[dict]:
    test = reference[reference["fold"] == "test"].reset_index(drop=True)
    by_id = test.set_index("identifier", drop=False)
    seen = set()
    joined = []
    for record in records:
        identifier = record.get("identifier")
        if identifier is None and legacy_index_order:
            index = record.get("b_idx")
            if index is None and str(record.get("sample_id", "")).startswith("test_"):
                index = int(record["sample_id"].removeprefix("test_"))
            if index is None or not 0 <= int(index) < len(test):
                raise ValueError("Legacy record has no valid test-row index")
            row = test.iloc[int(index)]
            identifier = row["identifier"]
            record_smiles = record.get("ground_truth_smiles") or record.get("smiles")
            if _canonical(record_smiles) != _canonical(row["smiles"]):
                raise ValueError(f"SMILES mismatch at test-row index {index}")
        if identifier is None or identifier not in by_id.index:
            raise ValueError(f"Prediction identifier is missing or outside test fold: {identifier}")
        if identifier in seen:
            raise ValueError(f"Duplicate prediction identifier: {identifier}")
        seen.add(identifier)
        joined.append({**record, "identifier": identifier})
    return joined


def _spectrum_from_reference(row: pd.Series) -> tuple[list[float], list[float]]:
    if "mzs" in row and "intensities" in row:
        return (
            [float(x) for x in row["mzs"].split(",")],
            [float(x) for x in row["intensities"].split(",")],
        )
    if "spectrum" in row:
        mz, intensity = ast.literal_eval(row["spectrum"])
        return mz, intensity
    raise ValueError("Reference needs mzs/intensities or spectrum")


def run(args: argparse.Namespace) -> dict:
    reference = load_reference(Path(args.reference))
    records = _attach_ids(_read_jsonl(Path(args.predictions)), reference,
                          args.legacy_index_order)
    test = reference[reference["fold"] == "test"]
    if args.task == "simulation":
        challenge = test["simulation_challenge"].map(
            lambda value: str(value).lower() in ("true", "1")
        )
        test = test[challenge]
    allowed = test.set_index("identifier", drop=False)
    records = [record for record in records if record["identifier"] in allowed.index]
    if not records:
        raise ValueError("No predictions matched the selected test challenge")

    if args.task == "denovo":
        truth = [allowed.loc[r["identifier"], "smiles"] for r in records]
        predictions = [r.get("predictions", r.get("topk_pred_smiles_decoded", []))
                       for r in records]
        metrics = score_denovo(truth, predictions, with_mces=not args.skip_mces)
        per_sample = []
    else:
        per_sample = []
        for record in records:
            row = allowed.loc[record["identifier"]]
            metrics_one = score_simulation_spectrum(
                _spectrum_from_reference(row),
                (record["pred_mz"], record["pred_int"]),
                mz_max=args.mz_max, bin_res=args.bin_res,
            )
            per_sample.append({"identifier": record["identifier"], **metrics_one})
        metrics = {name: sum(r[name] for r in per_sample) / len(per_sample)
                   for name in ("cos_sim", "cos_sim_sqrt", "js_sim")}

    result = {
        "task": args.task,
        "reference_name": Path(args.reference).name,
        "predictions_name": Path(args.predictions).name,
        "n_scored": len(records),
        "n_expected": len(test),
        "complete_test_fold": len(records) == len(test),
        "metric_backend": "MassSpecGym package official metric functions",
        "metrics": metrics,
    }
    if args.task == "simulation":
        result["mz_max"] = args.mz_max
        result["bin_res"] = args.bin_res
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    if per_sample:
        per_path = output.with_name(output.stem + "_per_sample.jsonl")
        per_path.write_text("".join(json.dumps(row) + "\n" for row in per_sample),
                            encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("denovo", "simulation"))
    parser.add_argument("--reference", required=True, help="Official MassSpecGym.tsv or equivalent CSV")
    parser.add_argument("--predictions", required=True, help="JSONL file or shard directory")
    parser.add_argument("--output", required=True, help="Summary JSON output")
    parser.add_argument("--legacy-index-order", action="store_true",
                        help="Map b_idx/test_NNN to official test order after checking SMILES")
    parser.add_argument("--skip-mces", action="store_true", help="Skip costly de novo MCES")
    parser.add_argument("--mz-max", type=float, default=1000.0)
    parser.add_argument("--bin-res", type=float, default=0.01)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
