# MolSpecFlow

<img src="assets/logo.png" alt="MolSpecFlow logo" width="180">

MolSpecFlow is a flow-matching model for molecular structures and tandem mass spectra. This repository provides the model, training configurations, inference command, and a MassSpecGym metric adapter.

Supported inference tasks:

- MS/MS spectrum → molecular structures (`spec_to_smiles`)
- molecular structure → MS/MS spectrum (`smiles_to_spec`)
- MS/MS spectrum → 2048-bit molecular fingerprint (`spec_to_ecfp`)

## Quick start: inference

Use Python 3.10 or newer. Install from this directory:

```bash
python -m pip install -e '.[inference]'
```

Place a compatible training checkpoint at `checkpoints/model.ckpt` and its matching Hugging Face tokenizer at `tokenizer/`. The checkpoint must contain the MolSpecFlow model weights and training config; if it does not contain the config, pass `--config training/configs/<matching-config>.yaml`. Only load checkpoints and tokenizers from sources you trust.

To generate molecular structures, create `inputs/spectra.jsonl` with one JSON object per line:

```json
{"identifier":"example_1","precursor_mz":181.0707,"adduct":"[M+H]+","collision_energy":30,"mz":[43.0184,77.0386,105.0335],"intensity":[0.3,0.7,1.0]}
```

Then run:

```bash
python -m molspecflow.infer \
  --task spec_to_smiles \
  --checkpoint checkpoints/model.ckpt --tokenizer tokenizer \
  --input inputs/spectra.jsonl --output runs/structures.jsonl \
  --num-samples 10 --steps 100
```

Each output line contains `identifier`, an ordered `predictions` list of decoded SMILES (invalid generations are `null`), and `raw_deepsmiles`. An optional `formula` field in the input supplies molecular-formula conditioning; without it, the formula vector is zero. Spectra are sorted by intensity and truncated/padded to the 20 peaks used during training. `peaks` as `[[mz, intensity], ...]` is also accepted in place of the two arrays.

For molecule-to-spectrum inference, use a JSONL input such as:

```json
{"identifier":"example_2","smiles":"CCO","adduct":"[M+H]+","collision_energy":30}
```

```bash
python -m molspecflow.infer \
  --task smiles_to_spec \
  --checkpoint checkpoints/model.ckpt --tokenizer tokenizer \
  --input inputs/molecules.jsonl --output runs/spectra.jsonl
```

The output contains `pred_mz` and `pred_int` arrays. For fingerprint inference, use the spectrum input format and `--task spec_to_ecfp`; the output contains `ecfp_bits`. Use a checkpoint trained or fine-tuned for the selected task. The command accepts `--device cpu` or `--device cuda` and `--seed` for reproducibility.

## Training

Install the training dependencies:

```bash
python -m pip install -e '.[training]'
```

The default training configs expect `data/raw/mass_spec_gym.csv`, `data/raw/GeMS_A.csv`, `data/raw/zinc100m/{train,valid,test}.csv`, and `tokenizer/`. The paired CSV needs `smiles`, `spectrum`, `PARENT_MASS`, `ADDUCT`, `COLLISION_ENERGY`, and `FOLD` columns. Paths can be changed with `--data-root`, `--tokenizer`, and `--output-dir`.

Joint pretraining:

```bash
python -m training.mptrain --config training/configs/mppretrain_joint_v2.yaml
```

Task fine-tuning from a compatible joint checkpoint:

```bash
python -m training.mptrain --config training/configs/mpfinetune_spec_to_smiles.yaml --pretrained checkpoints/joint.ckpt
python -m training.mptrain --config training/configs/mpfinetune_smiles_to_spec.yaml --pretrained checkpoints/joint.ckpt
python -m training.mptrain --config training/configs/mpfinetune_spec_to_ecfp.yaml --pretrained checkpoints/joint.ckpt
```

The supplied configs use eight devices by default; adjust `training.devices`, batch size, and workers for your hardware. `--debug` runs a small single-device check.

The molecular training losses are SMILES token cross-entropy, total-mass Huber loss, and molecular-formula Huber loss. Fingerprint prediction uses regression loss. Spectrum generation uses m/z, intensity, and multi-resolution binned cosine losses. The code does not implement a separate molecular graph-topology loss.

## MassSpecGym evaluation

Install the evaluation dependencies in a separate environment because MassSpecGym pins older scientific-library versions:

```bash
python -m pip install -e '.[evaluation]'
```

Score prediction JSONL against the official `MassSpecGym.tsv` or a column-equivalent CSV export:

```bash
python -m evaluation.massspecgym_official denovo \
  --reference data/MassSpecGym.tsv --predictions runs/structures.jsonl \
  --output runs/denovo_metrics.json

python -m evaluation.massspecgym_official simulation \
  --reference data/MassSpecGym.tsv --predictions runs/spectra.jsonl \
  --output runs/simulation_metrics.json
```

De novo scoring uses the official top-1/top-10 InChIKey exact match, maximum Morgan Tanimoto, and MCES distance definitions. Simulation scoring uses the official simulation-challenge test subset, `SpecToMzsInts` reference-spectrum transform, and sparse cosine, square-root cosine, and Jensen–Shannon metrics. Prediction records should carry the MassSpecGym `identifier`; older test-order outputs can use `--legacy-index-order` after a SMILES identity check. The scorer reports how many test examples were covered.

## Repository contents

- `molspecflow/`: model, flow matcher, spectrum loss, formula utilities, and inference CLI
- `training/`: data loaders, training entry point, and pretraining/fine-tuning configs
- `evaluation/`: offline MassSpecGym metric adapter
