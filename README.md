<h1 align="center"><img src="assets/logo.png" alt="MolSpecFlow icon" width="56"> MolSpecFlow</h1>

<p align="center">Molecular structure and tandem mass spectrum modeling with flow matching.</p>

<p align="center"><a href="#quick-start">Quick start</a> · <a href="#inference-tasks">Inference tasks</a> · <a href="#training">Training</a> · <a href="#massspecgym-evaluation">Evaluation</a></p>

MolSpecFlow connects MS/MS spectra, molecular structures, and molecular fingerprints in one model. This repository contains the model, inference command, pretraining and fine-tuning configurations, and a MassSpecGym metric adapter.

| Task | Input | Output |
| --- | --- | --- |
| `spec_to_smiles` | MS/MS spectrum | SMILES candidates |
| `smiles_to_spec` | SMILES | Predicted m/z and intensity arrays |
| `spec_to_ecfp` | MS/MS spectrum | 2048-bit molecular fingerprint |

## Quick start

Python 3.10 or newer is required. Install the inference dependencies from the repository root:

```bash
python -m pip install -e '.[inference]'
```

Put a compatible checkpoint at `checkpoints/model.ckpt` and its matching Hugging Face tokenizer at `tokenizer/`. Use a checkpoint fine-tuned for the task you want to run. If it does not embed its training configuration, add `--config training/configs/<matching-config>.yaml`. Only load checkpoints and tokenizers from sources you trust.

For spectrum-to-structure inference, create `inputs/spectra.jsonl` with one spectrum per line:

```json
{"identifier":"example_1","precursor_mz":181.0707,"adduct":"[M+H]+","collision_energy":30,"mz":[43.0184,77.0386,105.0335],"intensity":[0.3,0.7,1.0]}
```

Then generate ten molecular candidates:

```bash
python -m molspecflow.infer \
  --task spec_to_smiles \
  --checkpoint checkpoints/model.ckpt --tokenizer tokenizer \
  --input inputs/spectra.jsonl --output runs/structures.jsonl \
  --num-samples 10 --steps 100
```

The output is JSONL. Each record contains the input `identifier`, an ordered `predictions` list of decoded SMILES, and `raw_deepsmiles`. Invalid decoded structures appear as `null`.

## Inference tasks

### Spectrum → structure or fingerprint

Use the spectrum JSONL format above for both `spec_to_smiles` and `spec_to_ecfp`. You may provide peaks as `"peaks": [[mz, intensity], ...]` instead of separate `mz` and `intensity` arrays. An optional `formula` supplies molecular-formula conditioning; without it, the formula vector is zero. Spectra are sorted by intensity and truncated or padded to 20 peaks.

For fingerprint prediction, change the task and output path:

```bash
python -m molspecflow.infer \
  --task spec_to_ecfp \
  --checkpoint checkpoints/model.ckpt --tokenizer tokenizer \
  --input inputs/spectra.jsonl --output runs/fingerprints.jsonl
```

Each fingerprint record contains `ecfp_bits`.

### Structure → spectrum

Create `inputs/molecules.jsonl`:

```json
{"identifier":"example_2","smiles":"CCO","adduct":"[M+H]+","collision_energy":30}
```

Run:

```bash
python -m molspecflow.infer \
  --task smiles_to_spec \
  --checkpoint checkpoints/model.ckpt --tokenizer tokenizer \
  --input inputs/molecules.jsonl --output runs/spectra.jsonl
```

Each spectrum record contains `pred_mz` and `pred_int` arrays. The command also accepts `--device cpu` or `--device cuda`, and `--seed` for reproducibility.

## Training

Install the training dependencies:

```bash
python -m pip install -e '.[training]'
```

The default configurations use `data/raw/mass_spec_gym.csv`, `data/raw/GeMS_A.csv`, `data/raw/zinc100m/{train,valid,test}.csv`, and `tokenizer/`. The paired-spectrum CSV needs `smiles`, `spectrum`, `PARENT_MASS`, `ADDUCT`, `COLLISION_ENERGY`, and `FOLD` columns. Override locations with `--data-root`, `--tokenizer`, and `--output-dir`.

Start joint pretraining:

```bash
python -m training.mptrain --config training/configs/mppretrain_joint_v2.yaml
```

Fine-tune from a compatible joint checkpoint for each downstream task:

```bash
python -m training.mptrain --config training/configs/mpfinetune_spec_to_smiles.yaml --pretrained checkpoints/joint.ckpt
python -m training.mptrain --config training/configs/mpfinetune_smiles_to_spec.yaml --pretrained checkpoints/joint.ckpt
python -m training.mptrain --config training/configs/mpfinetune_spec_to_ecfp.yaml --pretrained checkpoints/joint.ckpt
```

The supplied configurations use eight devices by default. Adjust `training.devices`, batch size, and worker count for your hardware; `--debug` runs a small single-device check.

### Training objectives

Structure prediction combines SMILES token cross-entropy with total-mass and molecular-formula Huber losses. Fingerprint prediction uses a regression loss. Spectrum generation combines m/z, intensity, and multi-resolution binned-cosine losses. There is no separate molecular graph-topology loss.

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

De novo scoring uses top-1/top-10 InChIKey exact match, maximum Morgan Tanimoto, and MCES distance. Simulation scoring uses the simulation-challenge test subset, the `SpecToMzsInts` reference-spectrum transform, and sparse cosine, square-root cosine, and Jensen–Shannon metrics. Prediction records should carry the MassSpecGym `identifier`; older test-order outputs can use `--legacy-index-order` after a SMILES identity check. The scorer reports test-example coverage.

## Repository layout

- `molspecflow/` — model, flow matcher, losses, formula utilities, and inference CLI
- `training/` — data loaders, training entry point, and pretraining/fine-tuning configurations
- `evaluation/` — MassSpecGym metric adapter
