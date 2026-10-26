<h1 align="center"><img src="assets/logo.png" alt="MolSpecFlow icon" width="48"> MolSpecFlow</h1>

<p align="center"><strong>From tandem mass spectra to molecular structures—and back.</strong></p>

<p align="center"><a href="#quick-start">Quick start</a> · <a href="#other-inference-tasks">Inference</a> · <a href="#training">Training</a> · <a href="#massspecgym-evaluation">Evaluation</a></p>

MolSpecFlow is a flow-matching model for joint molecular and MS/MS modeling. It supports three directions:

- **Spectrum → structure:** generate SMILES candidates from an MS/MS spectrum.
- **Structure → spectrum:** predict fragment m/z values and intensities from a SMILES string.
- **Spectrum → fingerprint:** predict a 2048-bit molecular fingerprint.

The repository includes the model, an inference CLI, pretraining and fine-tuning configurations, and MassSpecGym evaluation code.

## Quick start

### 1. Install

Use Python 3.10 or newer. From the repository root:

```bash
python -m pip install -e '.[inference]'
```

Place a task-compatible checkpoint at `checkpoints/model.ckpt` and its matching tokenizer at `tokenizer/`.

### 2. Prepare a spectrum

Create `inputs/spectra.jsonl` with one JSON object per line:

```json
{"identifier":"example_1","precursor_mz":181.0707,"adduct":"[M+H]+","collision_energy":30,"mz":[43.0184,77.0386,105.0335],"intensity":[0.3,0.7,1.0]}
```

### 3. Generate structures

Generate ten candidate structures:

```bash
python -m molspecflow.infer \
  --task spec_to_smiles \
  --checkpoint checkpoints/model.ckpt --tokenizer tokenizer \
  --input inputs/spectra.jsonl --output runs/structures.jsonl \
  --num-samples 10 --steps 100
```

The output is JSONL: each record includes `identifier`, a `predictions` list of SMILES, and `raw_deepsmiles`. Invalid structures appear as `null`.

## Other inference tasks

### Spectrum → fingerprint

Use the same spectrum input with `spec_to_ecfp`:

```bash
python -m molspecflow.infer \
  --task spec_to_ecfp \
  --checkpoint checkpoints/model.ckpt --tokenizer tokenizer \
  --input inputs/spectra.jsonl --output runs/fingerprints.jsonl
```

The output contains `ecfp_bits`.

### Structure → spectrum

Create `inputs/molecules.jsonl` with a SMILES string:

```json
{"identifier":"example_2","smiles":"CCO","adduct":"[M+H]+","collision_energy":30}
```

```bash
python -m molspecflow.infer \
  --task smiles_to_spec \
  --checkpoint checkpoints/model.ckpt --tokenizer tokenizer \
  --input inputs/molecules.jsonl --output runs/spectra.jsonl
```

The output contains `pred_mz` and `pred_int` arrays.

### Input and checkpoint notes

- Spectrum input also accepts `"peaks": [[mz, intensity], ...]` instead of separate arrays. Peaks are sorted by intensity and truncated or padded to 20.
- An optional `formula` supplies molecular-formula conditioning. Without it, the formula vector is zero.
- Use a checkpoint trained for the selected task. If its configuration is not embedded, pass `--config training/configs/<matching-config>.yaml`. Load checkpoints and tokenizers only from sources you trust.
- Set `--device cpu` or `--device cuda`; `--seed` controls reproducibility.

## Training

Install the training dependencies:

```bash
python -m pip install -e '.[training]'
```

### Joint pretraining

```bash
python -m training.mptrain --config training/configs/mppretrain_joint_v2.yaml
```

### Task fine-tuning

Initialize from a compatible joint checkpoint. Choose the configuration for your task:

| Task | Configuration |
| --- | --- |
| Spectrum → structure | `training/configs/mpfinetune_spec_to_smiles.yaml` |
| Structure → spectrum | `training/configs/mpfinetune_smiles_to_spec.yaml` |
| Spectrum → fingerprint | `training/configs/mpfinetune_spec_to_ecfp.yaml` |

```bash
python -m training.mptrain \
  --config training/configs/mpfinetune_spec_to_smiles.yaml \
  --pretrained checkpoints/joint.ckpt
```

### Data and compute

Default paths are `data/raw/mass_spec_gym.csv`, `data/raw/GeMS_A.csv`, `data/raw/zinc100m/{train,valid,test}.csv`, and `tokenizer/`. The paired-spectrum CSV needs `smiles`, `spectrum`, `PARENT_MASS`, `ADDUCT`, `COLLISION_ENERGY`, and `FOLD` columns. Override locations with `--data-root`, `--tokenizer`, and `--output-dir`.

The supplied configurations use eight devices by default. Adjust `training.devices`, batch size, and worker count for your hardware. Use `--debug` for a small single-device check.

### Training objectives

| Output | Losses |
| --- | --- |
| Molecular structure | SMILES token cross-entropy; total-mass and molecular-formula Huber |
| Molecular fingerprint | Regression |
| MS/MS spectrum | m/z, intensity, and multi-resolution binned cosine |

There is no separate molecular graph-topology loss.

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

The adapter reports test-example coverage and follows the task-specific MassSpecGym metrics:

- **De novo:** top-1/top-10 InChIKey exact match, maximum Morgan Tanimoto, and MCES distance.
- **Simulation:** sparse cosine, square-root cosine, and Jensen–Shannon metrics on the simulation-challenge test subset, using the `SpecToMzsInts` reference-spectrum transform.

Prediction records should carry the MassSpecGym `identifier`. Older test-order outputs can use `--legacy-index-order` after a SMILES identity check.

## Repository layout

- `molspecflow/` — model, flow matcher, losses, formula utilities, and inference CLI
- `training/` — data loaders, training entry point, and pretraining/fine-tuning configurations
- `evaluation/` — MassSpecGym metric adapter
