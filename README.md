# PriSM Anonymous Implementation

This repository provides the anonymous implementation of **PriSM** for traffic forecasting.

PriSM is a prior-separated selective state-space model that decouples physical structural priors and lag-aware causal-context priors. The repository contains the model implementation, data loading pipeline, training and evaluation scripts, evaluation metrics, and the causal-prior construction script used in the submitted paper.

## Directory structure

```text
.
├── PriSM.py
├── main.py
├── prepare.py
├── trainer.py
├── metrics.py
├── pscan.py
├── build_c_prior.py
├── mamba_ssm/
├── scripts/
└── datasets/
```

The main files are listed below.

- `PriSM.py`: implementation of the PriSM model.
- `main.py`: training and evaluation entry point.
- `prepare.py`: dataset loading and preprocessing.
- `trainer.py`: training, validation, testing, and checkpoint logic.
- `metrics.py`: evaluation metrics.
- `pscan.py`: lightweight parallel scan implementation used by the selective state-space block.
- `build_c_prior.py`: script for constructing lag-aware causal priors.
- `mamba_ssm/`: cleaned compatibility directory for selective state-space related implementations.
- `scripts/`: example running scripts.
- `datasets/`: placeholder directory for datasets.

The raw datasets are not included in this anonymous repository. Please place the datasets under `./datasets/` before running the code.

## Expected dataset layout

```text
datasets/
├── PEMS03/
│   ├── PEMS03.npz
│   ├── PEMS03.csv
│   └── PEMS03_pcmci_L{lag}_alpha{x}_k{x}.npz
├── PEMS08/
│   ├── PEMS08.npz
│   ├── PEMS08.csv
│   └── PEMS08_pcmci_L{lag}_alpha{x}_k{x}.npz
├── HZMetro/
│   ├── graph_hz_conn.pkl
│   ├── train.pkl
│   ├── val.pkl
│   ├── test.pkl
│   └── HZMetro_pcmci_L{lag}_alpha{x}_k{x}.npz
└── SHMetro/
    ├── graph_sh_conn.pkl
    ├── train.pkl
    ├── val.pkl
    ├── test.pkl
    └── SHMetro_pcmci_L{lag}_alpha{x}_k{x}.npz
```

Here, `{lag}`, `{x}` in `alpha{x}`, and `{x}` in `k{x}` are placeholders for the causal-prior construction settings. For example, the following filenames are both valid:

```text
PEMS08_pcmci_L3_alpha0.01_k10.npz
PEMS08_pcmci_L2_alpha0.05_k20.npz
```

The repository supports different causal-prior hyperparameters. For paper-style reproduction, please use the causal-prior settings specified in the running scripts or explicitly pass the desired prior file with `--causal_path`.

If the causal-prior file is not automatically found or uses a different filename, pass it explicitly with `--causal_path`.

For HZMetro and SHMetro, the physical supports are loaded from `graph_hz_conn.pkl` and `graph_sh_conn.pkl`, respectively. If a graph file is stored elsewhere, pass it with `--graph_path`.

## Lugu dataset

Lugu is a newly constructed urban-road traffic dataset used in the submitted paper. The full de-identified version is not included in this anonymous repository because the data release process is still under privacy and institutional review.

This repository provides the expected data format, preprocessing pipeline, and causal-prior construction script to support reproducibility checking. Detailed dataset statistics and construction information are reported in the appendix of the submitted paper.

The full de-identified Lugu dataset, together with the processed road-network support and preprocessing scripts, is planned to be released after paper acceptance once the data release procedure is completed.

A recommended layout for Lugu is:

```text
datasets/
└── Lugu/
    ├── Lugu.npz
    ├── Lugu.csv
    └── Lugu_pcmci_L{lag}_alpha{x}_k{x}.npz
```

## Environment

Install the required dependencies with:

```bash
pip install -r requirements.txt
```

The repository includes `pscan.py` and a cleaned `mamba_ssm/` directory for compatibility. The PriSM implementation mainly uses a task-adapted selective state-space block with a lightweight parallel scan implementation in `pscan.py`, rather than directly calling the official `Mamba` class.

## Training and evaluation

Example scripts are provided under `scripts/`:

```bash
bash scripts/run_pems03.sh
bash scripts/run_pems08.sh
bash scripts/run_hzmetro.sh
bash scripts/run_shmetro.sh
```

For the standard forecasting setting reported in the main result table, use `--pred_len 12`. For the extended-horizon setting, set `--pred_len` to the evaluated horizon, such as 24, 36, or 48.

Example command for the standard 12-step setting:

```bash
python main.py \
  --dataset PEMS08 \
  --data_root ./datasets \
  --pred_len 12
```

Example command for an extended 48-step setting:

```bash
python main.py \
  --dataset PEMS08 \
  --data_root ./datasets \
  --pred_len 48
```

For metro datasets, specify the physical graph path when necessary:

```bash
python main.py \
  --dataset HZMetro \
  --data_root ./datasets \
  --graph_path ./datasets/HZMetro/graph_hz_conn.pkl \
  --pred_len 12
```

If the causal-prior file is not automatically found or uses a different filename, pass it explicitly:

```bash
python main.py \
  --dataset PEMS08 \
  --data_root ./datasets \
  --causal_path ./datasets/PEMS08/PEMS08_pcmci_L3_alpha0.01_k10.npz \
  --pred_len 12
```

During local testing, an external data path can be specified with `--data_root`. The default examples above use `./datasets` for portability.

## Build lag-aware causal prior

Lag-aware causal priors can be constructed using:

```bash
python build_c_prior.py \
  --dataset Lugu \
  --data_root ./datasets \
  --max_lag 2 \
  --alpha_level 0.01 \
  --topk_neighbors 10 \
  --downsample_step 3
```

The generated causal-prior file is saved under the corresponding dataset directory. The recommended naming pattern is:

```text
{dataset}_pcmci_L{max_lag}_alpha{alpha_level}_k{topk_neighbors}.npz
```

For example:

```text
Lugu_pcmci_L2_alpha0.01_k10.npz
PEMS08_pcmci_L3_alpha0.05_k20.npz
```

This filename is only a convention. Any causal-prior file can be used by passing its path through `--causal_path`.

The lag-aware prior is constructed offline from the training split only and is kept fixed during model training, validation, and testing. It is used as a contextual semantic prior rather than as a ground-truth causal graph or a direct propagation adjacency matrix.

## Notes

Raw datasets, checkpoints, logs, running outputs, TensorBoard files, and model weights are not included in this anonymous repository.

This repository is intended for anonymous review and reproducibility checking. A fully organized public release will be prepared after the review process.
