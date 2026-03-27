# StructFieldNet

StructFieldNet is a PyTorch implementation for **scalar nodal von Mises stress field reconstruction** on an unstructured wing mesh. The model follows the methodology described in the accompanying paper: a **DeepONet-style design-to-node lifting module** is coupled with a **Transolver-style Physics-Attention backbone** to learn the mapping

\[
(\mathbf{P}, \mathbf{d}) \rightarrow \mathbf{s},
\]

where:

- `P` is the reference mesh coordinate field,
- `d` is the grouped thickness design vector,
- `s` is the scalar nodal stress field under the gravity loading condition.

The repository is organized as a lightweight research codebase aligned with the structure and dependencies of the author's broader `WSNet` ecosystem.

## Highlights

- Clean PyTorch implementation of **StructFieldNet**
- Modular project layout for **data**, **models**, **losses**, **trainers**, and **utilities**
- JSON-based experiment configuration
- Mixed-precision-ready training loop with checkpointing and early stopping
- Built-in reproducibility utilities (`seed_everything`)
- Automatic handling of occasional **mixed SI-prefix units** in ANSYS exports
- Minimal dependency footprint, consistent with the existing WSNet environment

## Method Overview

StructFieldNet consists of four main stages:

1. **Design-to-node feature lifting**
   - A branch MLP encodes the global thickness vector `d`.
   - A trunk MLP encodes the node coordinates `P`.
   - The two embeddings are fused into node-wise latent features.

2. **Physics-Attention backbone**
   - Multiple Transolver-style Physics-Attention blocks operate on the node features.
   - Each block performs slice assignment, slice aggregation, slice self-attention, and node-wise residual updates.

3. **Output projection**
   - A linear head predicts the scalar nodal stress at every mesh node.

4. **Field-aware supervision**
   - Training uses a global field loss and a hotspot-weighted loss to improve accuracy in high-stress regions.

## Repository Structure

```text
StructFieldNet/
├── configs/
│   └── default.json
├── src/
│   └── structfieldnet/
│       ├── data/
│       ├── losses/
│       ├── models/
│       ├── trainers/
│       └── utils/
├── tests/
├── main.py
├── README.md
└── requirements.txt
```

## Requirements

The code is designed to stay close to the dependency set already used in `WSNet`.

Core packages:

- `torch`
- `numpy`
- `tqdm`

Optional but WSNet-aligned scientific packages:

- `scipy`
- `pandas`
- `matplotlib`
- `pyvista`

Install dependencies with:

```bash
/Users/wsn/pyenv/bin/python -m pip install -r requirements.txt
```

## Dataset Format

Each case is stored as one `.pt` file and must contain exactly three tensors:

- `coords`: `FloatTensor` of shape `(N, 3)`
- `design`: `FloatTensor` of shape `(M,)`
- `stress`: `FloatTensor` of shape `(N, 1)`

For the current wing benchmark:

- `M = 25`
- `N = 11234`

The default configuration expects the dataset to be located at:

```text
/Users/wsn/lab/rely_opt/dataset
```

If your dataset is stored elsewhere, update `paths.dataset_dir` in [`configs/default.json`](configs/default.json).

## Dataset Notes

The current ANSYS export contains one case with a different SI-prefix scale from the rest of the dataset:

- coordinates: millimeter-scale export instead of meter-scale export
- stress: MPa-scale export instead of Pa-scale export

To keep the training pipeline robust, the dataset loader performs **automatic SI-prefix harmonization** before mesh verification and normalization. This correction is conservative:

- it only applies when the mismatch spans at least one full decade in log-space
- it selects the closest multiplier from standard SI-prefix candidates (`1e-12` to `1e12`)

This behavior can be toggled with:

```json
"harmonize_units": true
```

in the data section of the config.

## Configuration

All experiment settings are stored in [`configs/default.json`](configs/default.json), including:

- dataset paths
- train/validation/test split ratios
- normalization settings
- model width and depth
- loss weights
- optimizer and scheduler hyperparameters
- training device and AMP options

Important defaults:

- coordinate normalization: `[-1, 1]`
- design normalization: standardization
- stress normalization: standardization
- mesh consistency check: enabled
- unit harmonization: enabled

## Quick Start

### Train

```bash
cd StructFieldNet
python main.py --config configs/default.json --mode train
```

### Evaluate

```bash
cd StructFieldNet
python main.py --config configs/default.json --mode eval
```

## Output Artifacts

Training artifacts are written to the directory specified by `paths.output_dir`. By default:

```text
/Users/wsn/Documents/StructFieldNet/runs/default
```

The trainer saves:

- `config.json`: frozen experiment configuration
- `splits.json`: train/validation/test case split
- `last.pt`: latest checkpoint
- `best.pt`: best validation checkpoint
- `history.json`: epoch-wise training history
- `test_metrics.json`: evaluation metrics on the test split

## Reproducibility

The code uses the local `seed_everything` utility to fix:

- Python random state
- NumPy random state
- PyTorch CPU and CUDA random states
- cuDNN deterministic behavior

The global seed is configured in [`configs/default.json`](configs/default.json).

## Verification Status

The current codebase has been checked with:

- module compilation via `python -m compileall`
- real-data dataloader construction
- one-batch forward pass
- one-batch loss computation
- one-batch backward pass

This verifies that the main training path is connected correctly for the provided dataset format.

## Citation

If you use this repository in academic work, please cite the corresponding StructFieldNet paper once it is publicly available.
