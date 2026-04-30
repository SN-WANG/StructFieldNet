# StructFieldNet

[![Role](https://img.shields.io/badge/Role-Research%20Code-0f766e)](https://github.com/SN-WANG/StructFieldNet)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

**StructFieldNet** is the structural-field repository in the WSNet family. It keeps the fixed-mesh stress-field reconstruction workflow local to this repository while reusing the lightweight training, normalization, and utility style of [WSNet](https://github.com/SN-WANG/WSNet).

## 📌 Overview

StructFieldNet keeps the full workflow for this task in one place:
dataset handling, memory probing, model training, case-wise inference, visualization, video export, and metric export.

The current scope includes:

- design-conditioned structural stress-field reconstruction
- fixed-mesh full-field stress prediction
- end-to-end training and inference workflows
- case-wise comparison visualization and MP4 animation
- diagnostic metrics for full-field and hotspot reconstruction quality

## ✨ Highlights

- `StructFieldNet` as the main model for fixed-mesh structural learning
- Unified `main.py` workflow for probe, train, and infer
- Deterministic train, validation, and test splitting with reusable split manifests
- Coordinate, design, and stress normalization restored from checkpoints during inference
- Stable MSE training with mixed precision, gradient clipping, cosine scheduling, and checkpointing
- Case-wise evaluation with `mse`, `rmse`, `mae`, `r2`, `accuracy`, and hotspot-oriented metrics
- PyVista comparison figures for ground truth, prediction, and absolute error
- MP4 comparison loop across all inferred test cases

## 🧱 Repository Layout

```text
StructFieldNet/
├── main.py                  # Unified entry point for probe / train / infer
├── config.py                # Command-line arguments and experiment configuration
├── models/
│   └── fieldnet.py
├── data/
│   ├── field_data.py
│   ├── field_metrics.py
│   ├── field_plot.py
│   └── field_vis.py
├── training/
│   ├── base_trainer.py
│   └── field_trainer.py
├── utils/
│   ├── scaler.py
│   ├── hue_logger.py
│   ├── seeder.py
│   └── sweeper.py
├── README.md
└── LICENSE
```

## 🚀 Running Experiments

### Clone the repository

```bash
git clone https://github.com/SN-WANG/StructFieldNet.git
cd StructFieldNet
```

### Install the dependencies you need

```bash
pip install numpy torch matplotlib tqdm pyvista pillow
```

MP4 rendering uses system `ffmpeg`. Make sure `ffmpeg` is on `PATH`.

### Probe GPU memory before training

```bash
python main.py --mode probe --data_dir ./dataset --output_dir ./runs
```

### Train StructFieldNet

```bash
python main.py --mode train --data_dir ./dataset --output_dir ./runs
```

### Run inference and generate visualizations

```bash
python main.py --mode infer --data_dir ./dataset --output_dir ./runs
```

This writes per-case comparison figures and, by default, a global MP4 loop across all inferred test cases.

### Run the full workflow

```bash
python main.py --mode probe train infer --data_dir ./dataset --output_dir ./runs
```

## 📂 Expected Data Format

The default workflow expects fixed-mesh structural cases named `dp<label>.pt`.
All samples are assumed to share one reference mesh coordinate tensor.

```text
dataset/
├── dp1.pt
├── dp2.pt
├── dp3.pt
└── ...
```

Each case file should be a PyTorch dictionary containing:

- `coords`: tensor of shape `(N, 3)`
- `design`: tensor of shape `(25,)` with grouped structural design parameters
- `stress`: tensor of shape `(N, 1)` with the nodal scalar stress field

## 🧾 Workflow Outputs

```text
runs/
├── ckpt.pt
├── best.pt
├── config.json
├── splits.json
├── history.json
├── test_metrics.json
├── test_summary.json
├── training_curve.png
├── metrics_summary.png
├── dp<label>_pred.pt
├── dp<label>_comparison.png
└── inference_comparison_loop.mp4
```

Checkpoints store model arguments, split metadata, and data metadata in `params`, while coordinate, design, and stress scalers are stored separately in `scaler_state_dict`.

## 🔗 Relationship to WSNet

StructFieldNet is built on top of [WSNet](https://github.com/SN-WANG/WSNet).
WSNet keeps the reusable core modules, while StructFieldNet keeps the structural dataset pipeline, task-specific model entry point, and experiment workflow.

## 📚 Citation

If this repository is useful in your work, please cite it as a software project.

```bibtex
@software{structfieldnet2026,
  author = {Shengning Wang},
  title = {StructFieldNet},
  year = {2026},
  url = {https://github.com/SN-WANG/StructFieldNet}
}
```

## 📄 License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
