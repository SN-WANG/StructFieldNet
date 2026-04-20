# StructFieldNet

[![Role](https://img.shields.io/badge/Role-Research%20Code-0f766e)](https://github.com/SN-WANG/StructFieldNet)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

**StructFieldNet** is the structural-field repository in the WSNet family. It inherits the same lightweight model, training, and utility foundations from [WSNet](https://github.com/SN-WANG/WSNet), while focusing on fixed-mesh stress field reconstruction from grouped design parameters.

## 📌 Overview

StructFieldNet keeps the full workflow for this task in one place:
dataset handling, model training, case-wise inference, visualization, and metric export.

The current scope includes:

- design-conditioned structural field reconstruction
- fixed-mesh stress prediction
- end-to-end training and inference workflows
- case-wise visualization and diagnostic metrics

## ✨ Highlights

- Full-field stress reconstruction from grouped structural design parameters
- `StructFieldNet` as the main model for fixed-mesh structural learning
- Unified `main.py` workflow for probe, train, and infer
- Deterministic dataset splitting and normalization pipeline
- Stable MSE training with mixed precision, gradient clipping, and checkpointing
- Case-wise evaluation with `mse`, `rmse`, `mae`, `r2`, `accuracy`, and hotspot-oriented metrics
- PyVista comparison figures for ground truth, prediction, and absolute error

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

### Install the project requirements

```bash
pip install -r requirements.txt
```

### Probe GPU memory before training

```bash
python main.py --mode probe --data_dir ./dataset --output_dir ./runs
```

### Train StructFieldNet

```bash
python main.py \
  --mode train \
  --data_dir ./dataset \
  --output_dir ./runs
```

### Run inference and generate visualizations

```bash
python main.py \
  --mode infer \
  --data_dir ./dataset \
  --output_dir ./runs
```

## 📂 Expected Data Format

```text
dataset/
├── dp1.pt
├── dp2.pt
├── dp3.pt
└── ...
```

Each case file should be a PyTorch dictionary containing:

- `coords`: tensor of shape `(N, 3)`
- `design`: tensor of shape `(25,)`
- `stress`: tensor of shape `(N, 1)`

All samples are assumed to share the same reference mesh coordinates.

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
