# Multi-Agent Trajectory Forecasting

UCLA ECE C247A Final Project

This project implements multi-agent trajectory forecasting models using LSTM and Transformer architectures for predicting vehicle trajectories in autonomous driving scenarios, based on the Argoverse 2 dataset.

> **Note:** Most pre-generated outputs (prediction plots, animated videos, evaluation results) are already available in the `outputs/` folder — you can browse them without running anything.

---

## Installation

1. Clone the repository:
   ```bash
   git clone git@github.com:michae-lzhou/Multi-Agent-Trajectory-Forecasting.git
   cd Multi-Agent-Trajectory-Forecasting
   ```

2. Install the package in development mode:
   ```bash
   pip install -e .
   ```

   This installs all required dependencies: `av2`, `torch`, `numpy`, `pandas`, `matplotlib`, and `tqdm`.

---

## Data Preparation

The project uses the [Argoverse 2 Motion Forecasting dataset](https://www.argoverse.org/av2.html).

### 1. Download and organize the data

```
data/
├── raw/
│   ├── train/
│   │   └── scenario_*.parquet  (+ map JSONs)
│   ├── val/
│   │   └── scenario_*.parquet  (+ map JSONs)
│   └── test/
│       └── scenario_*.parquet  (+ map JSONs)
└── processed/
    ├── train/
    ├── val/
    └── test/
```

### 2. Preprocess the data

```bash
# Split and sample from the raw dataset
python scripts/run_partition.py
# Preprocess train and val splits with neighbor cap of 22 (matches the best model)
python scripts/run_preprocess.py --splits train val --neighbor_cap 22 --data_prefix baseline
```

You can adjust `--neighbor_cap` and `--data_prefix` to experiment with different configurations.

---

## Best Model

The best performing model is a **Transformer** with the following configuration:

| Hyperparameter | Value |
|---|---|
| Hidden size | 256 |
| Layers | 2 |
| Attention heads | 2 |
| Residual connections | True |
| Layer normalization | None |
| Cell state initialization | Focal |
| Neighbor cap | 22 |

Its checkpoint lives at:
```
checkpoints/ncap=22/transformer_h256_heads2_resTrue_normno_norm_cell_statefocal/
```

---

## Evaluation

Run evaluation on the validation set:

```bash
python scripts/run_eval.py \
    --model transformer \
    --config checkpoints/ncap=22/transformer_h256_heads2_resTrue_normno_norm_cell_statefocal/config.yaml \
    --checkpoint checkpoints/ncap=22/transformer_h256_heads2_resTrue_normno_norm_cell_statefocal/best_model.pt \
    --data_dir data/processed/val \
    --neighbor_cap 22 \
    --data_prefix baseline
```

**Output metrics:**
- **minADE** — Minimum Average Displacement Error
- **minFDE** — Minimum Final Displacement Error
- **MissRate** — Fraction of predictions that miss the ground truth by more than a threshold

---

## Visualizations

> Pre-generated visualizations for the best model are already in `outputs/` — check there before running these scripts.

### Static Prediction Plots

Generates a multi-panel plot of predicted trajectories across several scenarios:

```bash
python scripts/viz_predictions.py \
    --config checkpoints/ncap=22/transformer_h256_heads2_resTrue_normno_norm_cell_statefocal/config.yaml \
    --checkpoint checkpoints/ncap=22/transformer_h256_heads2_resTrue_normno_norm_cell_statefocal/best_model.pt \
    --n_scenes 6 \
    --output outputs/plots/predictions.png \
    --shuffle
```

**Color legend:**
-  Blue — historical trajectory of the focal agent
-  Green — ground truth future trajectory
-  Red — model predictions
-  Gray — neighboring agents

### Animated Video Visualizations

Generates MP4 videos with predictions overlaid on the HD map:

```bash
python scripts/viz_video.py \
    --config checkpoints/ncap=22/transformer_h256_heads2_resTrue_normno_norm_cell_statefocal/config.yaml \
    --checkpoint checkpoints/ncap=22/transformer_h256_heads2_resTrue_normno_norm_cell_statefocal/best_model.pt \
    --scenario_dir data/raw/val \
    --processed_dir data/processed/val \
    --output_dir outputs/visualizations/videos \
    --n_scenes 6 \
    --fps 10 \
    --shuffle
```

**Color legend:**
-  Red rectangles — focal agent
-  Orange trajectory — model predictions
-  Green trajectory — ground truth
-  Blue rectangles — other vehicles
-  Purple rectangles — pedestrians

---

## Training New Models

```bash
python scripts/run_train.py --config configs/transformer.yaml
```

Edit the YAML files in `configs/` to experiment with different architectures and hyperparameters. Available config options include hidden size, number of layers, attention heads, residual connections, normalization type, and cell state initialization.

---

## Project Structure

```
├── src/matf/
│   ├── data/         # Data loading and preprocessing
│   ├── models/       # LSTM and Transformer implementations
│   ├── training/     # Training loop and utilities
│   └── utils/        # Metrics, config parsing, helpers
├── scripts/          # Entry-point scripts (train, eval, visualize)
├── configs/          # YAML config files
├── checkpoints/      # Saved model checkpoints
├── data/             # Raw and processed datasets
└── outputs/          # Pre-generated plots and videos
```

---

## Citation

If you use this code in your research, please cite this project appropriately.
