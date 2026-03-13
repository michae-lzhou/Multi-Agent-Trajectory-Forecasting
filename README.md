# Multi-Agent Trajectory Forecasting

UCLA ECE C247A Final Project

This project implements multi-agent trajectory forecasting models using LSTM and Transformer architectures for predicting vehicle trajectories in autonomous driving scenarios, based on the Argoverse 2 dataset.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Multi-Agent-Trajectory-Forecasting
   ```

2. Install the package in development mode:
   ```bash
   pip install -e .
   ```

   This will install the required dependencies including `av2`, `torch`, `numpy`, `pandas`, `matplotlib`, and `tqdm`.

## Data Preparation

The project uses the Argoverse 2 Motion Forecasting dataset. You need to download and organize the data as follows:

1. Download the Argoverse 2 dataset from the official source.

2. Organize the data in the following structure:
   ```
   data/
   ├── raw/
   │   ├── train/
   │   │   └── scenario_*.parquet (and corresponding map JSONs)
   │   ├── val/
   │   │   └── scenario_*.parquet (and corresponding map JSONs)
   │   └── test/
   │       └── scenario_*.parquet (and corresponding map JSONs)
   └── processed/
       ├── train/
       ├── val/
       └── test/
   ```

3. Preprocess the data:
   ```bash
   # For training/validation data with neighbor cap of 22 (matching the best model)
   python scripts/run_preprocess.py --splits train val --neighbor_cap 22 --data_prefix baseline

   # You can also preprocess with different neighbor caps or data prefixes as needed
   ```

## Evaluating the Best Model

The best performing model is a Transformer with the following configuration:
- Hidden size: 256
- Number of layers: 2
- Number of heads: 2
- Residual connections: True
- Layer normalization: No normalization
- Cell state initialization: Focal
- Neighbor cap: 22

To evaluate this model on the validation set:

```bash
python scripts/run_eval.py \
    --model transformer \
    --config checkpoints/ncap=22/transformer_h256_heads2_resTrue_normno_norm_cell_statefocal/config.yaml \
    --checkpoint checkpoints/ncap=22/transformer_h256_heads2_resTrue_normno_norm_cell_statefocal/best_model.pt \
    --data_dir data/processed/val \
    --neighbor_cap 22 \
    --data_prefix baseline
```

This will output the evaluation metrics:
- minADE (Minimum Average Displacement Error)
- minFDE (Minimum Final Displacement Error)
- MissRate

## Visualizations

### Static Prediction Plots

Generate static plots showing predicted trajectories for multiple scenarios:

```bash
python scripts/viz_predictions.py \
    --config checkpoints/ncap=22/transformer_h256_heads2_resTrue_normno_norm_cell_statefocal/config.yaml \
    --checkpoint checkpoints/ncap=22/transformer_h256_heads2_resTrue_normno_norm_cell_statefocal/best_model.pt \
    --n_scenes 6 \
    --output outputs/plots/predictions.png \
    --shuffle
```

This creates a plot file showing:
- Blue: Historical trajectory of the focal agent
- Green: Ground truth future trajectory
- Red: Model predictions
- Gray: Neighbor agents

### Animated Video Visualizations

Generate animated MP4 videos showing predictions overlaid on the map:

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

This creates video files showing:
- Red rectangles: Focal agent
- Orange trajectory: Model predictions
- Green trajectory: Ground truth
- Blue rectangles: Other vehicles
- Purple rectangles: Pedestrians

## Training New Models

To train a new model, use the training script:

```bash
python scripts/run_train.py --config configs/transformer.yaml
```

Modify the configuration files in `configs/` to experiment with different architectures and hyperparameters.

## Project Structure

- `src/matf/`: Main package code
  - `data/`: Data loading and preprocessing
  - `models/`: Model implementations (LSTM, Transformer)
  - `training/`: Training utilities
  - `utils/`: Metrics, configuration, and helpers
- `scripts/`: Executable scripts for training, evaluation, and visualization
- `configs/`: YAML configuration files
- `checkpoints/`: Saved model checkpoints
- `data/`: Raw and processed datasets
- `outputs/`: Generated plots and visualizations

## Citation

If you use this code in your research, please cite our project appropriately.
