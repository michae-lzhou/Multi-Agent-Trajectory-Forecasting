import argparse
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from matf.data.dataset import MATFDataset, collate_fn
from matf.models.cv import constant_velocity
from matf.models.lstm import LSTMTrajectoryForecaster
from matf.utils.metrics import min_ade, min_fde, miss_rate
from matf.utils.config import load_config, save_config, make_run_name, \
                              print_config

def evaluate(model, data_dir, neighbor_cap=None, batch_size=32):
    dataset = MATFDataset(data_dir, neighbor_cap=neighbor_cap)
    loader  = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

    all_ade = []
    all_fde = []
    all_mr  = []

    with torch.no_grad():
        for batch in loader:
            predictions = model(batch)
            # focal_obs    = batch["focal_obs"]
            # predictions  = constant_velocity(focal_obs)         # (B, 1, 60, 2)
            ground_truth = batch["focal_future"]                # (B, 60, 2)

            minADE = min_ade(predictions, ground_truth)         # (B, )
            minFDE = min_fde(predictions, ground_truth)         # (B, )
            mr = miss_rate(predictions, ground_truth)    # (B, )
            
            all_ade.append(minADE)
            all_fde.append(minFDE)
            all_mr.append(mr)

        final_ade = torch.cat(all_ade).mean().item()
        final_fde = torch.cat(all_fde).mean().item()
        final_mr  = torch.cat(all_mr).mean().item()

        print(f"minADE:    {final_ade:.4f} m")
        print(f"minFDE:    {final_fde:.4f} m")
        print(f"MissRate:  {final_mr:.4f}")

def load_model(args):
    if args.model == "cv":
        return cv_model

    elif args.model == "lstm":
        if args.config is None:
            raise ValueError("--config required for lstm")
        if args.checkpoint is None:
            raise ValueError("--checkpoint required for lstm")
        if not Path(args.checkpoint).exists():
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
        
        cfg = load_config(args.config)
        print_config(cfg)
        lstm_instance = LSTMTrajectoryForecaster(
                            hidden_size=cfg.model.hidden_size,
                            num_layers=cfg.model.num_layers,
                            dropout=cfg.model.dropout
                        )
        checkpoint = torch.load(args.checkpoint, weights_only=True)
        lstm_instance.load_state_dict(checkpoint["model_state_dict"])
        lstm_instance.eval()
        return make_lstm_callable(lstm_instance)

    elif args.model == "transformer":
        if args.config is None:
            raise ValueError("--config required for transformer")
        if args.checkpoint is None:
            raise ValueError("--checkpoint required for transformer")

    else:
        raise ValueError(f"Unknown model: {args.model}")

def cv_model(batch):
    return constant_velocity(batch["focal_obs"])

def make_lstm_callable(lstm_instance):
    def predict(batch):
        return lstm_instance(
            batch["focal_obs"],
            mode="last_pred"
        )
    return predict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--neighbor_cap",
        type=int,
        default=None,
        help="Maximum number of neighbors to include (default: None)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed/val",
        help="Path to dataset directory"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model to evaluate (cv, lstm, or transformer)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Config file for the model (config/lstm.yaml or "
             "config/transformer.yaml"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints",
        help="Folder to checkpoints"
    )

    args = parser.parse_args()

    model = load_model(args)

    evaluate(
        model=model,
        data_dir=args.data_dir,
        neighbor_cap=args.neighbor_cap
    )
