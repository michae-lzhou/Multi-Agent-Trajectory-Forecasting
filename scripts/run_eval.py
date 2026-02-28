import argparse
import torch
from torch.utils.data import DataLoader
from matf.data.dataset import MATFDataset, collate_fn
from matf.models.cv import constant_velocity
from matf.utils.metrics import min_ade, min_fde, miss_rate

def evaluate(data_dir, neighbor_cap=None, batch_size=32):
    dataset = MATFDataset(data_dir, neighbor_cap=neighbor_cap)
    loader  = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

    all_ade = []
    all_fde = []
    all_mr  = []

    with torch.no_grad():
        for batch in loader:
            focal_obs    = batch["focal_obs"]
            # TODO: temporarily keep CV here as the model
            #       modify later to generalize to all models
            predictions  = constant_velocity(focal_obs)         # (B, 1, 60, 2)
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

    args = parser.parse_args()

    evaluate(
        data_dir=args.data_dir,
        neighbor_cap=args.neighbor_cap
    )
