import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset

class MATFDataset(Dataset):
    def __init__(self, data_dir, neighbor_cap=None):
        self.data_dir = Path(data_dir)

        if neighbor_cap is None:
            pattern = "baseline_data_ncap_none.npz"
        else:
            pattern = f"baseline_data_ncap_{neighbor_cap}.npz"

        self.samples = sorted(self.data_dir.glob(f"*/{pattern}"))

        if len(self.samples) == 0:
            raise ValueError(f"No data found in {data_dir} with pattern {pattern}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        data = np.load(sample)

        focal_obs = torch.from_numpy(data["focal_obs"]).float()
        focal_future = torch.from_numpy(data["focal_future"]).float()
        neighbor_obs = torch.from_numpy(data["neighbor_obs"]).float()

        return {
            "focal_obs": focal_obs,
            "focal_future": focal_future,
            "neighbor_obs": neighbor_obs,
        }


def collate_fn(batch):
    # find max neighbors in batch
    max_neighbors = max(s["neighbor_obs"].shape[0] for s in batch)
    
    # stack focal tensors (already fixed size)
    focal_obs_batch = torch.stack([s["focal_obs"] for s in batch])
    focal_future_batch = torch.stack([s["focal_future"] for s in batch])

    # pad neighbor_obs to max_neighbors
    neighbor_obs_batch = []
    neighbor_mask_batch = []
    for s in batch:
        n_obs = s["neighbor_obs"]
        N = n_obs.shape[0]
        if N < max_neighbors:
            pad = torch.zeros((max_neighbors-N,50,4), dtype=n_obs.dtype)
            n_obs = torch.cat([n_obs, pad], dim=0)
        neighbor_obs_batch.append(n_obs)
        mask = torch.zeros(max_neighbors, dtype=torch.bool)
        mask[:N] = True
        neighbor_mask_batch.append(mask)
    
    neighbor_obs_batch = torch.stack(neighbor_obs_batch)
    neighbor_mask_batch = torch.stack(neighbor_mask_batch)

    return {
        "focal_obs": focal_obs_batch,
        "focal_future": focal_future_batch,
        "neighbor_obs": neighbor_obs_batch,
        "neighbor_mask": neighbor_mask_batch
    }
