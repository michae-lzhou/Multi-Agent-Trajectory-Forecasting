import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset

class MATFDataset(Dataset):
    def __init__(self, data_dir, neighbor_cap=None, data_prefix="baseline"):
        self.data_dir = Path(data_dir)

        if neighbor_cap is None:
            pattern = f"{data_prefix}_data_ncap_none.npz"
        else:
            pattern = f"{data_prefix}_data_ncap_{neighbor_cap}.npz"

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
        focal_type = torch.from_numpy(data["focal_type"]).long()
        neighbor_obs = torch.from_numpy(data["neighbor_obs"]).float()
        neighbor_types = torch.from_numpy(data["neighbor_types"]).long()

        return {
            "focal_obs": focal_obs,
            "focal_future": focal_future,
            "focal_type": focal_type,
            "neighbor_obs": neighbor_obs,
            "neighbor_types": neighbor_types
        }


def collate_fn(batch):
    # find max neighbors in batch
    B = len(batch)
    max_neighbors = max(s["neighbor_obs"].shape[0] for s in batch)
    
    # stack focal tensors (already fixed size)
    focal_obs_batch = torch.stack([s["focal_obs"] for s in batch])
    focal_future_batch = torch.stack([s["focal_future"] for s in batch])
    focal_type_batch = torch.stack([s["focal_type"] for s in batch])

    # pad neighbor_obs to max_neighbors
    neighbor_obs_batch   = torch.zeros(B, max_neighbors, 50, 5, dtype=torch.float32)
    neighbor_types_batch = torch.zeros(B, max_neighbors,         dtype=torch.long)
    neighbor_mask_batch  = torch.zeros(B, max_neighbors,         dtype=torch.bool)

    for i, b in enumerate(batch):
        N = b['neighbor_obs'].shape[0]
        neighbor_obs_batch[i,   :N] = b['neighbor_obs']
        neighbor_types_batch[i, :N] = b['neighbor_types']
        neighbor_mask_batch[i,  :N] = True
    
    return {
        "focal_obs": focal_obs_batch,
        "focal_future": focal_future_batch,
        "focal_type": focal_type_batch,
        "neighbor_obs": neighbor_obs_batch,
        "neighbor_mask": neighbor_mask_batch,
        "neighbor_types": neighbor_types_batch,
    }
