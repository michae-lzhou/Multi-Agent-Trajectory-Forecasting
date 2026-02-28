import torch

def _all_distances(predictions, ground_truth):
    # Compute the L2 distance across timesteps
    ground_truth = ground_truth.unsqueeze(1)
    diff = predictions - ground_truth
    dist_per_timestep = torch.norm(diff, dim=-1)    # norm over T (B, K, T)
    return dist_per_timestep

def min_ade(predictions, ground_truth):
    # Average displacement error
    # predictions:  (B, K, T, 2)
    # ground_truth: (B, T, 2)
    # B: batch, K: prediction modes, T: timesteps
    dist_per_timestep = _all_distances(predictions, ground_truth)
    ade_per_pred = dist_per_timestep.mean(dim=-1)   # mean over T (B, K)
    minADE = ade_per_pred.min(dim=-1).values        # min  over K (B, )
    minADE_scalar = minADE.mean()
    # return minADE_scalar
    return minADE

def min_fde(predictions, ground_truth):
    # Final displacement error
    # predictions:  (B, K, T, 2)
    # ground_truth: (B, T, 2)
    # B: batch, K: prediction modes, T: timesteps
    dist_per_timestep = _all_distances(predictions, ground_truth)
    fde_per_pred = dist_per_timestep[:, :, -1]       # no T, (B, K)
    minFDE = fde_per_pred.min(dim=-1).values        # min  over K (B, )
    minFDE_scalar = minFDE.mean()
    # return minFDE_scalar
    return minFDE

def miss_rate(predictions, ground_truth, threshold=2.0):
    # Miss rate threshold of 2.0 to be consistent with the av2 research paper
    dist_per_timestep = _all_distances(predictions, ground_truth)
    fde_per_pred = dist_per_timestep[:, :, -1]       # no T, (B, K)
    miss = (fde_per_pred > threshold)
    mr_per_batch = miss.all(dim=-1).float()         # (B, )
    mr_scalar = mr_per_batch.mean()
    # return mr_scalar
    return mr_per_batch
