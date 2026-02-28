import torch
import numpy as np

def constant_velocity(focal_obs, dt=0.1):
    # Simple extrapolation from the current positions forward for 60 frames
    # using the current velocity
    vel = focal_obs[:, -1, 2:]
    # print(f"vel.shape: {vel.shape}, expecting (B, 2)")
    t = torch.arange(1, 61, dtype=torch.float32, device=focal_obs.device) * dt
    # print(f"t.shape: {t.shape}, expecting (60, )")
    vel = vel.unsqueeze(1)      # (B, 1, 2)
    t = t.view(1, 60, 1)        # (1, 60, 1)
    traj = vel * t              # (B, 60, 2)
    traj = traj.unsqueeze(1)    # (B, 1, 60, 2)

    return traj
