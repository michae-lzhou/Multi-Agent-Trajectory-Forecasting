import numpy as np
import pandas as pd

OBJECT_TYPE_MAP = {
    "vehicle":            0,
    "pedestrian":         1,
    "motorcyclist":       2,
    "cyclist":            3,
    "bus":                4,
    "static":             5,
    "background":         6,
    "construction":       7,
    "riderless_bicycle":  8,
    "unknown":            9,
}

def load_scenario(file, neighbor_cap=None):
    df = pd.read_parquet(file)

    # Extract Focal Agent
    focal_id = df['focal_track_id'].iloc[0]
    focal = df[df['track_id'] == focal_id]
    focal = focal.sort_values(by=['timestep'])

    # Extract Neighbors in Category 1 and 2
    neighbors = df[df['object_category'].isin([1, 2])]
    neighbors_by_track = {
        track_id: neighbor.sort_values('timestep')
        for track_id, neighbor in neighbors.groupby('track_id')
    }

    # keys = neighbors_by_track.keys()
    # print(f"len neighbors: {len(neighbors) / len(list(keys))}")
    # print(keys)
    # print(neighbors_by_track[list(keys)[4]])

    # Build model input tensors
    focal_obs, focal_future, focal_type = build_focal_tensors(focal)
    # print(focal_obs[:5])
    # print(focal_future[:5])
    n_obs, n_types = build_neighbor_tensors(neighbors_by_track, neighbor_cap)
    # print(n_obs.shape)
    focal_obs, focal_future, n_obs = normalize_positions(
        focal_obs, focal_future, n_obs
    )
    # print(focal_obs)
    # print(focal_future)
    # print(n_obs)
    return {
        'focal_obs': focal_obs,
        'focal_future': focal_future,
        'focal_type': focal_type,
        'neighbor_obs': n_obs,
        'neighbor_types': n_types
    }

def build_focal_tensors(focal):
    obs_df = focal[focal['observed']]
    future_df = focal[~focal['observed']]

    # print(len(obs_df))
    # print(len(future_df))

    focal_obs = obs_df[['position_x',
                        'position_y',
                        'velocity_x',
                        'velocity_y',
                        'heading']].to_numpy(dtype=np.float32)
    focal_future = future_df[['position_x',
                              'position_y',]].to_numpy(dtype=np.float32)

    focal_type = np.array([OBJECT_TYPE_MAP[obs_df['object_type'].iloc[0]]],
                           dtype=np.int64)
    
    if focal_obs.shape != (50, 5):
        raise ValueError(f"focal_obs has shape {focal_obs.shape}, expected (50,4)")
    if focal_future.shape != (60, 2):
        raise ValueError(f"focal_future has shape {focal_future.shape}, expected (60,2)")

    return focal_obs, focal_future, focal_type

# TODO: if we move onto transformers, natural ablation study: distance-based
# neighbor selection more impactful than random truncation (what we have right
# now)?
def build_neighbor_tensors(neighbors_dict, neighbor_cap=None):
    n_list = list(neighbors_dict)
    N = len(n_list) if neighbor_cap is None else min(len(n_list), neighbor_cap)

    # We only need to observe the neighbors' history
    n_obs = np.zeros((N, 50, 5), dtype=np.float32)
    n_types = np.zeros((N, ), dtype=np.int64)
    for n in range(N):
        curr_neighbor = neighbors_dict[n_list[n]]
        curr_neighbor = curr_neighbor[curr_neighbor['observed']]
        n_obs[n] = curr_neighbor[['position_x',
                                          'position_y',
                                          'velocity_x',
                                          'velocity_y',
                                          'heading']].to_numpy(dtype=np.float32)
        n_types[n] = OBJECT_TYPE_MAP[curr_neighbor['object_type'].iloc[0]]

        # print(cleaned_neighbor.shape)
        # print(cleaned_neighbor)

    return n_obs, n_types

def normalize_positions(focal_obs, focal_future, neighbor_obs):
    t = 49  # present timestep

    focal_obs = focal_obs.copy()
    focal_future = focal_future.copy()
    neighbor_obs = neighbor_obs.copy()

    # 1. Translation origin = focal position at present
    origin_x = focal_obs[t, 0]
    origin_y = focal_obs[t, 1]
    origin = np.array([origin_x, origin_y], dtype=np.float32)

    # 2. Rotation angle: align focal heading to +x axis
    # Use displacement from t-1 to t to approximate heading
    dx = focal_obs[t, 0] - focal_obs[t - 1, 0]
    dy = focal_obs[t, 1] - focal_obs[t - 1, 1]

    # Fallback if nearly stationary
    if np.hypot(dx, dy) < 1e-8:
        dx = focal_obs[t, 2]
        dy = focal_obs[t, 3]

    theta = np.arctan2(dy, dx)

    # this matrix rotates points by -theta (points are row matrixes)
    c = np.cos(theta)
    s = np.sin(theta)
    rot_mat = np.array([
        [ c,  s],
        [-s,  c]
    ], dtype=np.float32)

    def apply_transform(pts, translate=True):
        """
        pts shape: (..., 2)
        """
        res = pts.copy()
        if translate:
            # Move the focal's agent current position to origin (0,0)
            res = res - origin
        # Rotate into the focal agent's heading-aligned frame
        return res @ rot_mat

    # Transform focal agent
    # For positions: do translation + rotation
    focal_obs[:, :2] = apply_transform(focal_obs[:, :2], translate=True)

    # For velocities: only rotate
    focal_obs[:, 2:4] = apply_transform(focal_obs[:, 2:4], translate=False)

    # Future positions are also positions, so translate + rotate
    focal_future[:, :2] = apply_transform(focal_future[:, :2], translate=True)

    # Transform neighbors
    # Valid rows are rows that are not all zeros
    neighbor_mask = np.any(neighbor_obs != 0, axis=-1)   # (N, 50)

    # Transform neighbor positions
    neighbor_obs[:, :, :2] = apply_transform(neighbor_obs[:, :, :2], translate=True)

    # Transform neighbor velocities
    neighbor_obs[:, :, 2:4] = apply_transform(neighbor_obs[:, :, 2:4], translate=False)

    # Restore padded rows to zero so the fake neighbors stay fake
    neighbor_obs[~neighbor_mask] = 0

    return focal_obs, focal_future, neighbor_obs

def save_scenario(data_dict, out_path):
    np.savez_compressed(
            out_path,
            focal_obs=data_dict['focal_obs'],
            focal_future=data_dict['focal_future'],
            focal_type=data_dict['focal_type'],
            neighbor_obs=data_dict['neighbor_obs'],
            neighbor_types=data_dict['neighbor_types'],
    )

# load_scenario("data/processed/train/00d3a91c-9c8e-4204-ba58-56bbb75e6503/scenario_00d3a91c-9c8e-4204-ba58-56bbb75e6503.parquet")
# load_scenario("data/processed/train/0003e31e-8142-47af-9215-b8a306a31bc9/scenario_0003e31e-8142-47af-9215-b8a306a31bc9.parquet")