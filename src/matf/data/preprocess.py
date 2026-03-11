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
    focal_pos_t49 = focal_obs[49, :2]
    n_obs, n_types = build_neighbor_tensors(neighbors_by_track, focal_pos_t49, neighbor_cap)
    # print(n_obs.shape)
    focal_obs, focal_future, n_obs = normalize_positions(focal_obs, focal_future, n_obs)

    # neighbor_mask_for_check = np.any(n_obs != 0, axis=-1)
    # sanity_check_normalization(focal_obs, focal_future, n_obs, neighbor_mask_for_check)
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

def build_neighbor_tensors(neighbors_dict, focal_pos_t49, neighbor_cap=None):
    n_list = list(neighbors_dict)
    N = len(n_list) if neighbor_cap is None else min(len(n_list), neighbor_cap)

    def get_dist(track_id):
        df = neighbors_dict[track_id]
        df = df[df['observed']]
        last = df.iloc[-1]  # t=49
        dx = last['position_x'] - focal_pos_t49[0]
        dy = last['position_y'] - focal_pos_t49[1]
        return np.hypot(dx, dy)

    n_list = sorted(n_list, key=get_dist)

    # We only need to observe the neighbors' history
    n_obs = np.zeros((N, 50, 5), dtype=np.float32)
    n_types = np.zeros((N, ), dtype=np.int64)
    for n in range(N):
        curr_neighbor = neighbors_dict[n_list[n]]
        curr_neighbor = curr_neighbor[curr_neighbor['observed']]
        n_obs[n] = curr_neighbor[['position_x', 'position_y',
                                  'velocity_x', 'velocity_y',
                                  'heading']].to_numpy(dtype=np.float32)
        n_types[n] = OBJECT_TYPE_MAP[curr_neighbor['object_type'].iloc[0]]

        # print(cleaned_neighbor.shape)
        # print(cleaned_neighbor)

    return n_obs, n_types

def normalize_positions(focal_obs, focal_future, neighbor_obs):
    t = 49  # present timestep

    origin = np.array([focal_obs[t, 0], focal_obs[t, 1]], dtype=np.float32)

    focal_obs = focal_obs.copy()
    focal_future = focal_future.copy()
    neighbor_obs = neighbor_obs.copy()

    vx, vy = focal_obs[t, 2], focal_obs[t, 3]
    speed = np.hypot(vx, vy)
    if speed > 0.5:
        theta = np.arctan2(vy, vx)   # direction of travel
    else:
        theta = focal_obs[t, 4]      # fallback to heading if nearly stopped

    c = np.cos(theta)
    s = np.sin(theta)
    rot_mat = np.array([
        [ c,  s],
        [-s,  c]
    ], dtype=np.float32)

    def apply_transform(pts, translate=True):
        res = pts.copy()
        if translate:
            res = res - origin
        return res @ rot_mat

    # Transform focal agent
    focal_obs[:, :2] = apply_transform(focal_obs[:, :2], translate=True)
    focal_obs[:, 2:4] = apply_transform(focal_obs[:, 2:4], translate=False)
    focal_future[:, :2] = apply_transform(focal_future[:, :2], translate=True)

    # Transform neighbors
    neighbor_mask = np.any(neighbor_obs != 0, axis=-1)
    neighbor_obs[:, :, :2] = apply_transform(neighbor_obs[:, :, :2], translate=True)
    neighbor_obs[:, :, 2:4] = apply_transform(neighbor_obs[:, :, 2:4], translate=False)

    # Heading transform
    focal_obs[:, 4] = focal_obs[:, 4] - theta
    focal_obs[:, 4] = (focal_obs[:, 4] + np.pi) % (2 * np.pi) - np.pi
    neighbor_obs[:, :, 4] = neighbor_obs[:, :, 4] - theta
    neighbor_obs[:, :, 4] = (neighbor_obs[:, :, 4] + np.pi) % (2 * np.pi) - np.pi

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

def sanity_check_normalization(focal_obs, focal_future, neighbor_obs, neighbor_mask):
    t = 49
    tol = 1e-3

    # --- focal position at t=49 is at origin ---
    assert abs(focal_obs[t, 0]) < tol, \
        f"focal x at t=49 should be 0, got {focal_obs[t, 0]:.6f}"
    assert abs(focal_obs[t, 1]) < tol, \
        f"focal y at t=49 should be 0, got {focal_obs[t, 1]:.6f}"

    # --- focal heading at t=49 is 0 (facing +x) ---
    # assert abs(focal_obs[t, 4]) < tol, \
    #     f"focal heading at t=49 should be 0, got {focal_obs[t, 4]:.6f}"

    # --- focal velocity at t=49 points in +x direction ---
    # i.e. vy should be ~0 and vx should be >= 0 (moving forward)
    vx, vy = focal_obs[t, 2], focal_obs[t, 3]
    speed = np.hypot(vx, vy)
    if speed > 0.5:
        lateral_fraction = abs(vy) / speed
        if lateral_fraction > 0.5:   # more than 30° off — just warn, don't fail
            print(f"  NOTE: high lateral velocity fraction {lateral_fraction:.2f} "
                  f"(vx={vx:.2f}, vy={vy:.2f}) — agent turning at t=49")

    # --- all headings are in [-pi, pi] ---
    assert np.all(focal_obs[:, 4] >= -np.pi) and np.all(focal_obs[:, 4] <= np.pi), \
        "focal headings out of [-pi, pi] range"

    real_neighbor_headings = neighbor_obs[neighbor_mask, 4]
    if len(real_neighbor_headings) > 0:
        assert np.all(real_neighbor_headings >= -np.pi) and \
               np.all(real_neighbor_headings <= np.pi), \
            "neighbor headings out of [-pi, pi] range"

    # --- padded neighbors are exactly zero ---
    padded = neighbor_obs[~neighbor_mask]
    assert np.all(padded == 0), \
        f"padded neighbor rows should be all zeros, found {(padded != 0).sum()} nonzero values"

    # --- future starts near origin (focal moves forward from t=49) ---
    # first future step should be close to (0,0) since dt=0.1s
    assert np.hypot(focal_future[0, 0], focal_future[0, 1]) < 5.0, \
        f"first future position suspiciously far from origin: {focal_future[0]}"

    # --- future is in +x direction on average (agent moving forward) ---
    mean_future_x = focal_future[:, 0].mean()
    # not a hard assert since some agents brake/reverse, but flag if negative
    if mean_future_x < -1.0:
        print(f"  WARNING: mean future x is {mean_future_x:.2f} — agent moving backward?")

    print("  All sanity checks passed.")

# load_scenario("data/processed/train/01777777777777777777777d3a91c-9c8e-4204-ba58-56bbb75e6503/scenario_00d3a91c-9c8e-4204-ba58-56bbb75e6503.parquet")
# load_scenario("data/processed/train/0003e31e-8142-47af-9215-b8a306a31bc9/scenario_0003e31e-8142-47af-9215-b8a306a31bc9.parquet")

# if __name__ == "__main__":
#     import sys
#     from pathlib import Path
# 
#     data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/processed/train")
#     n_test = int(sys.argv[2]) if len(sys.argv) > 2 else 200
# 
#     files = sorted(data_dir.glob("*/*.parquet"))[:n_test]
#     print(f"Testing {len(files)} scenarios from {data_dir}")
# 
#     n_passed = 0
#     n_failed = 0
#     for i, f in enumerate(files):
#         try:
#             load_scenario(f, neighbor_cap=22)
#             n_passed += 1
#         except AssertionError as e:
#             print(f"  [FAIL] {f.name}: {e}")
#             n_failed += 1
#         except Exception as e:
#             print(f"  [ERROR] {f.name}: {type(e).__name__}: {e}")
#             n_failed += 1
# 
#         if (i + 1) % 50 == 0:
#             print(f"  {i+1}/{len(files)} done — {n_failed} failures so far")
# 
#     print(f"\nDone. {n_passed} passed, {n_failed} failed out of {len(files)}")
