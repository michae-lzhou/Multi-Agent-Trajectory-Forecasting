"""
viz_predictions_av2.py

Renders per-scenario MP4 videos in the style of Argoverse's visualize_scenario:
  - Static map lanes drawn from the SDK's ArgoverseStaticMap
  - All agents drawn as oriented rectangles, animated timestep-by-timestep
  - Focal agent's model prediction overlaid as a trajectory

The raw .parquet scenario files + map JSONs are required (same layout as the
Argoverse 2 motion forecasting dataset on disk).

Usage:
    python scripts/viz_predictions_av2.py \
        --config          configs/transformer.yaml \
        --checkpoint      checkpoints/best.pt \
        --scenario_dir    data/raw/train \
        --processed_dir   data/processed/train \
        --output_dir      viz/videos_av2 \
        --n_scenes        6 \
        [--shuffle] \
        [--fps 10]

Dependencies:
    av2, matplotlib (ffmpeg writer or Pillow gif fallback)
"""

import argparse
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from matplotlib.patches import FancyArrow
from pathlib import Path
from torch.utils.data import DataLoader

from av2.datasets.motion_forecasting import scenario_serialization
from av2.map.map_api import ArgoverseStaticMap

from matf.data.dataset import MATFDataset, collate_fn
from matf.utils.config import load_config


# ── colours ──────────────────────────────────────────────────────────────────
C_FOCAL      = "#E74C3C"   # focal agent box        — red
C_FOCAL_PRED = "#F39C12"   # predicted trajectory   — orange
C_FOCAL_GT   = "#2ECC71"   # ground truth future    — green
C_VEHICLE    = "#4A90D9"   # other vehicles         — blue
C_PEDESTRIAN = "#9B59B6"   # pedestrians            — purple
C_OTHER      = "#95A5A6"   # everything else        — grey
C_LANE       = "#BDC3C7"   # lane centre-lines
C_LANE_BOUND = "#7F8C8D"   # lane boundaries

# Agent box dimensions (metres) — rough defaults
AGENT_LENGTH = {"vehicle": 4.5, "pedestrian": 0.5, "motorcyclist": 2.0,
                "cyclist": 1.8, "bus": 12.0, "default": 2.0}
AGENT_WIDTH  = {"vehicle": 2.0, "pedestrian": 0.5, "motorcyclist": 0.8,
                "cyclist": 0.6, "bus": 2.5,  "default": 1.0}


# ── model loading ─────────────────────────────────────────────────────────────

def load_model(cfg, ckpt_path, device):
    if cfg.model.type == "transformer":
        from matf.models.transformer import TransformerForecaster
        model = TransformerForecaster(
            hidden_size=cfg.model.hidden_size,
            num_layers=cfg.model.num_layers,
            num_heads=cfg.model.num_heads,
            dropout=cfg.model.dropout,
            use_residual=cfg.model.use_residual,
            layer_norm=cfg.model.layer_norm,
        )
    elif cfg.model.type == "lstm":
        from matf.models.lstm import LSTMTrajectoryForecaster
        model = LSTMTrajectoryForecaster(
            hidden_size=cfg.model.hidden_size,
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout,
        )
    else:
        raise ValueError(f"Unknown model type: {cfg.model.type}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model


def run_inference(model, batch, cfg, device):
    """Return prediction in the normalised coordinate frame: (60, 2)."""
    with torch.no_grad():
        if cfg.model.type == "transformer":
            pred = model(
                focal_obs      = batch["focal_obs"].to(device),
                focal_type     = batch["focal_type"].to(device),
                neighbor_obs   = batch["neighbor_obs"].to(device),
                neighbor_mask  = batch["neighbor_mask"].to(device),
                neighbor_types = batch["neighbor_types"].to(device),
                mode           = cfg.model.decoder_input,
                cell_state     = cfg.model.cell_state,
            )
        else:
            pred = model(focal_obs=batch["focal_obs"].to(device), mode="last_pred")
    return pred.squeeze(1).cpu().numpy()   # (B, 60, 2)


# ── coordinate helpers ────────────────────────────────────────────────────────

def rotation_matrix(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def to_global(pts_local, origin, theta):
    """Inverse of the normalise_positions transform: local → global coords."""
    rot = rotation_matrix(theta)          # inverse of rot_mat used in preprocess
    return (pts_local @ rot.T) + origin


def get_focal_origin_and_theta(scenario, focal_id):
    """
    Recover the normalisation origin (global x,y at t=49) and heading theta
    used during preprocessing, from the raw scenario parquet data.
    """
    import pandas as pd
    track = None
    for t in scenario.tracks:
        if t.track_id == focal_id:
            track = t
            break
    if track is None:
        raise ValueError(f"Focal track {focal_id} not found in scenario")

    # observed states: first 50 timesteps
    obs_states = [s for s in track.object_states if s.observed]
    obs_states.sort(key=lambda s: s.timestep)
    state49 = obs_states[-1]   # t = 49

    origin = np.array([state49.position[0], state49.position[1]], dtype=np.float64)
    vx, vy = state49.velocity[0], state49.velocity[1]
    speed  = np.hypot(vx, vy)
    if speed > 0.5:
        theta = np.arctan2(vy, vx)
    else:
        theta = state49.heading
    return origin, theta


# ── map drawing ───────────────────────────────────────────────────────────────

def draw_map(ax, static_map):
    """Draw lane centre-lines and boundaries onto ax."""
    for lane_seg in static_map.vector_lane_segments.values():
        try:
            centre = static_map.get_lane_segment_centerline(lane_seg.id)
            ax.plot(centre[:, 0], centre[:, 1],
                    color=C_LANE, linewidth=0.6, alpha=0.5, zorder=1)
        except Exception:
            pass
        try:
            left  = lane_seg.left_lane_boundary.xyz
            right = lane_seg.right_lane_boundary.xyz
            ax.plot(left[:, 0],  left[:, 1],
                    color=C_LANE_BOUND, linewidth=0.4, alpha=0.4, zorder=1)
            ax.plot(right[:, 0], right[:, 1],
                    color=C_LANE_BOUND, linewidth=0.4, alpha=0.4, zorder=1)
        except Exception:
            pass


# ── agent box helper ──────────────────────────────────────────────────────────

def agent_box(ax, x, y, heading, obj_type, color, alpha=0.85, zorder=5):
    """Draw an oriented rectangle representing one agent."""
    otype  = obj_type.lower() if obj_type else "default"
    length = AGENT_LENGTH.get(otype, AGENT_LENGTH["default"])
    width  = AGENT_WIDTH.get(otype,  AGENT_WIDTH["default"])

    # Corners in local frame
    corners = np.array([
        [ length/2,  width/2],
        [ length/2, -width/2],
        [-length/2, -width/2],
        [-length/2,  width/2],
    ])
    rot     = rotation_matrix(heading)
    corners = corners @ rot.T + np.array([x, y])
    patch   = mpatches.Polygon(corners, closed=True,
                               facecolor=color, edgecolor="white",
                               linewidth=0.5, alpha=alpha, zorder=zorder)
    ax.add_patch(patch)
    return patch


# ── main video renderer ───────────────────────────────────────────────────────

def render_scenario_video(
    scenario,
    static_map,
    pred_global: np.ndarray,    # (60, 2) in global coords
    out_path: Path,
    fps: int = 10,
):
    """
    Animate the scenario timestep-by-timestep with the model prediction overlaid.

    Observation phase (t=0..49): agents move, no future shown yet.
    Future phase    (t=50..109): agents continue, GT future shown,
                                  prediction trajectory unfolds.
    """
    T_OBS, T_FUTURE = 50, 60
    T_TOTAL = T_OBS + T_FUTURE

    focal_id = scenario.focal_track_id

    # ── collect per-track state arrays ────────────────────────────────────────
    # tracks_data[track_id] = list of (timestep, x, y, heading, obj_type, observed)
    tracks_data = {}
    for track in scenario.tracks:
        states = []
        for s in track.object_states:
            states.append((s.timestep, s.position[0], s.position[1],
                           s.heading, track.object_type.value, s.observed))
        states.sort(key=lambda r: r[0])
        tracks_data[track.track_id] = states

    # Build timestep → {track_id: (x,y,heading,obj_type)} lookup
    frame_lookup = [{} for _ in range(T_TOTAL)]
    for tid, states in tracks_data.items():
        for (ts, x, y, hdg, otype, _obs) in states:
            if 0 <= ts < T_TOTAL:
                frame_lookup[ts][tid] = (x, y, hdg, otype)

    # Scene bounding box from all observed positions
    all_xy = np.array([[x, y]
                        for states in tracks_data.values()
                        for (_, x, y, *_rest) in states])
    pad = 30.0
    xmin, xmax = all_xy[:, 0].min() - pad, all_xy[:, 0].max() + pad
    ymin, ymax = all_xy[:, 1].min() - pad, all_xy[:, 1].max() + pad

    # Focal ground-truth future positions (t=50..109)
    focal_future_global = []
    for ts in range(T_OBS, T_TOTAL):
        if ts in {r[0] for r in tracks_data.get(focal_id, [])}:
            for (t2, x, y, *_) in tracks_data[focal_id]:
                if t2 == ts:
                    focal_future_global.append([x, y])
                    break
    focal_future_global = np.array(focal_future_global) if focal_future_global \
                          else np.zeros((0, 2))

    # ── figure setup ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.set_facecolor("#1C1C1E")
    ax.axis("off")

    draw_map(ax, static_map)

    # Static prediction line (shown fully in future phase)
    pred_line, = ax.plot([], [], color=C_FOCAL_PRED, linewidth=2.0,
                         linestyle="-", zorder=8, label="Prediction")
    gt_line,   = ax.plot([], [], color=C_FOCAL_GT,  linewidth=2.0,
                         linestyle="--", zorder=7, label="Ground truth")

    phase_text = ax.text(0.02, 0.97, "", transform=ax.transAxes,
                         fontsize=9, color="white", va="top")

    legend_elements = [
        mpatches.Patch(color=C_FOCAL,      label="Focal agent"),
        mpatches.Patch(color=C_FOCAL_GT,   label="Ground truth future"),
        mpatches.Patch(color=C_FOCAL_PRED, label="Prediction"),
        mpatches.Patch(color=C_VEHICLE,    label="Vehicles"),
        mpatches.Patch(color=C_PEDESTRIAN, label="Pedestrians"),
    ]
    ax.legend(handles=legend_elements, loc="lower right",
              fontsize=7, framealpha=0.6,
              facecolor="#2C2C2E", labelcolor="white")

    # Container for per-frame agent patches (cleared each frame)
    agent_patches = []

    def update(frame):
        nonlocal agent_patches
        # Remove previous agent boxes
        for p in agent_patches:
            p.remove()
        agent_patches = []

        agents_at_t = frame_lookup[frame]

        for tid, (x, y, hdg, otype) in agents_at_t.items():
            is_focal = (tid == focal_id)
            if is_focal:
                color = C_FOCAL
                zorder = 6
            elif "pedestrian" in otype.lower():
                color = C_PEDESTRIAN
                zorder = 4
            elif "vehicle" in otype.lower() or "bus" in otype.lower():
                color = C_VEHICLE
                zorder = 4
            else:
                color = C_OTHER
                zorder = 3

            patch = agent_box(ax, x, y, hdg, otype, color,
                              alpha=0.9 if is_focal else 0.7,
                              zorder=zorder)
            agent_patches.append(patch)

        # Future phase: show unfolding gt and prediction
        if frame >= T_OBS:
            f = frame - T_OBS   # 0..59
            if len(focal_future_global) > 0:
                gt_so_far = focal_future_global[:f+1]
                gt_line.set_data(gt_so_far[:, 0], gt_so_far[:, 1])
            pred_so_far = pred_global[:f+1]
            pred_line.set_data(pred_so_far[:, 0], pred_so_far[:, 1])
            phase_text.set_text(f"Future  t={f+1}/{T_FUTURE}")
        else:
            gt_line.set_data([], [])
            pred_line.set_data([], [])
            phase_text.set_text(f"Observed  t={frame+1}/{T_OBS}")

        return agent_patches + [pred_line, gt_line, phase_text]

    ani = animation.FuncAnimation(
        fig, update, frames=T_TOTAL,
        interval=1000 // fps, blit=False,  # blit=False needed for patch removal
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        writer = animation.FFMpegWriter(fps=fps, bitrate=1800,
                                        extra_args=["-vcodec", "libx264"])
        ani.save(str(out_path), writer=writer)
        print(f"  [MP4] → {out_path}")
    except Exception as e:
        gif_path = out_path.with_suffix(".gif")
        print(f"  ffmpeg unavailable ({e}), saving GIF → {gif_path}")
        ani.save(str(gif_path), writer="pillow", fps=fps)

    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def main(args):
    cfg    = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model(cfg, args.checkpoint, device)

    # Collect all scenario parquet files from the raw directory
    scenario_dir = Path(args.scenario_dir)
    all_parquets = sorted(scenario_dir.rglob("scenario_*.parquet"))
    if not all_parquets:
        raise FileNotFoundError(f"No scenario_*.parquet files found in {scenario_dir}")

    if args.shuffle:
        random.shuffle(all_parquets)
    selected = all_parquets[:args.n_scenes]

    # We also need the processed .npz files to run the model — match by scenario id
    processed_dir = Path(args.processed_dir)
    data_prefix   = getattr(cfg.data, "data_prefix", "baseline")
    neighbor_cap  = cfg.data.neighbor_cap

    out_dir = Path(args.output_dir)

    for idx, parquet_path in enumerate(selected):
        scenario_id = parquet_path.stem.replace("scenario_", "")
        map_path    = parquet_path.parent / f"log_map_archive_{scenario_id}.json"

        if not map_path.exists():
            print(f"  [SKIP] map not found for {scenario_id}")
            continue

        # Find the matching processed npz
        npz_path = processed_dir / scenario_id / \
                   f"{data_prefix}_data_ncap_{neighbor_cap}.npz"
        if not npz_path.exists():
            print(f"  [SKIP] processed npz not found: {npz_path}")
            continue

        print(f"Rendering scenario {idx+1}/{len(selected)}: {scenario_id}")

        # Load raw scenario + map
        scenario   = scenario_serialization.load_argoverse_scenario_parquet(parquet_path)
        static_map = ArgoverseStaticMap.from_json(map_path)

        # Load processed tensors and run model (batch size 1)
        data    = np.load(npz_path)
        batch   = {
            "focal_obs":      torch.tensor(data["focal_obs"]).unsqueeze(0),
            "focal_future":   torch.tensor(data["focal_future"]).unsqueeze(0),
            "focal_type":     torch.tensor(data["focal_type"]).unsqueeze(0),
            "neighbor_obs":   torch.tensor(data["neighbor_obs"]).unsqueeze(0),
            "neighbor_types": torch.tensor(data["neighbor_types"]).unsqueeze(0),
            "neighbor_mask":  torch.ones(1, data["neighbor_obs"].shape[0],
                                         dtype=torch.bool),
        }
        pred_local = run_inference(model, batch, cfg, device)[0]   # (60, 2)

        # Convert prediction from normalised frame back to global coords
        origin, theta = get_focal_origin_and_theta(scenario, scenario.focal_track_id)
        pred_global   = to_global(pred_local, origin, theta)       # (60, 2)

        render_scenario_video(
            scenario    = scenario,
            static_map  = static_map,
            pred_global = pred_global,
            out_path    = out_dir / f"scenario_{idx+1:03d}_{scenario_id}.mp4",
            fps         = args.fps,
        )

    print(f"\nDone. Videos written to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Argoverse-style animated prediction visualizer."
    )
    parser.add_argument("--config",        required=True,
                        help="YAML config used for training.")
    parser.add_argument("--checkpoint",    required=True,
                        help="Model checkpoint (.pt).")
    parser.add_argument("--scenario_dir",  required=True,
                        help="Root dir of raw Argoverse 2 scenarios "
                             "(contains scenario_*.parquet + map JSON).")
    parser.add_argument("--processed_dir", required=True,
                        help="Root dir of preprocessed .npz files "
                             "(output of preprocess.py).")
    parser.add_argument("--output_dir",    default="viz/videos_av2",
                        help="Where to write MP4 files.")
    parser.add_argument("--n_scenes",      type=int, default=6)
    parser.add_argument("--fps",           type=int, default=10)
    parser.add_argument("--shuffle",       action="store_true",
                        help="Pick random scenarios instead of the first N.")
    args = parser.parse_args()
    main(args)
