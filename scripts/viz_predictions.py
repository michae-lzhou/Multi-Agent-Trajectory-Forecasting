import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from torch.utils.data import DataLoader

from matf.data.dataset import MATFDataset, collate_fn
from matf.utils.config import load_config
from matf.utils.metrics import min_ade, min_fde


# ── colour palette ────────────────────────────────────────────────────────────
C_HIST   = "#4A90D9"   # focal history      — blue
C_GT     = "#2ECC71"   # ground truth future — green
C_PRED   = "#E74C3C"   # prediction          — red
C_NBRS   = "#95A5A6"   # neighbors           — grey
C_START  = "#F39C12"   # start marker        — orange


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


def predict(model, batch, cfg, device):
    focal_obs      = batch["focal_obs"].to(device)
    focal_type     = batch["focal_type"].to(device)
    neighbor_obs   = batch["neighbor_obs"].to(device)
    neighbor_mask  = batch["neighbor_mask"].to(device)
    neighbor_types = batch["neighbor_types"].to(device)

    with torch.no_grad():
        if cfg.model.type == "transformer":
            pred = model(
                focal_obs=focal_obs,
                focal_type=focal_type,
                neighbor_obs=neighbor_obs,
                neighbor_mask=neighbor_mask,
                neighbor_types=neighbor_types,
                mode="last_pred",
            )
        else:
            pred = model(focal_obs=focal_obs, mode="last_pred")

    # pred: (B, 1, 60, 2) → (B, 60, 2)
    return pred.squeeze(1).cpu().numpy()


def draw_arrow(ax, xy, dxy, color, alpha=0.8):
    """Draw a small direction arrow."""
    ax.annotate(
        "", xy=(xy[0] + dxy[0], xy[1] + dxy[1]), xytext=xy,
        arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
        alpha=alpha,
    )


def plot_scenario(ax, i, focal_obs, focal_future, pred,
                  neighbor_obs, neighbor_mask, ade, fde):
    """Plot one scenario on a given axes."""
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2, linewidth=0.5)
    ax.set_facecolor("#F8F9FA")

    # ── neighbors ─────────────────────────────────────────────────────────────
    n_real = neighbor_mask[i].sum().item()
    for n in range(int(n_real)):
        traj = neighbor_obs[i, n]                  # (50, 4)
        ax.plot(traj[:, 0], traj[:, 1],
                color=C_NBRS, linewidth=0.8, alpha=0.5, zorder=1)
        # small dot at last observed position
        ax.scatter(traj[-1, 0], traj[-1, 1],
                   color=C_NBRS, s=10, alpha=0.6, zorder=2)

    # ── focal history ──────────────────────────────────────────────────────────
    hist = focal_obs[i]                            # (50, 4)
    ax.plot(hist[:, 0], hist[:, 1],
            color=C_HIST, linewidth=2.0, zorder=3, label="History")
    # start of history
    ax.scatter(hist[0, 0], hist[0, 1],
               color=C_HIST, s=30, zorder=4, marker="o")
    # last observed position — focal is at origin
    ax.scatter(0, 0, color=C_START, s=60, zorder=5,
               marker="*", label="t=49")

    # ── ground truth ───────────────────────────────────────────────────────────
    gt = focal_future[i]                           # (60, 2)
    ax.plot(gt[:, 0], gt[:, 1],
            color=C_GT, linewidth=2.0, linestyle="--",
            zorder=6, label="Ground truth")
    ax.scatter(gt[-1, 0], gt[-1, 1],
               color=C_GT, s=40, zorder=7, marker="^")

    # ── prediction ─────────────────────────────────────────────────────────────
    pr = pred[i]                                   # (60, 2)
    ax.plot(pr[:, 0], pr[:, 1],
            color=C_PRED, linewidth=2.0, zorder=8, label="Prediction")
    ax.scatter(pr[-1, 0], pr[-1, 1],
               color=C_PRED, s=40, zorder=9, marker="^")

    # ── error line between endpoints ───────────────────────────────────────────
    ax.plot([gt[-1, 0], pr[-1, 0]], [gt[-1, 1], pr[-1, 1]],
            color="black", linewidth=0.8, linestyle=":", alpha=0.6, zorder=10)

    # ── metrics in title ───────────────────────────────────────────────────────
    ax.set_title(f"Scenario {i+1}  |  ADE {ade:.2f}m  FDE {fde:.2f}m",
                 fontsize=9, pad=4)
    ax.tick_params(labelsize=7)


def main(args):
    cfg    = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model(cfg, args.checkpoint, device)

    val_dataset = MATFDataset(cfg.data.val_dir,
                              neighbor_cap=cfg.data.neighbor_cap,
                              data_prefix=cfg.data.data_prefix)
    loader = DataLoader(val_dataset, batch_size=args.n_scenes,
                        shuffle=args.shuffle, collate_fn=collate_fn)

    batch = next(iter(loader))
    focal_obs    = batch["focal_obs"].numpy()       # (B, 50, 4)
    focal_future = batch["focal_future"].numpy()    # (B, 60, 2)
    neighbor_obs  = batch["neighbor_obs"].numpy()   # (B, N, 50, 4)
    neighbor_mask = batch["neighbor_mask"].numpy()  # (B, N)

    pred = predict(model, batch, cfg, device)       # (B, 60, 2)

    # per-scenario ADE and FDE
    pred_t  = torch.tensor(pred).unsqueeze(1)       # (B, 1, 60, 2)
    gt_t    = torch.tensor(focal_future)            # (B, 60, 2)
    ades    = min_ade(pred_t, gt_t).numpy()
    fdes    = min_fde(pred_t, gt_t).numpy()

    # ── layout ─────────────────────────────────────────────────────────────────
    n_cols = 3
    n_rows = (args.n_scenes + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 4.5 * n_rows))
    axes = np.array(axes).reshape(-1)

    for i in range(args.n_scenes):
        plot_scenario(axes[i], i,
                      focal_obs, focal_future, pred,
                      neighbor_obs, neighbor_mask,
                      ades[i], fdes[i])

    # hide unused subplots
    for j in range(args.n_scenes, len(axes)):
        axes[j].set_visible(False)

    # ── shared legend ──────────────────────────────────────────────────────────
    legend_elements = [
        mpatches.Patch(color=C_HIST,  label="Focal history"),
        mpatches.Patch(color=C_GT,    label="Ground truth"),
        mpatches.Patch(color=C_PRED,  label="Prediction"),
        mpatches.Patch(color=C_NBRS,  label="Neighbors"),
        mpatches.Patch(color=C_START, label="Last observed (t=49)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=5, fontsize=9, frameon=True,
               bbox_to_anchor=(0.5, 0.01))

    mean_ade = ades.mean()
    mean_fde = fdes.mean()
    fig.suptitle(
        f"{cfg.model.type.upper()}  |  "
        f"hidden={cfg.model.hidden_size}  |  "
        f"Mean ADE={mean_ade:.2f}m  FDE={mean_fde:.2f}m",
        fontsize=12, fontweight="bold", y=1.01,
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n_scenes",   type=int, default=6)
    parser.add_argument("--output",     default="viz/predictions.png")
    parser.add_argument("--shuffle",    action="store_true",
                        help="shuffle val set to get random scenarios")
    args = parser.parse_args()
    main(args)
