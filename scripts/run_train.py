import torch
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
from matf.data.dataset import MATFDataset, collate_fn
from matf.models.lstm import LSTMTrajectoryForecaster
from matf.models.transformer import SocialLSTMTrajectoryForecaster
from matf.utils.logger import TrainingLogger
from matf.utils.viz import save_training_plot
from matf.utils.metrics import min_ade, min_fde, miss_rate
from matf.utils.config import load_config, save_config, make_run_name, \
                              print_config

SCALE_POS = 10.0
SCALE_VEL = 10.0

def log_gradients(model):
    """Print gradient norm for each top‑level module inside the model.

    The output is grouped by the prefix before the first ``.`` in each
    parameter name (e.g. ``encoder``, ``social_attn``, ``decoder``).  This
    makes it easy to inspect which component is receiving signal during
    backprop.
    """
    grads = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        prefix = name.split(".")[0]
        grads.setdefault(prefix, []).append(param.grad.norm().item())

    for prefix, norms in grads.items():
        mean_norm = sum(norms) / len(norms)
        print(f"  {prefix:12s} grad norm (mean over {len(norms)} params): {mean_norm:.4e}")


def train_epoch(model, loader, optimizer, criterion, cfg, device):
    model.train()
    total_loss = 0.0
    n_batches  = 0

    model_type = getattr(cfg.model, "type", "lstm")

    # whether to log gradients after every backward
    debug_grads = getattr(cfg.training, "log_gradients", False)

    for batch in loader:
        focal_obs    = batch["focal_obs"].to(device).clone()
        focal_future = batch["focal_future"].to(device).clone()
        focal_obs[..., :2] /= SCALE_POS
        focal_obs[..., 2:4] /= SCALE_VEL
        focal_future[..., :2] /= SCALE_POS

        # Forward pass; include neighbours if the model expects them
        if model_type == "social":
            neigh_obs = batch["neighbor_obs"].to(device).clone()
            neigh_obs[..., :2] /= SCALE_POS
            neigh_obs[..., 2:4] /= SCALE_VEL

            mask = batch["neighbor_mask"].to(device)
            pred = model(
                focal_obs=focal_obs,
                neighbors_obs=neigh_obs,
                neighbor_mask=mask,
                target=focal_future,
                mode=cfg.model.decoder_input,
            )
        else:
            pred = model(
                focal_obs=focal_obs,
                target=focal_future,
                mode=cfg.model.decoder_input,
            )

        pred = pred.squeeze(1)
        
        # Compute loss
        loss = criterion(pred, focal_future)
        optimizer.zero_grad()
        loss.backward()

        if debug_grads:
            print("gradient norms after backward:")
            log_gradients(model)

        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                 max_norm=cfg.training.grad_clip)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches

def val_epoch(model, loader, device, cfg):
    model.eval()
    all_ade, all_fde, all_mr = [], [], []

    model_type = getattr(cfg.model, "type", "lstm")

    with torch.no_grad():
        for batch in loader:
            focal_obs    = batch["focal_obs"].to(device).clone()
            focal_future = batch["focal_future"].to(device).clone()
            focal_obs[..., :2] /= SCALE_POS
            focal_obs[..., 2:4] /= SCALE_VEL
            focal_future[..., :2] /= SCALE_POS

            if model_type == "social":
                neigh_obs = batch["neighbor_obs"].to(device).clone()
                neigh_obs[..., :2] /= SCALE_POS
                neigh_obs[..., 2:4] /= SCALE_VEL

                mask = batch["neighbor_mask"].to(device)
                pred = model(
                    focal_obs=focal_obs,
                    neighbors_obs=neigh_obs,
                    neighbor_mask=mask,
                    mode="last_pred",
                )
            else:
                pred = model(focal_obs=focal_obs, mode="last_pred")

            all_ade.append(min_ade(pred, focal_future))
            all_fde.append(min_fde(pred, focal_future))
            all_mr.append(miss_rate(pred, focal_future))

    return (
        torch.cat(all_ade).mean().item(),
        torch.cat(all_fde).mean().item(),
        torch.cat(all_mr).mean().item(),
    )

def train(config_path):
    cfg      = load_config(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    run_name = make_run_name(cfg)
    run_dir  = Path(cfg.training.checkpoint_dir) / run_name

    save_config(config_path, run_dir)
    logger = TrainingLogger(run_dir)
    print_config(cfg)

    train_dataset = MATFDataset(cfg.data.train_dir,
                                neighbor_cap=cfg.data.neighbor_cap)
    val_dataset = MATFDataset(cfg.data.val_dir,
                              neighbor_cap=cfg.data.neighbor_cap)

    train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size,
                              shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=cfg.training.batch_size,
                            shuffle=False, collate_fn=collate_fn)

    # choose model based on configuration
    model_type = getattr(cfg.model, "type", "lstm")
    if model_type == "social":
        model = SocialLSTMTrajectoryForecaster(
            hidden_size=cfg.model.hidden_size,
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout,
            num_heads=cfg.model.num_heads,
            use_residual=cfg.model.use_residual,
            use_layer_norm=cfg.model.use_layer_norm,
            cell_state=cfg.model.cell_state,
        ).to(device)
    else:
        model = LSTMTrajectoryForecaster(
            hidden_size=cfg.model.hidden_size,
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout
        ).to(device)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=float(cfg.training.learning_rate))
    criterion = torch.nn.MSELoss()

    # TODO: consider if val ade is the best criteria to judge performance
    best_val_ade = float("inf")
    train_losses = []
    val_ades, val_fdes, val_mrs = [], [], []

    for epoch in range(1, cfg.training.num_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion,
                                 cfg, device)
        val_ade, val_fde, val_mr = val_epoch(model, val_loader, device, cfg)
        
        train_losses.append(train_loss)
        val_ades.append(val_ade)
        val_fdes.append(val_fde)
        val_mrs.append(val_mr)

        is_best = val_ade < best_val_ade
        if is_best:
            best_val_ade = val_ade
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_ade": best_val_ade,
                "train_losses": train_losses,
                "val_ades": val_ades,
                "val_fdes": val_fdes,
                "val_mrs": val_mrs,
            }, run_dir / "best_model.pt")

        logger.log(epoch, train_loss, val_ade, val_fde, val_mr, is_best=is_best)
        save_training_plot(train_losses, val_ades, val_fdes, val_mrs, run_dir)

    logger.close()
    print(f"\nDone. Best val ADE: {best_val_ade:.4f} m")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lstm.yaml")
    args = parser.parse_args()
    train(args.config)
