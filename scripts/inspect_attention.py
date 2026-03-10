import torch
from torch.utils.data import DataLoader
from matf.data.dataset import MATFDataset, collate_fn
from matf.utils.config import load_config
from matf.models.transformer import TransformerForecaster

# --- config ---
CONFIG   = "configs/transformer.yaml"
CKPT     = "checkpoints/ncap=22/transformer_h256_heads2_resTrue_normno_norm_cell_statefocal/best_model.pt"  # fill in your run name
N_SCENES = 5   # how many scenarios to inspect

cfg    = load_config(CONFIG)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- load model ---
model = TransformerForecaster(
    hidden_size=cfg.model.hidden_size,
    num_layers=cfg.model.num_layers,
    num_heads=cfg.model.num_heads,
    dropout=cfg.model.dropout,
    use_residual=cfg.model.use_residual,
    layer_norm=cfg.model.layer_norm,
).to(device)
ckpt = torch.load(CKPT, map_location=device, weights_only=True)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# --- load one batch ---
val_dataset = MATFDataset(cfg.data.val_dir, neighbor_cap=cfg.data.neighbor_cap,
                          data_prefix=cfg.data.data_prefix)
val_loader  = DataLoader(val_dataset, batch_size=N_SCENES,
                         shuffle=False, collate_fn=collate_fn)
batch = next(iter(val_loader))

focal_obs      = batch["focal_obs"].to(device)
focal_type     = batch["focal_type"].to(device)
neighbor_obs   = batch["neighbor_obs"].to(device)
neighbor_mask  = batch["neighbor_mask"].to(device)
neighbor_types = batch["neighbor_types"].to(device)

# --- forward pass ---
# you need TransformerForecaster.forward to return attn_weights too
# temporarily modify forward to return them, or expose via a hook
with torch.no_grad():
    pred, attn_weights = model(
        focal_obs=focal_obs,
        focal_type=focal_type,
        neighbor_obs=neighbor_obs,
        neighbor_mask=neighbor_mask,
        neighbor_types=neighbor_types,
        mode="last_pred",
        return_attn=True   # add this flag to forward
    )

# attn_weights: (B, 1, N+1)
# index 0 = focal token, index 1..N = neighbors
print("=" * 60)
for i in range(N_SCENES):
    weights = attn_weights[i, 0]          # (N+1,)
    mask    = neighbor_mask[i]            # (N,) True=real
    n_real  = mask.sum().item()

    print(f"\nScenario {i+1}  |  real neighbors: {int(n_real)}")
    print(f"  focal token weight:    {weights[0].item():.4f}")
    for j in range(neighbor_mask.shape[1]):
        status = "real   " if mask[j] else "PADDED "
        print(f"  neighbor {j+1} [{status}]:  {weights[j+1].item():.4f}")
