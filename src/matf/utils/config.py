import yaml
import shutil
from types import SimpleNamespace
from pathlib import Path

def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{
            k: dict_to_namespace(v) for k, v in d.items()
        })
    elif isinstance(d, list):
        return [dict_to_namespace(v) for v in d]
    else:
        return d

def load_config(path):
    with open(path, "r") as f:
        config_dict = yaml.safe_load(f)
    cfg = dict_to_namespace(config_dict)
    cfg._raw = config_dict
    return cfg

def make_run_name(cfg):
    model_type = getattr(cfg.model, "type", "lstm")
    parts = [model_type]

    # common hyperparameters
    parts.append(f"h{cfg.model.hidden_size}")
    parts.append(f"l{cfg.model.num_layers}")

    if model_type == "lstm":
        parts.insert(1, f"dec{cfg.model.decoder_input}")
    else:  # social/transformer-style model
        parts.append(f"heads{cfg.model.num_heads}")
        if cfg.model.use_residual:
            parts.append("res")
        if cfg.model.use_layer_norm:
            parts.append("ln")
        parts.append(f"cs{cfg.model.cell_state}")
        parts.insert(1, f"dec{cfg.model.decoder_input}")

    return "_".join(parts)

def save_config(config_path, run_dir):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(config_path, run_dir / "config.yaml")

def print_config(cfg):
    print(yaml.dump(cfg._raw, default_flow_style=False))

# cfg = load_config("configs/lstm.yaml")
# print_config(cfg)
# 
# run_name = make_run_name(cfg)
# run_dir = Path("checkpoints") / run_name
# save_config("configs/lstm.yaml", run_dir)
# 
# # verify
# print(run_dir)
# print(list(run_dir.iterdir()))
