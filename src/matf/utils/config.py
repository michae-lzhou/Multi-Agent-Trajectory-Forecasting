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
    return (f"lstm_dec{cfg.model.decoder_input}"
            f"_h{cfg.model.hidden_size}"
            f"_l{cfg.model.num_layers}")

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
