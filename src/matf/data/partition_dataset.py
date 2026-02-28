from pathlib import Path
import random
import shutil
import sys

SPLIT_CONFIG = {
    "train": 20_000,
    "val":    2_000,
    "test":   2_000,
}


def get_scenario_dirs(raw_split_dir: Path) -> list[Path]:
    dirs = [p for p in raw_split_dir.iterdir() if p.is_dir()]
    if not dirs:
        print(f"[WARNING] No subdirectories found in {raw_split_dir}")
    return dirs


def sample_and_copy(raw_dir, out_dir, n, seed, overwrite):
    scenarios = get_scenario_dirs(raw_dir)
    available = len(scenarios)

    if available < n:
        print(f"[WARNING] Only {available} available; requested {n}")
        n = available

    random.seed(seed)
    sampled = random.sample(scenarios, n)

    out_dir.mkdir(parents=True, exist_ok=True)

    for src in sampled:
        dst = out_dir / src.name

        if dst.exists():
            if overwrite:
                shutil.rmtree(dst)
            else:
                continue

        shutil.copytree(src, dst)


def run_partition(root: Path, seed: int, overwrite: bool):
    raw_base = root / "data" / "raw"
    processed_base = root / "data" / "processed"

    # Sanity check
    for split in SPLIT_CONFIG:
        if not (raw_base / split).exists():
            raise RuntimeError(f"Missing raw split: {split}")

    # Safety check
    for split in SPLIT_CONFIG:
        out_dir = processed_base / split
        if out_dir.exists() and any(out_dir.iterdir()):
            raise RuntimeError(f"Processed dir not empty: {out_dir}")

    # Run partition
    for split, n in SPLIT_CONFIG.items():
        sample_and_copy(
            raw_base / split,
            processed_base / split,
            n,
            seed,
            overwrite,
        )
