import argparse
import os
import random
import shutil
import sys
from pathlib import Path

SPLIT_CONFIG = {
    "train": 20_000,
    "val":    2_000,
    "test":   2_000,
}

DEFAULT_SEED = 42

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_scenario_dirs(raw_split_dir: Path) -> list[Path]:
    """Return all immediate subdirectories (one per scenario)."""
    dirs = [p for p in raw_split_dir.iterdir() if p.is_dir()]
    if not dirs:
        print(f"  [WARNING] No subdirectories found in {raw_split_dir}")
    return dirs


def sample_and_copy(
    raw_dir: Path,
    out_dir: Path,
    n: int,
    seed: int,
    overwrite: bool,
) -> None:
    scenarios = get_scenario_dirs(raw_dir)
    available = len(scenarios)

    if available < n:
        print(
            f"  [WARNING] Only {available} scenarios available in {raw_dir}; "
            f"requested {n}. Using all {available}."
        )
        n = available

    random.seed(seed)
    sampled = random.sample(scenarios, n)

    out_dir.mkdir(parents=True, exist_ok=True)

    already_done = 0
    copied = 0
    errors = 0

    for i, src in enumerate(sampled, 1):
        dst = out_dir / src.name

        if dst.exists():
            if overwrite:
                shutil.rmtree(dst)
            else:
                already_done += 1
                continue

        try:
            shutil.copytree(src, dst)
            copied += 1
        except Exception as e:
            print(f"  [ERROR] Could not copy {src.name}: {e}")
            errors += 1

        # Progress every 1000
        if i % 1_000 == 0 or i == n:
            print(f"    {i}/{n} processed...", flush=True)

    print(
        f"  Done — copied: {copied}, skipped (already exist): {already_done}, "
        f"errors: {errors}"
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Partition Argoverse 2 raw data into processed splits.")
    parser.add_argument("--project_root", type=str, default=".",
                        help="Path to project root (default: current directory)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"Random seed for reproducibility (default: {DEFAULT_SEED})")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-copy scenarios that already exist in processed/")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    raw_base = root / "data" / "raw"
    processed_base = root / "data" / "processed"

    print(f"Project root : {root}")
    print(f"Random seed  : {args.seed}")
    print(f"Overwrite    : {args.overwrite}")
    print()

    # Sanity-check raw directories exist
    for split in SPLIT_CONFIG:
        split_dir = raw_base / split
        if not split_dir.exists():
            print(f"[ERROR] Raw directory not found: {split_dir}")
            sys.exit(1)

    for split, n in SPLIT_CONFIG.items():
        raw_dir = raw_base / split
        out_dir = processed_base / split
        print(f"[{split}] Sampling {n:,} scenarios from {raw_dir}")
        print(f"        → {out_dir}")
        sample_and_copy(raw_dir, out_dir, n, seed=args.seed, overwrite=args.overwrite)
        print()

    print("All splits complete!")
    print()
    print("Quick sanity check — scenario counts in data/processed/:")
    for split in SPLIT_CONFIG:
        count = sum(1 for p in (processed_base / split).iterdir() if p.is_dir())
        print(f"  {split:5s}: {count:,}")


if __name__ == "__main__":
    main()
