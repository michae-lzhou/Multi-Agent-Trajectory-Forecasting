import argparse
from pathlib import Path

from matf.data.partition_dataset import run_partition


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", type=str, default=".")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")

    args = parser.parse_args()

    root = Path(args.project_root).resolve()

    print(f"Running partition on: {root}")
    run_partition(root, seed=args.seed, overwrite=args.overwrite)
    print("Done.")


if __name__ == "__main__":
    main()
