import argparse
from pathlib import Path
from matf.data.preprocess import load_scenario, save_scenario


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
    )

    parser.add_argument(
        "--neighbor_cap",
        type=str,
        default="none",
        help='int or "none"',
    )

    parser.add_argument(
        "--data_prefix",
        type=str,
        default="baseline",
        help="baseline, info, orien, sort",
    )

    return parser.parse_args()


def parse_neighbor_cap(cap_str):
    if cap_str.lower() == "none":
        return None
    return int(cap_str)


def run(split, neighbor_cap, data_prefix):
    base_dir = Path("data/processed") / split
    scenario_dirs = sorted(base_dir.iterdir())

    count = 0

    for scenario_folder in scenario_dirs:
        if not scenario_folder.is_dir():
            continue

        parquet_files = list(scenario_folder.glob("scenario_*.parquet"))
        if not parquet_files:
            continue

        parquet_path = parquet_files[0]

        cap_str = "none" if neighbor_cap is None else str(neighbor_cap)
        out_path = scenario_folder / f"{data_prefix}_data_ncap_{cap_str}.npz"

        if out_path.exists():
            continue

        data = load_scenario(parquet_path, neighbor_cap)
        save_scenario(data, out_path)

        count += 1
        if count % 500 == 0:
            print(f"[{split}] Processed {count}")

    print(f"[{split}] Done: {count} scenarios")


if __name__ == "__main__":
    args = parse_args()
    neighbor_cap = parse_neighbor_cap(args.neighbor_cap)
    data_prefix = args.data_prefix

    for split in args.splits:
        run(split, neighbor_cap, data_prefix)
