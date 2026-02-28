import argparse
from matf.data.eda import run_eda

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/processed/train")
    parser.add_argument("--output", type=str, default="outputs/plots/scenario_summary.png")
    parser.add_argument("--num_scenarios", type=int, default=50)

    args = parser.parse_args()

    run_eda(
        data_dir=args.data_dir,
        output_path=args.output,
        num_scenarios=args.num_scenarios,
    )


if __name__ == "__main__":
    main()
