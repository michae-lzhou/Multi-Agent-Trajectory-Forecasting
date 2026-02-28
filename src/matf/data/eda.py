from pathlib import Path
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# -----------------------------
# Core Feature Functions
# -----------------------------

def compute_speed(df: pd.DataFrame) -> pd.Series:
    return np.sqrt(df["velocity_x"]**2 + df["velocity_y"]**2)


def compute_angle_diff(df: pd.DataFrame) -> pd.Series:
    vel_angle = np.arctan2(df["velocity_y"], df["velocity_x"])
    angle_diff = df["heading"] - vel_angle
    return np.abs(np.arctan2(np.sin(angle_diff), np.cos(angle_diff)))


def load_random_scenarios(data_dir: Path, n: int, seed: int):
    files = glob.glob(str(data_dir / "*/*.parquet"))
    np.random.seed(seed)
    return np.random.choice(files, size=min(n, len(files)), replace=False)


# -----------------------------
# Data Collection
# -----------------------------

def collect_statistics(files):

    neighbor_counts = []
    track_types = []
    vel_all = []
    angle_diff_all = []
    timesteps_stats = []
    coord_stats = []

    for f in files:
        df = pd.read_parquet(f)

        neighbors = df[df["object_category"].isin([1, 2])]
        neighbor_counts.append(neighbors["track_id"].nunique())
        track_types.append(neighbors["object_type"].value_counts())

        df["speed"] = compute_speed(df)
        vel_all.append(df[["object_type", "object_category", "speed"]])

        ts = (
            df[df["object_category"].isin([1, 2, 3])]
            .groupby("object_category")["timestep"]
            .nunique()
            .describe()
        )
        timesteps_stats.append(ts)

        coord_stats.append(df[["position_x", "position_y"]].describe())

        df["angle_diff"] = compute_angle_diff(df)
        angle_diff_all.append(df[["object_category", "angle_diff"]])

    return {
        "neighbor_counts": np.array(neighbor_counts),
        "track_types_df": pd.DataFrame(track_types).fillna(0),
        "timesteps_df": pd.concat(timesteps_stats, axis=1).T.describe(),
        "coords_df": pd.concat(coord_stats, axis=0).describe(),
        "vel_df": pd.concat(vel_all, axis=0),
        "angle_diff_df": pd.concat(angle_diff_all, axis=0),
    }


# -----------------------------
# Visualization
# -----------------------------

def plot_summary(stats: dict, output_path: Path):

    neighbor_counts = stats["neighbor_counts"]
    track_types_df = stats["track_types_df"]
    timesteps_df = stats["timesteps_df"]
    coords_df = stats["coords_df"]
    vel_df = stats["vel_df"]
    angle_diff_df = stats["angle_diff_df"]

    # print("\n[Neighbor Counts]")
    # 
    # p5 = np.percentile(neighbor_counts, 5)
    # p50 = np.percentile(neighbor_counts, 50)
    # p95 = np.percentile(neighbor_counts, 95)
    # 
    # print(f"P5 : {p5:.2f}")
    # print(f"P50: {p50:.2f}")
    # print(f"P95: {p95:.2f}")
    # print(f"Mean: {neighbor_counts.mean():.2f}")

    sns.set(style="whitegrid")
    fig, axs = plt.subplots(3, 2, figsize=(16, 12))

    sns.histplot(neighbor_counts, bins=20, ax=axs[0, 0])
    axs[0, 0].set_title("Neighbor Counts (Categories 1+2)")

    track_types_df.plot(kind="bar", stacked=True, ax=axs[0, 1])
    axs[0, 1].set_title("Track Type Composition")

    sns.violinplot(data=vel_df, x="object_type", y="speed", ax=axs[1, 0])
    axs[1, 0].set_title("Velocity Distribution")

    timesteps_df["mean"].plot(kind="bar", ax=axs[1, 1])
    axs[1, 1].set_title("Time Step Coverage")

    coords_df[["position_x", "position_y"]].plot(kind="bar", ax=axs[2, 0])
    axs[2, 0].set_title("Coordinate Statistics")

    sns.violinplot(data=angle_diff_df, x="object_category", y="angle_diff", ax=axs[2, 1])
    axs[2, 1].set_title("Heading vs Velocity Alignment")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


# -----------------------------
# Public Entry
# -----------------------------

def run_eda(data_dir: str, output_path: str, num_scenarios: int = 100, seed: int = 42):

    data_dir = Path(data_dir)
    output_path = Path(output_path)

    files = load_random_scenarios(data_dir, num_scenarios, seed)
    stats = collect_statistics(files)
    plot_summary(stats, output_path)

    return stats
