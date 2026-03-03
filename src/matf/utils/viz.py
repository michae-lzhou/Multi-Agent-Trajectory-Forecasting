import matplotlib.pyplot as plt
from pathlib import Path

# TODO: look into whether we want to stick to using best minADE to mark best
# epoch or a weighted combination of all three metrics

def save_training_plot(train_losses, val_ades, val_fdes, val_mrs, run_dir):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(train_losses) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ---- Left: Train Loss ----
    axes[0].plot(epochs, train_losses, marker='o', label="Train Loss")
    axes[0].set_title("Train Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].grid(True)

    # ---- Right: Validation Metrics ----
    axes[1].plot(epochs, val_ades, marker='o', label="ADE")
    axes[1].plot(epochs, val_fdes, marker='o', label="FDE")

    # ---- Twin axis for MR ----
    ax2 = axes[1].twinx()
    ax2.plot(
        epochs,
        val_mrs,
        marker='s',
        color='green',
        linestyle='--',
        label='MR'
    )
    ax2.set_ylabel("Miss Rate", color='green')
    ax2.tick_params(axis='y', labelcolor='green')

    # ---- Best epoch (based on ADE) ----
    best_epoch = val_ades.index(min(val_ades)) + 1

    axes[0].axvline(
        x=best_epoch,
        color='red',
        linestyle='--',
        alpha=0.5,
        label=f'Best epoch {best_epoch}'
    )

    axes[1].axvline(
        x=best_epoch,
        color='red',
        linestyle='--',
        alpha=0.5
    )

    # ---- Legends ----
    axes[0].legend()
    axes[1].legend(loc="upper left")
    ax2.legend(loc="upper right")

    axes[1].set_title("Validation Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("ADE / FDE")
    axes[1].grid(True)

    plt.tight_layout()

    save_path = run_dir / "training_plot.png"
    plt.savefig(save_path)
    plt.close()
