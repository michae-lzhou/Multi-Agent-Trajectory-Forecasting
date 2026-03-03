import csv
from pathlib import Path

class TrainingLogger:
    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.file_path = self.run_dir / "training_log.csv"
        self.file = open(self.file_path, mode="w", newline="")
        self.writer = csv.writer(self.file)

        # Header
        self.writer.writerow(["epoch", "train_loss", "val_ade", "val_fde", "val_mr"])
        self.file.flush()

    def log(self, epoch, train_loss, val_ade, val_fde, val_mr, is_best=True):
        # Write row
        self.writer.writerow([epoch, train_loss, val_ade, val_fde, val_mr])
        self.file.flush()
        best_marker = " <- best" if is_best else ""

        # Clean terminal output
        print(
            f"[Epoch {epoch:03d}] "
            f"Train Loss: {train_loss:.4f} | "
            f"ADE: {val_ade:.4f} | "
            f"FDE: {val_fde:.4f} | "
            f"MR: {val_mr:.4f}"
            f"{best_marker}"
        )

    def close(self):
        if self.file:
            self.file.close()
            self.file = None
