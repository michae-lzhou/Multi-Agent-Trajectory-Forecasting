import sys
from pathlib import Path
from matf.utils.inspect import inspect_parquet

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_parquet.py path_to_file.parquet")
        sys.exit(1)

    file_path = Path(sys.argv[1])
    if not file_path.exists():
        print("File does not exist.")
        sys.exit(1)

    inspect_parquet(file_path)
