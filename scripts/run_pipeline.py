import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    subprocess.run([sys.executable, str(ROOT / "scripts" / "run_cleaning.py")], check=True)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "run_analysis.py")], check=True)


if __name__ == "__main__":
    main()
