import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.did_multiperiod import generate_panel_data


def main() -> None:
    config = json.loads((ROOT / "config" / "assignment.json").read_text(encoding="utf-8"))
    cleaned_dir = ROOT / "cleaned"
    cleaned_dir.mkdir(parents=True, exist_ok=True)

    panel_no_cov = generate_panel_data(config, heterogeneous_trend=False)
    panel_cov = generate_panel_data(config, heterogeneous_trend=True)

    panel_no_cov.to_csv(cleaned_dir / "panel_no_covariate_trend.csv", index=False)
    panel_cov.to_csv(cleaned_dir / "panel_covariate_trend.csv", index=False)


if __name__ == "__main__":
    main()
