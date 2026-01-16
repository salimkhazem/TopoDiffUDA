"""Aggregate metrics across seeds."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    summaries = list(root.glob("**/summary.csv"))
    rows = []
    for summary in summaries:
        try:
            df = pd.read_csv(summary)
        except Exception:
            continue
        if df.empty:
            continue
        row = df.iloc[0].to_dict()
        parts = summary.parts
        if "runs" in parts:
            idx = parts.index("runs")
            exp = parts[idx + 1]
            dataset = parts[idx + 2] if len(parts) > idx + 2 else "unknown"
            model = parts[idx + 3] if len(parts) > idx + 3 else "unknown"
            seed = parts[idx + 4] if len(parts) > idx + 4 else "unknown"
        else:
            exp = dataset = model = seed = "unknown"
        row.update({"exp": exp, "dataset": dataset, "model": model, "seed": seed})
        rows.append(row)

    if not rows:
        print("No summary.csv files found.")
        return

    df = pd.DataFrame(rows)
    metrics = [c for c in df.columns if c not in {"exp", "dataset", "model", "seed"}]
    grouped = df.groupby(["exp", "dataset", "model"])[metrics]
    agg = grouped.agg(["mean", "std"]).reset_index()
    agg.columns = ["_".join([c for c in col if c]) if isinstance(col, tuple) else col for col in agg.columns]
    agg.to_csv(root / "aggregate.csv", index=False)
    agg.to_json(root / "aggregate.json", orient="records", indent=2)
    print(f"Saved aggregate to {root / 'aggregate.csv'}")


if __name__ == "__main__":
    main()
