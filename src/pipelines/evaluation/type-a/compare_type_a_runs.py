import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_dir", required=True)
    parser.add_argument("--output_csv", required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    summary_dir = Path(args.summary_dir)
    output_csv = Path(args.output_csv)

    summary_files = sorted(summary_dir.glob("*_summary.csv"))

    if not summary_files:
        raise FileNotFoundError(f"No summary files found in {summary_dir}")

    dfs = [pd.read_csv(f) for f in summary_files]
    combined = pd.concat(dfs, ignore_index=True)

    combined = combined.sort_values(
        by=["top1_accuracy", "avg_cosine_similarity", "full_structure_match_rate"],
        ascending=False
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_csv, index=False)

    print("\n=== Combined Run Comparison ===")
    print(combined.to_string(index=False))
    print(f"\nSaved comparison CSV to: {output_csv}")


if __name__ == "__main__":
    main()