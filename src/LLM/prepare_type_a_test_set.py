import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

master_csv = ROOT / "src" / "data" / "type-a" / "master.csv"
test_indices_csv = ROOT / "src" / "pipelines" / "results" / "checkpoints" / "type-a" / "test_indices.csv"
output_csv = ROOT / "src" / "LLM" / "type_a_test_set.csv"

master_df = pd.read_csv(master_csv)
test_idx_df = pd.read_csv(test_indices_csv)

row_indices = test_idx_df["row_index"].tolist()
test_df = master_df.iloc[row_indices].reset_index(drop=True)

test_df.to_csv(output_csv, index=False)

print(f"Saved: {output_csv}")
print(test_df[["path", "label"]].head())
print(f"Rows: {len(test_df)}")