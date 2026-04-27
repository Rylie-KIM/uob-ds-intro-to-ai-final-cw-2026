import pandas as pd
import time
from pathlib import Path
from gemini import Gemini

ROOT = Path(__file__).resolve().parent.parent.parent

input_csv = ROOT / "src" / "LLM" / "type_a_test_set.csv"
output_csv = ROOT / "src" / "LLM" / "type_a_llm_outputs_full.csv"

df = pd.read_csv(input_csv)

sample_df = df.copy()

results = []

for _, row in sample_df.iterrows():
    path = row["path"]
    gold_label = row["label"]

    # extract image id from path like src/data/images/type-a/png/59.png
    img_id = int(Path(path).stem)

    try:
        gem = Gemini(img_id)
        llm_output = gem.get_output()
        print(f"{img_id}: {llm_output}")
    except Exception as e:
        llm_output = f"ERROR: {e}"
        print(f"{img_id}: ERROR -> {e}")

    results.append({
        "image_id": img_id,
        "path": path,
        "gold_label": gold_label,
        "llm_output": llm_output,
    })

    time.sleep(1)

out_df = pd.DataFrame(results)
out_df.to_csv(output_csv, index=False)

print(f"\nSaved sample outputs to: {output_csv}")