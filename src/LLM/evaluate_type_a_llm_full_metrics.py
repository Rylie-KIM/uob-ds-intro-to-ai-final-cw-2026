import pandas as pd
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
input_csv = ROOT / "src" / "LLM" / "type_a_llm_outputs_full.csv"

COLOURS = ["red", "blue", "green", "yellow", "orange", "purple", "pink", "black"]
SIZES = ["big", "medium", "small"]
OBJECTS = ["circle", "triangle", "square", "diamond", "hexagon", "octagon"]
RELATIONS = ["above", "below", "left of", "right of"]

opposite_relation = {
    "above": "below",
    "below": "above",
    "left of": "right of",
    "right of": "left of"
}


def tokenize_label(text: str):
    text = str(text).strip().lower()

    found_colours = [c for c in COLOURS if c in text]
    found_sizes = [s for s in SIZES if s in text]
    found_objects = [o for o in OBJECTS if o in text]
    found_relation = next((r for r in RELATIONS if r in text), None)

    return {
        "colours": found_colours,
        "sizes": found_sizes,
        "objects": found_objects,
        "relation": found_relation,
    }


def overlap_count(a, b):
    return len(set(a) & set(b))


def parse_entity(text):
    pattern = r"(?:a|the)\s+(big|medium|small)\s+(red|blue|green|yellow|orange|purple|pink|black)\s+(circle|triangle|square|diamond|hexagon|octagon)"
    m = re.search(pattern, text)
    if not m:
        return None
    return {
        "size": m.group(1),
        "colour": m.group(2),
        "object": m.group(3)
    }


def parse_sentence(text):
    text = str(text).strip().lower()

    m = re.match(
        r"(above|below|left of|right of)\s+(a\s+(?:big|medium|small)\s+(?:red|blue|green|yellow|orange|purple|pink|black)\s+(?:circle|triangle|square|diamond|hexagon|octagon))\s+is\s+(a\s+(?:big|medium|small)\s+(?:red|blue|green|yellow|orange|purple|pink|black)\s+(?:circle|triangle|square|diamond|hexagon|octagon))$",
        text
    )
    if m:
        rel = m.group(1)
        entity2 = parse_entity(m.group(2))
        entity1 = parse_entity(m.group(3))
        if entity1 and entity2:
            return {"entity1": entity1, "relation": rel, "entity2": entity2}

    m = re.match(
        r"((?:a|the)\s+(?:big|medium|small)\s+(?:red|blue|green|yellow|orange|purple|pink|black)\s+(?:circle|triangle|square|diamond|hexagon|octagon))\s+(?:is|is positioned|can be seen)\s+(above|below|left of|right of)\s+((?:a|the)\s+(?:big|medium|small)\s+(?:red|blue|green|yellow|orange|purple|pink|black)\s+(?:circle|triangle|square|diamond|hexagon|octagon))$",
        text
    )
    if m:
        entity1 = parse_entity(m.group(1))
        rel = m.group(2)
        entity2 = parse_entity(m.group(3))
        if entity1 and entity2:
            return {"entity1": entity1, "relation": rel, "entity2": entity2}

    return None


def semantic_match(gold_struct, pred_struct):
    if gold_struct is None or pred_struct is None:
        return False

    if (
        gold_struct["entity1"] == pred_struct["entity1"]
        and gold_struct["relation"] == pred_struct["relation"]
        and gold_struct["entity2"] == pred_struct["entity2"]
    ):
        return True

    if (
        gold_struct["entity1"] == pred_struct["entity2"]
        and gold_struct["entity2"] == pred_struct["entity1"]
        and opposite_relation[gold_struct["relation"]] == pred_struct["relation"]
    ):
        return True

    return False


df = pd.read_csv(input_csv)

rows = []
for _, row in df.iterrows():
    gold = str(row["gold_label"]).strip().lower()
    pred = str(row["llm_output"]).strip().lower()

    gold_tok = tokenize_label(gold)
    pred_tok = tokenize_label(pred)

    gold_struct = parse_sentence(gold)
    pred_struct = parse_sentence(pred)

    exact_match = int(gold == pred)
    relation_match = int(gold_tok["relation"] == pred_tok["relation"]) if pred_tok["relation"] else 0

    colour_overlap = overlap_count(gold_tok["colours"], pred_tok["colours"])
    size_overlap = overlap_count(gold_tok["sizes"], pred_tok["sizes"])
    object_overlap = overlap_count(gold_tok["objects"], pred_tok["objects"])
    total_attribute_overlap = colour_overlap + size_overlap + object_overlap

    rows.append({
        "image_id": row["image_id"],
        "exact_match": exact_match,
        "relation_match": relation_match,
        "colour_overlap": colour_overlap,
        "size_overlap": size_overlap,
        "object_overlap": object_overlap,
        "total_attribute_overlap": total_attribute_overlap,
        "gold_parse_ok": int(gold_struct is not None),
        "pred_parse_ok": int(pred_struct is not None),
        "semantic_match": int(semantic_match(gold_struct, pred_struct)),
    })

out_df = pd.DataFrame(rows)

print("\n=== Type-A LLM Full Metrics ===")
print(f"Rows evaluated: {len(out_df)}")
print(f"Exact match rate: {out_df['exact_match'].mean():.3f}")
print(f"Relation match rate: {out_df['relation_match'].mean():.3f}")
print(f"Avg colour overlap: {out_df['colour_overlap'].mean():.3f}")
print(f"Avg size overlap: {out_df['size_overlap'].mean():.3f}")
print(f"Avg object overlap: {out_df['object_overlap'].mean():.3f}")
print(f"Avg total attribute overlap: {out_df['total_attribute_overlap'].mean():.3f}")
print(f"Gold parse success: {out_df['gold_parse_ok'].mean():.3f}")
print(f"Prediction parse success: {out_df['pred_parse_ok'].mean():.3f}")
print(f"Semantic match rate: {out_df['semantic_match'].mean():.3f}")

summary_csv = ROOT / "src" / "LLM" / "type_a_llm_metrics_summary.csv"
details_csv = ROOT / "src" / "LLM" / "type_a_llm_metrics_details.csv"

summary_df = pd.DataFrame([{
    "rows_evaluated": len(out_df),
    "exact_match_rate": out_df['exact_match'].mean(),
    "relation_match_rate": out_df['relation_match'].mean(),
    "avg_colour_overlap": out_df['colour_overlap'].mean(),
    "avg_size_overlap": out_df['size_overlap'].mean(),
    "avg_object_overlap": out_df['object_overlap'].mean(),
    "avg_total_attribute_overlap": out_df['total_attribute_overlap'].mean(),
    "gold_parse_success": out_df['gold_parse_ok'].mean(),
    "prediction_parse_success": out_df['pred_parse_ok'].mean(),
    "semantic_match_rate": out_df['semantic_match'].mean(),
}])

summary_df.to_csv(summary_csv, index=False)
out_df.to_csv(details_csv, index=False)

print(f"\nSaved summary to: {summary_csv}")
print(f"Saved details to: {details_csv}")