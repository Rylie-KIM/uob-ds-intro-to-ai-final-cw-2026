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


def parse_entity(text):
    """
    Parse phrases like:
    'a medium red diamond'
    'the big green octagon'
    """
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
    """
    Converts any allowed template into canonical form:
    {
        entity1: {size, colour, object},
        relation: ...,
        entity2: {size, colour, object}
    }
    """
    text = str(text).strip().lower()

    # Template 4:
    # "above a big green octagon is a small red triangle"
    m = re.match(
        r"(above|below|left of|right of)\s+(a\s+(?:big|medium|small)\s+(?:red|blue|green|yellow|orange|purple|pink|black)\s+(?:circle|triangle|square|diamond|hexagon|octagon))\s+is\s+(a\s+(?:big|medium|small)\s+(?:red|blue|green|yellow|orange|purple|pink|black)\s+(?:circle|triangle|square|diamond|hexagon|octagon))$",
        text
    )
    if m:
        rel = m.group(1)
        entity2_text = m.group(2)
        entity1_text = m.group(3)

        entity1 = parse_entity(entity1_text)
        entity2 = parse_entity(entity2_text)

        if entity1 and entity2:
            return {
                "entity1": entity1,
                "relation": rel,
                "entity2": entity2
            }

    # Templates 1,2,3:
    # "a small red triangle is above a big green octagon"
    # "the small red triangle is positioned above a big green octagon"
    # "a small red triangle can be seen above a big green octagon"
    m = re.match(
        r"((?:a|the)\s+(?:big|medium|small)\s+(?:red|blue|green|yellow|orange|purple|pink|black)\s+(?:circle|triangle|square|diamond|hexagon|octagon))\s+(?:is|is positioned|can be seen)\s+(above|below|left of|right of)\s+((?:a|the)\s+(?:big|medium|small)\s+(?:red|blue|green|yellow|orange|purple|pink|black)\s+(?:circle|triangle|square|diamond|hexagon|octagon))$",
        text
    )
    if m:
        entity1 = parse_entity(m.group(1))
        rel = m.group(2)
        entity2 = parse_entity(m.group(3))

        if entity1 and entity2:
            return {
                "entity1": entity1,
                "relation": rel,
                "entity2": entity2
            }

    return None


def semantic_match(gold_struct, pred_struct):
    """
    Checks if two parsed sentences describe the same scene.
    Handles reversed object order + inverse relation.
    """
    if gold_struct is None or pred_struct is None:
        return False

    # direct match
    if (
        gold_struct["entity1"] == pred_struct["entity1"]
        and gold_struct["relation"] == pred_struct["relation"]
        and gold_struct["entity2"] == pred_struct["entity2"]
    ):
        return True

    # reversed match
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

    gold_struct = parse_sentence(gold)
    pred_struct = parse_sentence(pred)

    rows.append({
        "image_id": row["image_id"],
        "gold_label": gold,
        "llm_output": pred,
        "gold_parse_ok": gold_struct is not None,
        "pred_parse_ok": pred_struct is not None,
        "semantic_match": int(semantic_match(gold_struct, pred_struct))
    })

out_df = pd.DataFrame(rows)

print("\n=== Semantic Evaluation ===")
print(out_df[[
    "image_id",
    "gold_parse_ok",
    "pred_parse_ok",
    "semantic_match"
]])

print("\n=== Summary ===")
print(f"Gold parse success: {out_df['gold_parse_ok'].mean():.3f}")
print(f"Prediction parse success: {out_df['pred_parse_ok'].mean():.3f}")
print(f"Semantic match rate: {out_df['semantic_match'].mean():.3f}")