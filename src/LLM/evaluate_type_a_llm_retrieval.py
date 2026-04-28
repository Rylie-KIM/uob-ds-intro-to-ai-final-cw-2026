import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
from sentence_transformers import SentenceTransformer


master_csv = ROOT / "src" / "data" / "type-a" / "master.csv"
llm_csv = ROOT / "src" / "LLM" / "type_a_llm_outputs_full.csv"

summary_csv = ROOT / "src" / "LLM" / "type_a_llm_retrieval_summary.csv"
details_csv = ROOT / "src" / "LLM" / "type_a_llm_retrieval_details.csv"


def load_tensor(path_str: str) -> torch.Tensor:
    path = Path(path_str)
    if not path.is_absolute():
        path = ROOT / path
    return torch.load(path, map_location="cpu").float().squeeze()


def normalize_text(text: str) -> str:
    return " ".join(str(text).strip().split()).lower()


print("Loading master.csv...")
master_df = pd.read_csv(master_csv)

print("Loading LLM outputs...")
llm_df = pd.read_csv(llm_csv)

print("Loading full Type-A SBERT corpus embeddings...")
all_sentences = [normalize_text(x) for x in master_df["label"].tolist()]
all_embeddings = []

for emb_path in master_df["sbert_emb"].tolist():
    emb = load_tensor(emb_path)
    all_embeddings.append(emb)

all_embeddings = torch.stack(all_embeddings)
all_embeddings = F.normalize(all_embeddings, dim=1)

print(f"Corpus size: {len(all_sentences)}")

print("Loading SBERT model for LLM outputs...")
sbert_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

# Faster lookup for true sentence index
sentence_to_idx = {sent: idx for idx, sent in enumerate(all_sentences)}

rows = []

for i, row in llm_df.iterrows():
    true_sentence = normalize_text(row["gold_label"])
    llm_text = normalize_text(row["llm_output"])

    try:
        query_vec = sbert_model.encode(
            [llm_text],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        query_vec = torch.tensor(query_vec).float()

        sims = F.cosine_similarity(query_vec, all_embeddings, dim=1)
        sorted_idx = sims.argsort(descending=True).tolist()

        best_idx = sorted_idx[0]
        pred_sentence = all_sentences[best_idx]
        best_cosine = float(sims[best_idx].item())

        if true_sentence not in sentence_to_idx:
            raise ValueError(f"True sentence not found in corpus: {true_sentence}")

        true_idx = sentence_to_idx[true_sentence]
        true_rank = sorted_idx.index(true_idx) + 1  # 1-indexed

        rows.append({
            "image_id": row["image_id"],
            "gold_label": true_sentence,
            "llm_output": llm_text,
            "pred_sentence": pred_sentence,
            "cosine_sim": best_cosine,
            "true_rank": true_rank,
            "top_1_correct": int(true_rank <= 1),
            "top_5_correct": int(true_rank <= 5),
        })

    except Exception as e:
        rows.append({
            "image_id": row["image_id"],
            "gold_label": true_sentence,
            "llm_output": llm_text,
            "pred_sentence": f"ERROR: {e}",
            "cosine_sim": np.nan,
            "true_rank": np.nan,
            "top_1_correct": 0,
            "top_5_correct": 0,
        })

    if (i + 1) % 100 == 0:
        print(f"Processed {i + 1}/{len(llm_df)}")

details_df = pd.DataFrame(rows)

valid_ranks = details_df["true_rank"].dropna().to_numpy()
valid_cosines = details_df["cosine_sim"].dropna().to_numpy()

summary = {
    "rows_evaluated": len(details_df),
    "valid_rows": int(len(valid_ranks)),
    "top_1_acc": float(details_df["top_1_correct"].mean()),
    "top_5_acc": float(details_df["top_5_correct"].mean()),
    "mrr": float(np.mean(1.0 / valid_ranks)) if len(valid_ranks) > 0 else np.nan,
    "mean_cosine_sim": float(np.mean(valid_cosines)) if len(valid_cosines) > 0 else np.nan,
    "mean_rank": float(np.mean(valid_ranks)) if len(valid_ranks) > 0 else np.nan,
    "median_rank": float(np.median(valid_ranks)) if len(valid_ranks) > 0 else np.nan,
}

summary_df = pd.DataFrame([summary])

details_df.to_csv(details_csv, index=False)
summary_df.to_csv(summary_csv, index=False)

print("\n=== Type-A LLM Retrieval Metrics ===")
for k, v in summary.items():
    print(f"{k}: {v}")

print(f"\nSaved summary to: {summary_csv}")
print(f"Saved details to: {details_csv}")