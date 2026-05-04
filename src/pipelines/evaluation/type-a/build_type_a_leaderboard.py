from pathlib import Path
import pandas as pd

METRICS_DIR = Path("src/pipelines/results/metrics/type-a")
OUT_PATH = METRICS_DIR / "type_a_leaderboard.csv"

W_MRR = 0.35
W_MEDIAN_RANK = 0.25
W_TOP1 = 0.20
W_TOP5 = 0.15
W_COSINE = 0.05

CORPUS_SIZE = 1000  # ranking is computed against the 1000-item Type-A test subset, not the full dataset
EMBED_DIMS = {
    "sbert": 384,
    "TB_pooler": 312,
    "TB_mean": 312,
    "B_pooler": 768,
    "B_mean": 768,
    "P_wvec": 300,
}

def infer_stage(run_name: str) -> str:
    prefix = int(run_name.split("_", 1)[0])
    return "Stage 1" if prefix <= 4 else "Stage 2"

def main():
    summary_files = sorted(METRICS_DIR.glob("*_summary.csv"))
    if not summary_files:
        raise FileNotFoundError(f"No summary CSVs found in {METRICS_DIR}")

    frames = [pd.read_csv(f) for f in summary_files]
    df = pd.concat(frames, ignore_index=True)

    required = {
        "run_name",
        "model_name",
        "embedding_name",
        "loss_name",
        "top1_accuracy",
        "top5_accuracy",
        "avg_cosine_similarity",
        "test_mrr",
        "test_median_rank",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns for leaderboard: {sorted(missing)}. "
            f"Rerun evaluate_type_a_run.py after adding MRR/median-rank support."
        )

    df["stage"] = df["run_name"].apply(infer_stage)
    df["embedding_dim"] = df["embedding_name"].map(EMBED_DIMS)
    df["embedding_display"] = df.apply(
        lambda r: f"{r['embedding_name']} ({int(r['embedding_dim'])})",
        axis=1,
    )

    # shared normalization across all runs
    max_mrr = df["test_mrr"].max()
    max_top1 = df["top1_accuracy"].max()
    max_top5 = df["top5_accuracy"].max()

    df["mrr_norm"] = df["test_mrr"] / max_mrr if max_mrr > 0 else 0.0
    df["median_rank_norm"] = 1.0 - (df["test_median_rank"] / CORPUS_SIZE)
    df["top1_norm"] = df["top1_accuracy"] / max_top1 if max_top1 > 0 else 0.0
    df["top5_norm"] = df["top5_accuracy"] / max_top5 if max_top5 > 0 else 0.0
    df["cosine_norm"] = df["avg_cosine_similarity"]

    df["composite_score"] = (
        W_MRR * df["mrr_norm"]
        + W_MEDIAN_RANK * df["median_rank_norm"]
        + W_TOP1 * df["top1_norm"]
        + W_TOP5 * df["top5_norm"]
        + W_COSINE * df["cosine_norm"]
    )

    df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)

    leaderboard = df[
        [
            "stage",
            "rank",
            "run_name",
            "model_name",
            "embedding_display",
            "top1_accuracy",
            "top5_accuracy",
            "test_mrr",
            "test_median_rank",
            "avg_cosine_similarity",
            "composite_score",
        ]
    ].copy()

    leaderboard.columns = [
        "Stage",
        "Rank",
        "Run",
        "Model",
        "Embedding (dim)",
        "T@1",
        "T@5",
        "MRR",
        "Med.Rk",
        "MeanCos.",
        "Comp.",
    ]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    leaderboard.to_csv(OUT_PATH, index=False)

    print("\n=== Type-A Leaderboard ===")
    print(leaderboard.to_string(index=False))
    print(f"\nSaved to: {OUT_PATH}")

if __name__ == "__main__":
    main()