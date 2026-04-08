# Determines the optimal n_components for TruncatedSVD by finding the elbow in the cumulative explained variance curve.
# "elbow" is where cumulative variance stops increasing steeply.
from __future__ import annotations

import sys
from pathlib import Path
from typing import TypeAlias, TypedDict

import matplotlib
matplotlib.use("Agg")   # headless — no display required
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

_HERE     = Path(__file__).resolve().parent
_ROOT     = _HERE.parent.parent.parent.parent
_DATA_DIR = _ROOT / "src" / "data" / "type-b"
_OUT_DIR  = _HERE   

MAX_COMPONENTS = 500   # upper bound for SVD sweep; capped at vocab_size - 1
RANDOM_STATE   = 42


class VarianceRow(TypedDict):
    n_components:   int    # SVD dimension tested
    explained_var:  float  # variance explained by this component alone
    cumulative_var: float  
    marginal_gain:  float  
    
VarianceDF: TypeAlias = pd.DataFrame  # schema: VarianceRow (one row per n_components)

def load_sentences() -> list[str]:
    sentences_df = pd.read_csv(_DATA_DIR / "sentences_b.csv")
    sentences = sentences_df["sentence"].drop_duplicates().tolist()
    print(f"[data]  {len(sentences)} unique sentences loaded from Type-B corpus")
    return sentences


def run_variance_analysis(sentences: list[str]) -> tuple[VarianceDF, int]:
    """
    Fit TF-IDF vectoriser and sweep TruncatedSVD from 1 to MAX_COMPONENTS.

    Returns
    -------
    tuple[pd.DataFrame, int]
        df         : DataFrame[VarianceRow] — one row per SVD dimension sweep
        vocab_size : number of unique tokens in the TF-IDF vocabulary
    """
    print("[tfidf] Fitting TF-IDF vectoriser on full vocabulary ...")
    tfidf = TfidfVectorizer(lowercase=True, max_features=None)
    tfidf_matrix = tfidf.fit_transform(sentences)
    vocab_size = len(tfidf.vocabulary_)
    print(f"[tfidf] vocab_size={vocab_size}  matrix shape={tfidf_matrix.shape}")

    # SVD can have at most min(n_samples, n_features) - 1 components
    max_k = min(MAX_COMPONENTS, vocab_size - 1, len(sentences) - 1)
    print(f"[svd]   Fitting TruncatedSVD with n_components={max_k} ...")

    svd = TruncatedSVD(n_components=max_k, random_state=RANDOM_STATE)
    svd.fit(tfidf_matrix)

    cumulative = np.cumsum(svd.explained_variance_ratio_)
    marginal   = np.diff(cumulative, prepend=0.0)

    df = pd.DataFrame({
        "n_components":   np.arange(1, max_k + 1, dtype=int),
        "explained_var":  svd.explained_variance_ratio_,
        "cumulative_var": cumulative,
        "marginal_gain":  marginal,
    })

    # Annotate threshold crossings
    for threshold in [0.80, 0.85, 0.90, 0.95]:
        idx = int(np.searchsorted(cumulative, threshold))
        if idx < max_k:
            k_at = idx + 1
            print(f"[info]  {threshold*100:.0f}% variance explained at n_components = {k_at}")

    return df, vocab_size


# ── Save CSV ───────────────────────────────────────────────────────────────────

def save_csv(df: pd.DataFrame) -> Path:
    out_path = _OUT_DIR / "tfidf_lsa_variance_type_b.csv"
    df.to_csv(out_path, index=False, float_format="%.6f")
    print(f"[saved] {out_path.name}  ({len(df)} rows)")
    return out_path


# ── Plot ───────────────────────────────────────────────────────────────────────

def save_plot(df: pd.DataFrame, vocab_size: int) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"TF-IDF LSA — Explained Variance Analysis (Type-B)\n"
        f"vocab_size={vocab_size}, corpus=10,008 sentences",
        fontsize=13,
    )

    # ── Left: cumulative explained variance ────────────────────────────────────
    ax = axes[0]
    ax.plot(df["n_components"], df["cumulative_var"], color="steelblue", linewidth=1.5)
    ax.set_xlabel("n_components")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.set_title("Cumulative Explained Variance")
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)

    # Mark threshold lines
    colours = {"80%": "#e67e22", "90%": "#e74c3c", "95%": "#8e44ad"}
    for label, threshold, colour in [
        ("80%", 0.80, colours["80%"]),
        ("90%", 0.90, colours["90%"]),
        ("95%", 0.95, colours["95%"]),
    ]:
        ax.axhline(threshold, linestyle="--", color=colour, alpha=0.7, label=f"{label} threshold")
        idx = int(np.searchsorted(df["cumulative_var"].values, threshold))
        if idx < len(df):
            k_val = int(df["n_components"].iloc[idx])
            ax.axvline(k_val, linestyle=":", color=colour, alpha=0.5)
            ax.annotate(
                f"k={k_val}",
                xy=(k_val, threshold),
                xytext=(k_val + 5, threshold - 0.04),
                fontsize=8,
                color=colour,
            )
    ax.legend(fontsize=8)

    # ── Right: marginal gain (per-component variance) ──────────────────────────
    ax2 = axes[1]
    ax2.plot(df["n_components"], df["marginal_gain"], color="darkorange", linewidth=1.0)
    ax2.set_xlabel("n_components")
    ax2.set_ylabel("Marginal Gain (variance added per component)")
    ax2.set_title("Marginal Gain per Additional Component")
    ax2.grid(True, alpha=0.3)

    # Mark candidate range
    for k in [100, 200, 300]:
        if k <= len(df):
            ax2.axvline(k, linestyle="--", color="grey", alpha=0.5)
            ax2.annotate(f"k={k}", xy=(k, ax2.get_ylim()[1] * 0.9),
                         fontsize=8, color="grey", ha="center")

    plt.tight_layout()
    out_path = _OUT_DIR / "tfidf_lsa_variance_type_b.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path.name}")
    return out_path

def print_summary(df: pd.DataFrame) -> None:
    """Print key checkpoints for quick reading."""
    checkpoints = [50, 100, 150, 200, 250, 300, 400, 500]
    rows = df[df["n_components"].isin(checkpoints)].copy()
    rows["cumulative_var_%"] = (rows["cumulative_var"] * 100).round(2)
    rows["marginal_gain_%"]  = (rows["marginal_gain"]  * 100).round(4)
    print("\n── Key checkpoints ────────────────────────────────")
    print(rows[["n_components", "cumulative_var_%", "marginal_gain_%"]].to_string(index=False))
    print()

def main() -> None:
    sentences = load_sentences()
    df, vocab_size = run_variance_analysis(sentences)
    save_csv(df)
    save_plot(df, vocab_size)
    print_summary(df)
    print(
        "Recommendation: choose n_components at the 90–95% cumulative variance threshold.\n"
        "Higher n_components → better number token separation but harder CNN regression.\n"
        f"Results saved to: {_OUT_DIR}"
    )


if __name__ == "__main__":
    main()
