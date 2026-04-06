"""
src/evaluation/gemini_comparison.py
Compare CNN retrieval performance against the Gemini Vision API (commercial LLM baseline).

Protocol
--------
1. Load the test split (same indices as CNN training experiments, via split CSV).
2. For each test image, send it to Gemini with a structured prompt.
3. Map Gemini's free-text response to the nearest sentence in the corpus
   using cosine similarity of SBERT embeddings.
4. Compute the same retrieval metrics as evaluate.py (top-1 acc, cosine sim, etc.).
5. Save results to src/training/type-b/results/metrics/gemini_predictions.csv
   and append a row to results_summary.csv for direct comparison with CNN runs.

Requirements
------------
  pip install google-generativeai python-dotenv
  Set GEMINI_API_KEY in a .env file at the project root.

Usage
-----
  python src/evaluation/gemini_comparison.py
  python src/evaluation/gemini_comparison.py --embedding sbert --max-samples 100
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

# ── Path bootstrap ─────────────────────────────────────────────────────────────
_ROOT = next(p for p in Path(__file__).resolve().parents if (p / '.git').exists())
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config.paths import (
    EMBED_RESULTS_B,
    METRICS_B,
    TYPE_B_IMAGES,
    TYPE_B_SPLITS,
)
from pipelines.evaluation.evaluate import save_results

# ── Prompt ─────────────────────────────────────────────────────────────────────
# Structured prompt that matches the Type-B sentence format exactly.
# This gives Gemini the best chance of producing parseable output while
# still testing its visual understanding of colour, size, and digit identity.
_PROMPT = (
    "Describe this image in the following exact format: "
    "'[size] [colour] [number]'\n"
    "Rules:\n"
    "- size must be exactly one of: large, small\n"
    "- colour must be exactly one of: red, blue, green, yellow\n"
    "- number is the digit(s) shown in the image (e.g. 5, 42, 1337)\n"
    "Respond with only the description, nothing else."
)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _load_test_records(
    embedding_name: str,
    seed: int = 42,
) -> tuple[list[tuple[Path, str, torch.Tensor]], torch.Tensor, list[str]]:
    """
    Load test-split records and full corpus embeddings.

    Returns
    -------
    test_records   : list of (image_path, sentence, embedding) for test samples
    all_embeddings : Tensor (N_corpus, dim) — all corpus embeddings
    all_sentences  : list[str] — all corpus sentences (same order)
    """
    from src.config.paths import TYPE_B_SENTENCES, TYPE_B_IMAGE_MAP

    # Load embedding cache
    cache_path = EMBED_RESULTS_B / f'{embedding_name}_embedding_result_typeb.pt'
    if not cache_path.exists():
        raise FileNotFoundError(
            f'Embedding cache not found: {cache_path}\n'
            f'Run: python src/embeddings/computed-embeddings/type-b/'
            f'generate_embeddings_type_b.py --embedding {embedding_name}'
        )
    cache = torch.load(cache_path, map_location='cpu')
    all_sentences:  list[str]    = cache['sentences']
    all_embeddings: torch.Tensor = cache['embeddings'].float()
    sentence_to_emb = {s: all_embeddings[i] for i, s in enumerate(all_sentences)}

    # Merge CSVs
    image_map = pd.read_csv(TYPE_B_IMAGE_MAP)
    sentences = pd.read_csv(TYPE_B_SENTENCES)
    df = image_map.merge(sentences, on='sentence_id')

    # Load test split indices
    split_csv = TYPE_B_SPLITS / f'type_b_splits_seed{seed}.csv'
    if not split_csv.exists():
        raise FileNotFoundError(
            f'Split CSV not found: {split_csv}\n'
            'Run training first to generate splits:\n'
            '  python src/training/type-b/train_type_b.py --model cnn --embedding sbert --epochs 1'
        )
    split_df  = pd.read_csv(split_csv)
    test_idx  = split_df[split_df['split'] == 'test']['idx'].tolist()

    # Build test records
    test_records: list[tuple[Path, str, torch.Tensor]] = []
    for idx in test_idx:
        row      = df.iloc[idx]
        img_path = TYPE_B_IMAGES / row['filename']
        sentence = row['sentence']
        emb      = sentence_to_emb[sentence]
        test_records.append((img_path, sentence, emb))

    return test_records, all_embeddings, all_sentences


def _gemini_describe(model, img_path: Path, retries: int = 3) -> str:
    """Send image to Gemini and return the raw text response."""
    img = Image.open(img_path).convert('RGB')
    for attempt in range(retries):
        try:
            response = model.generate_content([img, _PROMPT])
            return response.text.strip().lower()
        except Exception as exc:
            if attempt < retries - 1:
                wait = 2 ** attempt
                print(f'    [retry {attempt+1}] {exc}  — waiting {wait}s')
                time.sleep(wait)
            else:
                raise


def _find_nearest(
    query_sentence: str,
    query_emb_model,                # SBERT model for encoding Gemini output
    all_embeddings: torch.Tensor,   # (N, dim)
    all_sentences:  list[str],
) -> tuple[str, float, int]:
    """
    Encode query_sentence with SBERT and find nearest corpus sentence.

    Returns (nearest_sentence, cosine_sim, rank_of_exact_match_if_any)
    """
    vec = torch.tensor(
        query_emb_model.encode([query_sentence], convert_to_numpy=True)
    ).float()                                   # (1, dim)

    sims = F.cosine_similarity(
        vec.unsqueeze(1),                        # (1, 1, dim)
        all_embeddings.unsqueeze(0),             # (1, N, dim)
        dim=2,
    ).squeeze(0)                                 # (N,)

    sorted_idx   = sims.argsort(descending=True)
    best_idx     = int(sorted_idx[0].item())
    best_sim     = float(sims[best_idx].item())
    best_sentence = all_sentences[best_idx]
    return best_sentence, best_sim, best_idx


# ══════════════════════════════════════════════════════════════════════════════
# Main comparison
# ══════════════════════════════════════════════════════════════════════════════

def run_gemini_comparison(
    embedding_name: str = 'sbert',
    max_samples:    int | None = None,
    seed:           int = 42,
    api_delay_s:    float = 0.5,
) -> None:
    """
    Run Gemini Vision API on Type-B test images and evaluate retrieval performance.

    Parameters
    ----------
    embedding_name : SBERT embedding used to map Gemini text output to corpus
                     (always 'sbert' since we need a common embedding space)
    max_samples    : limit the number of test samples (None = all)
    seed           : random seed used when creating train/val/test splits
    api_delay_s    : seconds to wait between API calls to avoid rate limiting
    """
    # ── Load API key ──────────────────────────────────────────────────────────
    try:
        from dotenv import load_dotenv
        load_dotenv(_ROOT / '.env')
    except ImportError:
        pass

    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        raise EnvironmentError(
            'GEMINI_API_KEY not found. Set it in a .env file at the project root:\n'
            '  GEMINI_API_KEY=your_key_here'
        )

    import google.generativeai as genai
    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
    print('[gemini] Model loaded: gemini-1.5-flash')

    # ── Load SBERT for encoding Gemini output ─────────────────────────────────
    from sentence_transformers import SentenceTransformer
    sbert = SentenceTransformer('all-MiniLM-L6-v2')
    print('[gemini] SBERT loaded for corpus matching')

    # ── Load test data ─────────────────────────────────────────────────────────
    test_records, all_embeddings, all_sentences = _load_test_records(
        embedding_name='sbert', seed=seed
    )
    if max_samples:
        test_records = test_records[:max_samples]
    print(f'[gemini] Evaluating {len(test_records)} test images')

    # ── Evaluate ───────────────────────────────────────────────────────────────
    rows: list[dict] = []
    n_top1 = 0
    n_top5 = 0

    for i, (img_path, true_sentence, _) in enumerate(test_records, 1):
        gemini_raw   = _gemini_describe(gemini_model, img_path)
        pred_sentence, best_sim, _ = _find_nearest(
            gemini_raw, sbert, all_embeddings, all_sentences
        )

        # Rank of true sentence in the similarity-sorted list
        true_emb = all_embeddings[all_sentences.index(true_sentence)].unsqueeze(0)
        query_vec = torch.tensor(sbert.encode([gemini_raw])).float()
        sims = F.cosine_similarity(
            query_vec.unsqueeze(1), all_embeddings.unsqueeze(0), dim=2
        ).squeeze(0)
        sorted_idx = sims.argsort(descending=True).tolist()
        true_idx   = all_sentences.index(true_sentence)
        rank       = sorted_idx.index(true_idx) + 1  # 1-indexed

        top1 = int(rank <= 1)
        top5 = int(rank <= 5)
        n_top1 += top1
        n_top5 += top5

        rows.append({
            'true_sentence':   true_sentence,
            'gemini_raw':      gemini_raw,
            'pred_sentence':   pred_sentence,
            'true_rank':       rank,
            'cosine_sim':      best_sim,
            'top_1_correct':   top1,
            'top_5_correct':   top5,
        })

        if i % 50 == 0 or i == len(test_records):
            print(f'  [{i}/{len(test_records)}]  top-1 so far: {n_top1/i:.3f}')

        if api_delay_s > 0:
            time.sleep(api_delay_s)

    n = len(rows)
    metrics = {
        'top_1_acc':       n_top1 / n,
        'top_5_acc':       n_top5 / n,
        'mean_cosine_sim': sum(r['cosine_sim'] for r in rows) / n,
        'mean_rank':       sum(r['true_rank'] for r in rows) / n,
        'n_test':          n,
    }

    results_df = pd.DataFrame(rows)
    save_results(metrics, results_df, METRICS_B, 'gemini_comparison')

    print(f'\n[gemini] Results:')
    print(f'  top-1 accuracy   : {metrics["top_1_acc"]:.4f}')
    print(f'  top-5 accuracy   : {metrics["top_5_acc"]:.4f}')
    print(f'  mean cosine sim  : {metrics["mean_cosine_sim"]:.4f}')
    print(f'  mean rank        : {metrics["mean_rank"]:.1f}')


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gemini Vision API comparison for Type-B')
    parser.add_argument('--embedding',   default='sbert',
                        help='Embedding used to map Gemini output to corpus (default: sbert)')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Limit number of test samples (default: all)')
    parser.add_argument('--seed',        type=int, default=42)
    parser.add_argument('--api-delay',   type=float, default=0.5,
                        help='Seconds to wait between API calls (default: 0.5)')
    args = parser.parse_args()

    run_gemini_comparison(
        embedding_name=args.embedding,
        max_samples=args.max_samples,
        seed=args.seed,
        api_delay_s=args.api_delay,
    )
