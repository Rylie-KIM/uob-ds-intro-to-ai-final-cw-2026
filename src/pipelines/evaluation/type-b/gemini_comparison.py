"""
src/pipelines/evaluation/type-b/gemini_comparison.py
Compare CNN retrieval performance against the Gemini Vision API (commercial LLM baseline).

Protocol
--------
1. Load the test split (same indices as CNN training experiments, via split CSV).
2. For each test image, send it to Gemini with a structured prompt.
3. Map Gemini's free-text response to the nearest sentence in the corpus
   using cosine similarity of SBERT embeddings.
4. Compute the same retrieval metrics as eval_metrics_b.py (top-k acc, MRR, etc.).
5. Save per-sample predictions to PREDICTIONS_B/gemini_predictions.csv
   and summary metrics to PREDICTIONS_B/gemini_summary.csv.

Requirements
------------
  pip install google-generativeai python-dotenv sentence-transformers
  Set GEMINI_API_KEY in a .env file at the project root.

Usage
-----
  python src/pipelines/evaluation/type-b/gemini_comparison.py
  python src/pipelines/evaluation/type-b/gemini_comparison.py --max-samples 100
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# ── Path bootstrap ──────────────────────────────────────────────────────────────
_ROOT = next(p for p in Path(__file__).resolve().parents if (p / '.git').exists())
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config.paths import (
    EMBED_RESULTS_B,
    PREDICTIONS_B_COMMERCIAL_AI,
    TYPE_B_IMAGE_MAP,
    TYPE_B_IMAGES,
    TYPE_B_SENTENCES,
    TYPE_B_SPLITS,
)

# ── Prompt ──────────────────────────────────────────────────────────────────────
# Structured prompt matching the Type-B sentence format exactly.
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
    seed: int = 42,
) -> tuple[list[tuple[Path, str, int]], torch.Tensor, list[str]]:
    """
    Load test-split records and full SBERT corpus embeddings.

    Returns
    -------
    test_records   : list of (image_path, sentence, n_digits) for test samples
    all_embeddings : Tensor (N_corpus, dim) — L2-normalised SBERT corpus embeddings
    all_sentences  : list[str] — corpus sentences (same order as all_embeddings)
    """
    cache_path = EMBED_RESULTS_B / 'sbert_embedding_result_typeb.pt'
    if not cache_path.exists():
        raise FileNotFoundError(
            f'SBERT embedding cache not found: {cache_path}\n'
            'Run: python src/embeddings/computed-embeddings/type-b/'
            'generate_embeddings_type_b.py --embedding sbert'
        )
    cache = torch.load(cache_path, map_location='cpu')
    all_sentences:  list[str]    = cache['sentences']
    all_embeddings: torch.Tensor = F.normalize(cache['embeddings'].float(), dim=1)

    image_map    = pd.read_csv(TYPE_B_IMAGE_MAP)
    sentences_df = pd.read_csv(TYPE_B_SENTENCES)
    df = image_map.merge(sentences_df, on='sentence_id')

    split_csv = TYPE_B_SPLITS / f'type_b_splits_seed{seed}.csv'
    if not split_csv.exists():
        raise FileNotFoundError(
            f'Split CSV not found: {split_csv}\n'
            'Run training first to generate splits.'
        )
    split_df = pd.read_csv(split_csv)
    test_idx = split_df[split_df['split'] == 'test']['idx'].tolist()

    test_records: list[tuple[Path, str, int]] = []
    for idx in test_idx:
        row = df.iloc[idx]
        test_records.append((
            TYPE_B_IMAGES / row['filename'],
            str(row['sentence']),
            int(row['n_digits']),
        ))

    return test_records, all_embeddings, all_sentences


def _gemini_describe(
    client, model: str, img_path: Path, retries: int = 3
) -> tuple[str, dict]:
    """
    Send image to Gemini and return (lowercased text, raw response as dict).

    The raw dict preserves the full API response including usage_metadata,
    finish_reason, safety_ratings, etc.
    """
    from google.genai import types as genai_types

    with open(img_path, 'rb') as f:
        img_bytes = f.read()
    img_part = genai_types.Part.from_bytes(data=img_bytes, mime_type='image/png')

    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=[img_part, _PROMPT],
            )
            text = response.text.strip().lower()
            raw  = response.model_dump()   # full response as plain dict
            return text, raw
        except Exception as exc:
            # parse retry_delay from 429 response if available
            wait = 2 ** attempt
            exc_str = str(exc)
            if '429' in exc_str:
                import re
                m = re.search(r"'seconds':\s*(\d+)", exc_str)
                if m:
                    wait = int(m.group(1)) + 2  # honour the server's retry-after
            if attempt < retries - 1:
                print(f'    [retry {attempt+1}/{retries-1}] waiting {wait}s — {exc_str[:120]}')
                time.sleep(wait)
            else:
                raise


def _retrieve(
    gemini_text:    str,
    sbert_model,
    all_embeddings: torch.Tensor,   # (N, dim) — already L2-normalised
    all_sentences:  list[str],
    true_sentence:  str,
    top_k:          tuple[int, ...] = (1, 2, 3, 4, 5),
) -> dict:
    """
    Encode gemini_text with SBERT, retrieve nearest corpus sentence,
    and compute rank of the ground-truth sentence.

    Returns a dict with: pred_sentence, cosine_sim, true_rank, top_k_correct flags.
    """
    vec = torch.tensor(
        sbert_model.encode([gemini_text], convert_to_numpy=True)
    ).float()                                   # (1, dim)
    vec = F.normalize(vec, dim=1)               # L2-normalise to match corpus

    sims       = (vec @ all_embeddings.T).squeeze(0)   # (N,)
    sorted_idx = sims.argsort(descending=True).tolist()

    best_idx      = sorted_idx[0]
    pred_sentence = all_sentences[best_idx]
    cosine_sim    = float(sims[best_idx].item())

    true_idx = all_sentences.index(true_sentence)
    rank     = sorted_idx.index(true_idx) + 1   # 1-indexed

    return {
        'pred_sentence': pred_sentence,
        'cosine_sim':    cosine_sim,
        'true_rank':     rank,
        **{f'top_{k}_correct': int(rank <= k) for k in top_k},
    }


def _build_metrics(rows: list[dict], model: str) -> dict:
    n     = len(rows)
    ranks = np.array([r['true_rank'] for r in rows])
    return {
        'model':           model,
        'embedding':       'sbert',
        'n_test':          n,
        'top_1_acc':       float(np.mean([r['top_1_correct'] for r in rows])),
        'top_2_acc':       float(np.mean([r['top_2_correct'] for r in rows])),
        'top_3_acc':       float(np.mean([r['top_3_correct'] for r in rows])),
        'top_5_acc':       float(np.mean([r['top_5_correct'] for r in rows])),
        'mrr':             float(np.mean(1.0 / ranks)),
        'mean_cosine_sim': float(np.mean([r['cosine_sim'] for r in rows])),
        'mean_rank':       float(np.mean(ranks)),
        'median_rank':     float(np.median(ranks)),
    }


def _save_results(
    rows: list[dict], raw_responses: list[dict], metrics: dict, out_dir: Path
) -> None:
    """
    Save per-sample predictions, raw API responses, and summary metrics.

    Files written
    -------------
    gemini_predictions.csv    — one row per test sample (metrics + text)
    gemini_raw_responses.jsonl — one JSON object per line (full API response)
    gemini_summary.csv        — single-row aggregate metrics
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_path = out_dir / 'gemini_predictions.csv'
    pd.DataFrame(rows).to_csv(pred_path, index=False)
    print(f'[saved] predictions  → {pred_path}')

    jsonl_path = out_dir / 'gemini_raw_responses.jsonl'
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for entry in raw_responses:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    print(f'[saved] raw JSON     → {jsonl_path}')

    summary_path = out_dir / 'gemini_summary.csv'
    pd.DataFrame([metrics]).to_csv(summary_path, index=False)
    print(f'[saved] summary      → {summary_path}')


# ══════════════════════════════════════════════════════════════════════════════
# Main comparison
# ══════════════════════════════════════════════════════════════════════════════

def run_gemini_comparison(
    max_samples: int | None = None,
    seed:        int = 42,
    api_delay_s: float = 0.5,
) -> None:
    """
    Run Gemini Vision API on Type-B test images and evaluate retrieval performance.

    Parameters
    ----------
    max_samples : limit the number of test samples (None = all)
    seed        : random seed used when creating train/val/test splits
    api_delay_s : seconds to wait between API calls to avoid rate limiting
    """
    # ── API key ───────────────────────────────────────────────────────────────
    try:
        from dotenv import load_dotenv
        load_dotenv(_ROOT / '.env')
    except ImportError:
        pass

    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        raise EnvironmentError(
            'GEMINI_API_KEY not found. Add it to a .env file at the project root:\n'
            '  GEMINI_API_KEY=your_key_here'
        )

    from google import genai
    from google.genai import types as genai_types
    client = genai.Client(api_key=api_key)
    # gemini-2.5-flash-lite: lightest available multimodal model on this key
    _MODEL = 'gemini-2.5-flash-lite'
    print(f'[gemini] Model: {_MODEL}')

    # ── SBERT for encoding Gemini output ─────────────────────────────────────
    from sentence_transformers import SentenceTransformer
    sbert = SentenceTransformer('all-MiniLM-L6-v2')
    print('[gemini] SBERT loaded (all-MiniLM-L6-v2)')

    # ── Test data ─────────────────────────────────────────────────────────────
    test_records, all_embeddings, all_sentences = _load_test_records(seed=seed)
    if max_samples:
        test_records = test_records[:max_samples]
    print(f'[gemini] Evaluating {len(test_records)} test images')

    # ── Evaluate ──────────────────────────────────────────────────────────────
    rows:          list[dict] = []
    raw_responses: list[dict] = []
    n_top1 = 0
    n_total = len(test_records)
    pad = len(str(n_total))

    for i, (img_path, true_sentence, n_digits) in enumerate(test_records, 1):
        try:
            gemini_raw, raw_resp = _gemini_describe(client, _MODEL, img_path)
        except Exception:
            # save whatever was collected before the quota/error hit
            if rows:
                print(f'\n[gemini] Interrupted at sample {i} — saving {len(rows)} completed results.')
                _save_results(rows, raw_responses, _build_metrics(rows, _MODEL), PREDICTIONS_B_COMMERCIAL_AI)
            raise
        result = _retrieve(gemini_raw, sbert, all_embeddings, all_sentences, true_sentence)

        n_top1 += result['top_1_correct']
        hit = 'HIT ' if result['top_1_correct'] else 'MISS'

        # per-sample terminal log
        print(
            f'  [{i:{pad}d}/{n_total}] {hit} | '
            f'rank={result["true_rank"]:>5d} | '
            f'cos={result["cosine_sim"]:.4f} | '
            f'true="{true_sentence}"  gemini="{gemini_raw}"'
        )

        rows.append({
            'true_sentence': true_sentence,
            'gemini_raw':    gemini_raw,
            'pred_sentence': result['pred_sentence'],
            'n_digits':      n_digits,
            'true_rank':     result['true_rank'],
            'cosine_sim':    result['cosine_sim'],
            **{k: v for k, v in result.items() if k.startswith('top_') and k.endswith('_correct')},
        })

        # raw response — attach sample metadata for traceability
        raw_responses.append({
            'sample_idx':    i,
            'img_path':      str(img_path),
            'true_sentence': true_sentence,
            'gemini_raw':    gemini_raw,
            'api_response':  raw_resp,
        })

        if api_delay_s > 0:
            time.sleep(api_delay_s)

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    metrics = _build_metrics(rows, _MODEL)

    _save_results(rows, raw_responses, metrics, PREDICTIONS_B_COMMERCIAL_AI)

    print(f'\n[gemini] Results:')
    print(f'  top-1 accuracy : {metrics["top_1_acc"]:.4f}')
    print(f'  top-5 accuracy : {metrics["top_5_acc"]:.4f}')
    print(f'  MRR            : {metrics["mrr"]:.4f}')
    print(f'  mean cosine    : {metrics["mean_cosine_sim"]:.4f}')
    print(f'  mean rank      : {metrics["mean_rank"]:.1f}')
    print(f'  median rank    : {metrics["median_rank"]:.1f}')


# ── CLI ─────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gemini Vision comparison for Type-B')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Limit number of test samples (default: all)')
    parser.add_argument('--seed',        type=int, default=42)
    parser.add_argument('--api-delay',   type=float, default=0.5,
                        help='Seconds between API calls (default: 0.5)')
    args = parser.parse_args()

    run_gemini_comparison(
        max_samples=args.max_samples,
        seed=args.seed,
        api_delay_s=args.api_delay,
    )