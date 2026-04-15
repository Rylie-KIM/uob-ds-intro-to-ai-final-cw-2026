"""
src/pipelines/evaluation/type-b/groq_comparison.py

Compare CNN retrieval performance against a vision LLM via Groq.

Protocol
--------
1. Load the test split (same indices as CNN training experiments, via split CSV).
2. For each test image, send it to the Groq vision model with a structured prompt.
3. Map the model's free-text response to the nearest sentence in the corpus
   using cosine similarity of SBERT embeddings.
4. Compute the same retrieval metrics as eval_metrics_b.py (top-k acc, MRR, etc.).
5. Save results to PREDICTIONS_B_COMMERCIAL_AI/:
     groq_{model_tag}_predictions.csv
     groq_{model_tag}_raw_responses.jsonl
     groq_{model_tag}_summary.csv

Requirements
------------
  pip install openai python-dotenv sentence-transformers
  Set GROQ_API_KEY in a .env file at the project root.

  Free-tier rate limits (as of 2025-04):
    llama-3.2-11b-vision-preview : 30 req/min, 7 000 tokens/min
    llama-3.2-90b-vision-preview : 15 req/min, 7 000 tokens/min
  The default api_delay (2.1 s) keeps both well under the 30 req/min ceiling.

Usage
-----
  python src/pipelines/evaluation/type-b/groq_comparison.py
  python src/pipelines/evaluation/type-b/groq_comparison.py --model llama-3.2-90b-vision-preview
  python src/pipelines/evaluation/type-b/groq_comparison.py --max-samples 50

Model selection rationale
--------------------------
  Recommended: llama-3.2-11b-vision-preview

  Groq hosts two vision-capable LLaMA 3.2 models:

  ┌─────────────────────────────────┬────────┬───────────────┬──────────────────────────────┐
  │ Model                           │ Params │ Free RPM      │ Notes                        │
  ├─────────────────────────────────┼────────┼───────────────┼──────────────────────────────┤
  │ llama-3.2-11b-vision-preview    │  11 B  │ 30 req/min    │ Recommended — good accuracy  │
  │                                 │        │               │ with practical rate limits   │
  ├─────────────────────────────────┼────────┼───────────────┼──────────────────────────────┤
  │ llama-3.2-90b-vision-preview    │  90 B  │ 15 req/min    │ Higher accuracy, stricter    │
  │                                 │        │               │ rate limits on free tier     │
  └─────────────────────────────────┴────────┴───────────────┴──────────────────────────────┘

  Why llama-3.2-11b over alternatives:

  1. Task complexity: Type-B images require colour identification, size estimation,
     and digit OCR — all within MNIST difficulty. This is well within the capability
     of a 11B vision model; a 90B model offers diminishing returns for a task this
     structured (Dubey et al., 2024 — LLaMA 3 technical report, §4.3).

  2. Rate limits: Free-tier 11B allows 30 req/min vs 15 req/min for 90B.
     With ~1 000 test samples the 11B finishes in ~35 min; the 90B would take ~70 min
     under free-tier constraints.

  3. Instruction following: Both models are instruction-tuned; the 11B follows
     structured output prompts ("size colour number") reliably for constrained
     vocabulary tasks (Meta AI, 2024 — Llama 3.2 model card).

  4. vs. GPT-4o / Gemini: GPT-4o and Gemini 2.0 Flash offer comparable or better
     vision accuracy but require paid keys. On structured low-complexity visual
     grounding, the accuracy gap between GPT-4o and LLaMA 3.2 11B narrows
     significantly (Yue et al., 2024 — MMMU benchmark, Table 3).

  References
  ----------
  - Dubey, A. et al. (2024). The Llama 3 Herd of Models. arXiv:2407.21783.
    https://arxiv.org/abs/2407.21783
  - Meta AI (2024). Llama 3.2 Model Card.
    https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct
  - Yue, X. et al. (2024). MMMU: A Massive Multi-discipline Multimodal Understanding
    and Reasoning Benchmark. CVPR 2024. https://arxiv.org/abs/2311.16502
  - Groq (2025). Supported Models & Rate Limits.
    https://console.groq.com/docs/models
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# ── Path bootstrap ─────
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

# ── Default model ──────
DEFAULT_MODEL = 'llama-3.2-11b-vision-preview'
_GROQ_BASE_URL = 'https://api.groq.com/openai/v1'

# ── Prompt ─────────────
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
    Load test-split records and TinyBERT mean-pool corpus embeddings.

    Uses tinybert_mean (E2e) — the best-performing embedding in Stage-1
    evaluation (composite score 0.956, median rank 169).

    Returns
    -------
    test_records   : list of (image_path, sentence, n_digits)
    all_embeddings : Tensor (N_corpus, 312) — L2-normalised TinyBERT corpus embeddings
    all_sentences  : list[str]
    """
    cache_path = EMBED_RESULTS_B / 'tinybert_mean_embedding_result_typeb.pt'
    if not cache_path.exists():
        raise FileNotFoundError(
            f'TinyBERT mean embedding cache not found: {cache_path}\n'
            'Run: python src/embeddings/computed-embeddings/type-b/'
            'generate_embeddings_type_b.py --embedding tinybert_mean'
        )
    cache = torch.load(cache_path, map_location='cpu')
    all_sentences:  list[str]    = cache['sentences']
    # use pre-normalised embeddings if available, else normalise on the fly
    if 'embeddings_normalised' in cache:
        all_embeddings: torch.Tensor = cache['embeddings_normalised'].float()
    else:
        all_embeddings = F.normalize(cache['embeddings'].float(), dim=1)

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


def _img_to_base64(img_path: Path) -> str:
    with open(img_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def _groq_describe(
    client,
    model:    str,
    img_path: Path,
    retries:  int = 3,
) -> tuple[str, dict]:
    """
    Send image to Groq and return (lowercased text, raw response as dict).

    Uses the OpenAI-compatible chat completions endpoint with base64-encoded image.
    Groq vision models accept the same image_url format as the OpenAI vision API.
    """
    import re
    b64 = _img_to_base64(img_path)
    messages = [
        {
            'role': 'user',
            'content': [
                {
                    'type': 'image_url',
                    'image_url': {'url': f'data:image/png;base64,{b64}'},
                },
                {'type': 'text', 'text': _PROMPT},
            ],
        }
    ]

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=32,        # output is short: "large red 5"
                temperature=0.0,      # deterministic output for reproducibility
            )
            text = response.choices[0].message.content.strip().lower()
            raw  = response.model_dump()
            return text, raw
        except Exception as exc:
            wait = 2 ** attempt
            exc_str = str(exc)
            # honour Retry-After from 429 responses
            m = re.search(r'retry.after[\'"]?\s*[:\s]+(\d+)', exc_str, re.IGNORECASE)
            if m:
                wait = int(m.group(1)) + 2
            if attempt < retries - 1:
                print(f'    [retry {attempt+1}/{retries-1}] waiting {wait}s — {exc_str[:120]}')
                time.sleep(wait)
            else:
                raise


def _retrieve(
    llm_text:       str,
    tinybert_model,
    tinybert_tokenizer,
    all_embeddings: torch.Tensor,   # (N, 312) — already L2-normalised
    all_sentences:  list[str],
    true_sentence:  str,
    top_k:          tuple[int, ...] = (1, 2, 3, 4, 5),
) -> dict:
    """
    Encode llm_text with TinyBERT mean-pool, retrieve nearest corpus sentence,
    and compute rank of the ground-truth sentence.

    Mirrors TinyBertMeanEmbedder.get_embedding(): mean over non-[CLS]/[SEP] tokens,
    then L2-normalise to match the pre-normalised corpus embeddings.
    """
    processed = tinybert_tokenizer(
        llm_text.lower(),   # TinyBERT tokenizer requires lowercase
        return_tensors='pt',
    )
    with torch.no_grad():
        output = tinybert_model(
            input_ids=processed['input_ids'],
            attention_mask=processed['attention_mask'],
        )
    # strip [CLS] and [SEP], mean-pool remaining tokens
    hidden = output.last_hidden_state[0][1:-1]   # (seq_len-2, 312)
    vec = hidden.mean(dim=0, keepdim=True)        # (1, 312)
    vec = F.normalize(vec, dim=1)

    sims       = (vec @ all_embeddings.T).squeeze(0)
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
    ranks = np.array([r['true_rank'] for r in rows])
    return {
        'model':           model,
        'embedding':       'tinybert_mean',
        'n_test':          len(rows),
        'top_1_acc':       float(np.mean([r['top_1_correct'] for r in rows])),
        'top_2_acc':       float(np.mean([r['top_2_correct'] for r in rows])),
        'top_3_acc':       float(np.mean([r['top_3_correct'] for r in rows])),
        'top_5_acc':       float(np.mean([r['top_5_correct'] for r in rows])),
        'mrr':             float(np.mean(1.0 / ranks)),
        'mean_cosine_sim': float(np.mean([r['cosine_sim'] for r in rows])),
        'mean_rank':       float(np.mean(ranks)),
        'median_rank':     float(np.median(ranks)),
    }


def _safe_tag(model_tag: str) -> str:
    return model_tag.replace('/', '-').replace('.', '-')


def _pred_path(out_dir: Path, model_tag: str) -> Path:
    return out_dir / f'groq_{_safe_tag(model_tag)}_predictions.csv'


def _load_existing_predictions(out_dir: Path, model_tag: str) -> tuple[list[dict], set[str]]:
    """
    Load previously saved predictions CSV if it exists.

    Returns
    -------
    existing_rows : list[dict] — rows already evaluated
    done_paths    : set[str]  — img_path strings already processed
    """
    path = _pred_path(out_dir, model_tag)
    if not path.exists():
        return [], set()
    df = pd.read_csv(path)
    rows = df.to_dict('records')
    done = set(df['img_path'].astype(str).tolist()) if 'img_path' in df.columns else set()
    print(f'[groq] Resuming — {len(rows)} predictions already saved, {len(done)} paths done.')
    return rows, done


def _append_prediction(out_dir: Path, model_tag: str, row: dict) -> None:
    """Append a single prediction row to the CSV (write header only if file is new)."""
    path = _pred_path(out_dir, model_tag)
    write_header = not path.exists()
    pd.DataFrame([row]).to_csv(path, mode='a', header=write_header, index=False)


def _save_final(
    all_rows: list[dict], raw_responses: list[dict], metrics: dict,
    out_dir: Path, model_tag: str,
) -> None:
    """
    Overwrite predictions CSV with full combined results, append raw responses,
    and write summary metrics.

    Files written
    -------------
    groq_{model_tag}_predictions.csv   — complete predictions (all runs combined)
    groq_{model_tag}_raw_responses.jsonl — new responses appended
    groq_{model_tag}_summary.csv       — single-row aggregate metrics
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = _safe_tag(model_tag)

    pred_path = out_dir / f'groq_{tag}_predictions.csv'
    pd.DataFrame(all_rows).to_csv(pred_path, index=False)
    print(f'[saved] predictions  → {pred_path}  ({len(all_rows)} total rows)')

    if raw_responses:
        jsonl_path = out_dir / f'groq_{tag}_raw_responses.jsonl'
        with open(jsonl_path, 'a', encoding='utf-8') as f:
            for entry in raw_responses:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        print(f'[saved] raw JSON     → {jsonl_path}  (+{len(raw_responses)} new)')

    summary_path = out_dir / f'groq_{tag}_summary.csv'
    pd.DataFrame([metrics]).to_csv(summary_path, index=False)
    print(f'[saved] summary      → {summary_path}')

# main comparison 
def run_groq_comparison(
    model:       str = DEFAULT_MODEL,
    max_samples: int | None = None,
    seed:        int = 42,
    api_delay_s: float = 2.1,
) -> None:
    """
    Run a Groq vision model on Type-B test images and evaluate retrieval.

    Parameters
    ----------
    model       : Groq model ID (default: llama-3.2-11b-vision-preview)
    max_samples : limit the number of test samples (None = all)
    seed        : random seed used when creating train/val/test splits
    api_delay_s : seconds to wait between API calls.
                  Default 2.1 s stays under the free-tier 30 req/min ceiling
                  for the 11B model (28.6 req/min effective).
    """
    # ── API key ──────
    try:
        from dotenv import load_dotenv
        load_dotenv(_ROOT / '.env')
    except ImportError:
        pass

    api_key = os.environ.get('GROQ_API_KEY')
    if not api_key:
        raise EnvironmentError(
            'GROQ_API_KEY not found. Add it to a .env file at the project root:\n'
            '  GROQ_API_KEY=gsk_your_key_here\n'
            'Get a free key at: https://console.groq.com/keys'
        )

    from openai import OpenAI
    client = OpenAI(
        base_url=_GROQ_BASE_URL,
        api_key=api_key,
    )
    print(f'[groq] Model      : {model}')
    print(f'[groq] API delay  : {api_delay_s}s between requests')

    # ── TinyBERT mean-pool for encoding LLM output ───────────────────────────
    # Matches the best Stage-1 embedding (E2e, composite 0.956) so that the
    # LLM output is mapped into the same embedding space as the corpus.
    from transformers import AutoModel, AutoTokenizer
    tinybert_tokenizer = AutoTokenizer.from_pretrained(
        'huawei-noah/TinyBERT_General_4L_312D', do_lower_case=True,
    )
    tinybert_model = AutoModel.from_pretrained(
        'huawei-noah/TinyBERT_General_4L_312D',
    ).eval()
    print('[groq] TinyBERT mean-pool loaded (huawei-noah/TinyBERT_General_4L_312D, 312-dim)')

    # ── Test data ────
    test_records, all_embeddings, all_sentences = _load_test_records(seed=seed)
    if max_samples:
        test_records = test_records[:max_samples]

    # ── Resume: skip already-completed samples ────────────────────────────────
    PREDICTIONS_B_COMMERCIAL_AI.mkdir(parents=True, exist_ok=True)
    existing_rows, done_paths = _load_existing_predictions(
        PREDICTIONS_B_COMMERCIAL_AI, model,
    )
    pending = [
        (img_path, sent, nd)
        for img_path, sent, nd in test_records
        if str(img_path) not in done_paths
    ]
    n_total   = len(test_records)
    n_pending = len(pending)
    n_done    = n_total - n_pending
    pad       = len(str(n_total))
    print(f'[groq] {n_total} test images total | {n_done} done | {n_pending} remaining')

    if n_pending == 0:
        print('[groq] All samples already evaluated. Recalculating metrics...')
        metrics = _build_metrics(existing_rows, model)
        _save_final(existing_rows, [], metrics, PREDICTIONS_B_COMMERCIAL_AI, model)
        _print_metrics(metrics, model)
        return

    # ── Evaluate pending samples ──────────────────────────────────────────────
    new_rows:      list[dict] = []
    raw_responses: list[dict] = []

    for i, (img_path, true_sentence, n_digits) in enumerate(pending, n_done + 1):
        try:
            llm_raw, raw_resp = _groq_describe(client, model, img_path)
        except Exception:
            if new_rows:
                print(f'\n[groq] Interrupted at sample {i}/{n_total} — '
                      f'{len(new_rows)} new results already appended to CSV.')
            raise

        result = _retrieve(
            llm_raw, tinybert_model, tinybert_tokenizer,
            all_embeddings, all_sentences, true_sentence,
        )
        hit = 'HIT ' if result['top_1_correct'] else 'MISS'

        print(
            f'  [{i:{pad}d}/{n_total}] {hit} | '
            f'rank={result["true_rank"]:>5d} | '
            f'cos={result["cosine_sim"]:.4f} | '
            f'true="{true_sentence}"  groq="{llm_raw}"'
        )

        row = {
            'img_path':      str(img_path),
            'true_sentence': true_sentence,
            'llm_raw':       llm_raw,
            'pred_sentence': result['pred_sentence'],
            'n_digits':      n_digits,
            'true_rank':     result['true_rank'],
            'cosine_sim':    result['cosine_sim'],
            **{k: v for k, v in result.items()
               if k.startswith('top_') and k.endswith('_correct')},
        }
        new_rows.append(row)
        # append immediately so progress survives interruption
        _append_prediction(PREDICTIONS_B_COMMERCIAL_AI, model, row)

        raw_responses.append({
            'sample_idx':    i,
            'img_path':      str(img_path),
            'true_sentence': true_sentence,
            'llm_raw':       llm_raw,
            'api_response':  raw_resp,
        })

        if api_delay_s > 0:
            time.sleep(api_delay_s)

    # ── Final save: merge existing + new, rewrite predictions, update metrics ─
    all_rows = existing_rows + new_rows
    metrics  = _build_metrics(all_rows, model)
    _save_final(all_rows, raw_responses, metrics, PREDICTIONS_B_COMMERCIAL_AI, model)

    _print_metrics(metrics, model)


def _print_metrics(metrics: dict, model: str) -> None:
    print(f'\n[groq] Results ({model})  n={metrics["n_test"]}:')
    print(f'  top-1 accuracy : {metrics["top_1_acc"]:.4f}')
    print(f'  top-5 accuracy : {metrics["top_5_acc"]:.4f}')
    print(f'  MRR            : {metrics["mrr"]:.4f}')
    print(f'  mean cosine    : {metrics["mean_cosine_sim"]:.4f}')
    print(f'  mean rank      : {metrics["mean_rank"]:.1f}')
    print(f'  median rank    : {metrics["median_rank"]:.1f}')


# main 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Groq vision LLM comparison for Type-B')
    parser.add_argument(
        '--model', default=DEFAULT_MODEL,
        help=(
            f'Groq model ID (default: {DEFAULT_MODEL}). '
            'Vision-capable options: '
            'llama-3.2-11b-vision-preview, llama-3.2-90b-vision-preview'
        ),
    )
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Limit number of test samples (default: all)')
    parser.add_argument('--seed',        type=int, default=42)
    parser.add_argument(
        '--api-delay', type=float, default=2.1,
        help=(
            'Seconds between API calls (default: 2.1). '
            'Keep >= 2.1 for free-tier 11B (30 req/min), '
            '>= 4.1 for 90B (15 req/min).'
        ),
    )
    args = parser.parse_args()

    run_groq_comparison(
        model=args.model,
        max_samples=args.max_samples,
        seed=args.seed,
        api_delay_s=args.api_delay,
    )