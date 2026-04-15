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

# google/gemini-2.0-flash-lite-001: cheapest multimodal model on OpenRouter
# ($0.000000075/image). Change via --model flag.
DEFAULT_MODEL = 'google/gemini-2.0-flash-lite-001'

_OPENROUTER_BASE_URL = 'https://openrouter.ai/api/v1'
_SPLIT_SEED = 42  # must match the seed used during training

# prompt 
_PROMPT = (
    "Describe this image in the following exact format: "
    "'[size] [colour] [number]'\n"
    "Rules:\n"
    "- size must be exactly one of: large, small\n"
    "- colour must be exactly one of: red, blue, green, yellow\n"
    "- number is the digit(s) shown in the image (e.g. 5, 42, 1337)\n"
    "Respond with only the description, nothing else."
)



def _load_test_records() -> tuple[list[tuple[Path, str, int]], torch.Tensor, list[str]]:
    cache_path = EMBED_RESULTS_B / 'tinybert_mean_embedding_result_typeb.pt'
    if not cache_path.exists():
        raise FileNotFoundError(
            f'TinyBERT-mean embedding cache not found: {cache_path}\n'
            'Run: python src/embeddings/computed-embeddings/type-b/'
            'generate_embeddings_type_b.py --embedding tinybert_mean'
        )
    cache = torch.load(cache_path, map_location='cpu')
    all_sentences:  list[str]    = cache['sentences']
    all_embeddings: torch.Tensor = F.normalize(cache['embeddings'].float(), dim=1)

    image_map    = pd.read_csv(TYPE_B_IMAGE_MAP)
    sentences_df = pd.read_csv(TYPE_B_SENTENCES)
    df = image_map.merge(sentences_df, on='sentence_id')

    split_csv = TYPE_B_SPLITS / f'type_b_splits_seed{_SPLIT_SEED}.csv'
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


def _openrouter_describe(
    client,
    model:    str,
    img_path: Path,
    retries:  int = 3,
) -> tuple[str, dict]:
    """
    Send image to OpenRouter and return (lowercased text, raw response as dict).
    Uses the OpenAI chat completions format with base64-encoded image.
    """
    import re
    b64 = _img_to_base64(img_path)
    messages = [
        {
            'role': 'user',
            'content': [
                {'type': 'image_url',
                 'image_url': {'url': f'data:image/png;base64,{b64}'}},
                {'type': 'text', 'text': _PROMPT},
            ],
        }
    ]

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
            )
            text = response.choices[0].message.content.strip().lower()
            raw  = response.model_dump()
            return text, raw
        except Exception as exc:
            wait = 2 ** attempt
            exc_str = str(exc)
            # honour Retry-After if present in 429
            m = re.search(r"retry.after['\"]?\s*[:\s]+(\d+)", exc_str, re.IGNORECASE)
            if m:
                wait = int(m.group(1)) + 2
            if attempt < retries - 1:
                print(f'    [retry {attempt+1}/{retries-1}] waiting {wait}s — {exc_str[:120]}')
                time.sleep(wait)
            else:
                raise


def _retrieve(
    llm_text:        str,
    tinybert_model,
    all_embeddings:  torch.Tensor,   # (N, dim) — already L2-normalised
    all_sentences:   list[str],
    true_sentence:   str,
    top_k:           tuple[int, ...] = (1, 2, 3, 4, 5),
) -> dict:
    vec = torch.tensor(
        tinybert_model.transform([llm_text])
    ).float()
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
    n     = len(rows)
    ranks = np.array([r['true_rank'] for r in rows])
    return {
        'model':           model,
        'embedding':       'tinybert_mean_312d',
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


def _get_processed_paths(pred_path: Path) -> set[str]:
    """Return set of img_path strings already written to the predictions CSV.

    If the file exists but uses the old format (no img_path column), delete it
    so a fresh run starts cleanly rather than appending mismatched columns.
    """
    if not pred_path.exists():
        return set()
    df = pd.read_csv(pred_path)
    if 'img_path' not in df.columns:
        print(f'[openrouter] Old-format predictions file detected — removing and starting fresh: {pred_path.name}')
        pred_path.unlink()
        return set()
    return set(df['img_path'].tolist())


def _append_prediction_row(row: dict, pred_path: Path) -> None:
    """Append one prediction row to CSV; write header only on first call."""
    write_header = not pred_path.exists()
    pd.DataFrame([row]).to_csv(pred_path, mode='a', header=write_header, index=False)


def _append_raw_response(entry: dict, jsonl_path: Path) -> None:
    """Append one raw API response to the JSONL file."""
    with open(jsonl_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')


def _save_summary(metrics: dict, summary_path: Path) -> None:
    """Overwrite the summary CSV with the latest aggregated metrics."""
    pd.DataFrame([metrics]).to_csv(summary_path, index=False)
    print(f'[saved] summary      → {summary_path}')

# major function 
def run_openrouter_comparison(
    model:       str = DEFAULT_MODEL,
    max_samples: int | None = None,
    api_delay_s: float = 0.5,
) -> None:
    """
    Run an OpenRouter vision model on Type-B test images and evaluate retrieval.

    Parameters
    ----------
    model       : OpenRouter model ID (default: google/gemini-2.0-flash-lite-001)
    max_samples : limit the number of test samples (None = all)
    api_delay_s : seconds to wait between API calls
    """
    # load my API key 
    try:
        from dotenv import load_dotenv
        load_dotenv(_ROOT / '.env')
    except ImportError:
        pass

    api_key = os.environ.get('OPEN_ROUTER_API_KEY')
    if not api_key:
        raise EnvironmentError(
            'OPEN_ROUTER_API_KEY not found. Add it to a .env file at the project root:\n'
            '  OPEN_ROUTER_API_KEY=your_key_here'
        )

    from openai import OpenAI
    client = OpenAI(
        base_url=_OPENROUTER_BASE_URL,
        api_key=api_key,
    )
    print(f'[openrouter] Model: {model}')

    from src.embeddings.pretrained.only_type_b_tinybert_mean_embeddings import TinyBertMeanEmbedder
    tinybert = TinyBertMeanEmbedder()
    print('[openrouter] TinyBERT loaded (huawei-noah/TinyBERT_General_4L_312D, mean pooling)')

    # output path 
    PREDICTIONS_B_COMMERCIAL_AI.mkdir(parents=True, exist_ok=True)
    safe_tag     = model.replace('/', '-')
    pred_path    = PREDICTIONS_B_COMMERCIAL_AI / f'openrouter_{safe_tag}_predictions.csv'
    jsonl_path   = PREDICTIONS_B_COMMERCIAL_AI / f'openrouter_{safe_tag}_raw_responses.jsonl'
    summary_path = PREDICTIONS_B_COMMERCIAL_AI / f'openrouter_{safe_tag}_summary.csv'

    test_records, all_embeddings, all_sentences = _load_test_records()
    if max_samples:
        test_records = test_records[:max_samples]
    n_total = len(test_records)
    pad     = len(str(n_total))

    # Resume: skip already-processed images 
    processed = _get_processed_paths(pred_path)
    if processed:
        print(f'[openrouter] Resuming — {len(processed)} already done, {n_total - len(processed)} remaining')
    else:
        print(f'[openrouter] Starting fresh — {n_total} images to process')

    # evaluate one image at a time 
    n_done = len(processed)

    for i, (img_path, true_sentence, n_digits) in enumerate(test_records, 1):
        if str(img_path) in processed:
            continue

        try:
            llm_raw, raw_resp = _openrouter_describe(client, model, img_path)
        except Exception as exc:
            print(f'\n[openrouter] Stopped at sample {i}/{n_total} — {exc}')
            print(f'[openrouter] {n_done} samples saved to {pred_path}')
            raise

        result = _retrieve(llm_raw, tinybert, all_embeddings, all_sentences, true_sentence)
        n_done += 1
        hit = 'HIT ' if result['top_1_correct'] else 'MISS'

        print(
            f'  [{i:{pad}d}/{n_total}] {hit} | '
            f'rank={result["true_rank"]:>5d} | '
            f'cos={result["cosine_sim"]:.4f} | '
            f'true="{true_sentence}"  llm="{llm_raw}"'
        )

        # Write this sample immediately so a crash loses at most one result
        row = {
            'image_id':      img_path.name,
            'img_path':      str(img_path),
            'true_sentence': true_sentence,
            'llm_raw':       llm_raw,
            'pred_sentence': result['pred_sentence'],
            'n_digits':      n_digits,
            'true_rank':     result['true_rank'],
            'cosine_sim':    result['cosine_sim'],
            **{k: v for k, v in result.items() if k.startswith('top_') and k.endswith('_correct')},
        }
        _append_prediction_row(row, pred_path)
        _append_raw_response({
            'sample_idx':    i,
            'img_path':      str(img_path),
            'true_sentence': true_sentence,
            'llm_raw':       llm_raw,
            'api_response':  raw_resp,
        }, jsonl_path)

        if api_delay_s > 0:
            time.sleep(api_delay_s)

    # Final metrics from the full predictions CSV 
    all_rows = pd.read_csv(pred_path).to_dict('records')
    metrics  = _build_metrics(all_rows, model)
    _save_summary(metrics, summary_path)

    print(f'\n[openrouter] Results ({len(all_rows)} samples):')
    print(f'  top-1 accuracy : {metrics["top_1_acc"]:.4f}')
    print(f'  top-5 accuracy : {metrics["top_5_acc"]:.4f}')
    print(f'  MRR            : {metrics["mrr"]:.4f}')
    print(f'  mean cosine    : {metrics["mean_cosine_sim"]:.4f}')
    print(f'  mean rank      : {metrics["mean_rank"]:.1f}')
    print(f'  median rank    : {metrics["median_rank"]:.1f}')

# run 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OpenRouter vision LLM comparison for Type-B')
    parser.add_argument('--model',       default=DEFAULT_MODEL,
                        help=f'OpenRouter model ID (default: {DEFAULT_MODEL})')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Limit number of test samples (default: all)')
    parser.add_argument('--api-delay',   type=float, default=0.5,
                        help='Seconds between API calls (default: 0.5)')
    args = parser.parse_args()

    run_openrouter_comparison(
        model=args.model,
        max_samples=args.max_samples,
        api_delay_s=args.api_delay,
    )