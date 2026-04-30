from __future__ import annotations

import argparse
import base64
import concurrent.futures as cf
import hashlib
import json
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


# Project root


_ROOT = next(p for p in Path(__file__).resolve().parents if (p / ".git").exists())
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# Type-C paths


PROJECT_ROOT = Path(r"D:\Users\git-repo\1")

TYPE_C_IMAGE_MAP = PROJECT_ROOT / "src" / "data" / "type-c" / "image_map_c.csv"
TYPE_C_SENTENCES = PROJECT_ROOT / "src" / "data" / "type-c" / "sentences_c.csv"
TYPE_C_IMAGES = PROJECT_ROOT / "src" / "data" / "images" / "type-c"

# Use the same test split as the main CNN/GloVe experiment.
TYPE_C_SPLIT_MANIFEST = PROJECT_ROOT / "results22" / "cnn3" / "Glove" / "split_manifest_glove.csv"

TYPE_C_EMBED_DIR = PROJECT_ROOT / "src" / "data" / "type-c"
OUTPUT_DIR = PROJECT_ROOT / "results22" / "LLM_TYPEC_OPENROUTER"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# OpenRouter config


DEFAULT_MODEL = "google/gemini-2.0-flash-lite-001"
DEFAULT_EMBEDDING = "sbert"
_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


# Type-C prompt

_PROMPT_DETAILED = (
    "Describe this tic-tac-toe board in the following exact format:\n"
    "'X is in [positions]. O is in [positions].'\n\n"
    "Rules:\n"
    "- Only mention occupied cells.\n"
    "- Use only these position names: top left, top center, top right, "
    "middle left, center, middle right, bottom left, bottom center, bottom right.\n"
    "- Describe X first, then O.\n"
    "- If a player has multiple positions, connect positions with 'and'.\n"
    "- If a player has no marks, omit that player.\n"
    "- Respond with only the description, nothing else."
)

_PROMPT_MINIMAL = "Describe the tic-tac-toe board. Mention the positions of X and O only."


# Helpers


def _normalise_prompt_tag(prompt_tag: str | None) -> str:
    if prompt_tag is None:
        return ""
    tag = str(prompt_tag).strip()
    if tag.lower() in ("", "none", "null"):
        return ""
    return tag


def _build_stem(model: str, embedding_name: str | None = None, prompt_tag: str | None = None) -> str:
    safe_model = model.replace("/", "-").replace(":", "-")
    parts = ["openrouter_typec"]
    if embedding_name:
        parts.append(embedding_name)
    tag = _normalise_prompt_tag(prompt_tag)
    if tag:
        parts.append(tag)
    parts.append(safe_model)
    return "_".join(parts)


def _img_to_base64(img_path: Path) -> str:
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _cache_key(model: str, prompt_tag: str, img_path: Path) -> str:
    raw = f"{model}|{prompt_tag}|{str(img_path)}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _canonicalise_typec_output(text: str) -> str:
    x = str(text).strip().lower()
    x = x.replace("\n", " ")
    x = re.sub(r"\s+", " ", x)

    replacements = {
        "centre": "center",
        "middle centre": "center",
        "middle center": "center",
        "upper left": "top left",
        "upper centre": "top center",
        "upper center": "top center",
        "upper right": "top right",
        "lower left": "bottom left",
        "lower centre": "bottom center",
        "lower center": "bottom center",
        "lower right": "bottom right",
        "mid left": "middle left",
        "mid right": "middle right",
        "located in": "in",
        "located at": "in",
        "placed in": "in",
        "placed at": "in",
        "has gone in": "is in",
        "is at": "is in",
        "x's": "x",
        "o's": "o",
    }
    for a, b in replacements.items():
        x = x.replace(a, b)

    fillers = [
        "the board shows",
        "the board contains",
        "the board has",
        "on the board",
        "in the image",
        "the image shows",
        "there is",
        "there are",
    ]
    for f in fillers:
        x = x.replace(f, "")

    x = re.sub(r"\s+", " ", x).strip()
    x = re.sub(r"\bx\b", "X", x)
    x = re.sub(r"\bo\b", "O", x)

    if x and not x.endswith("."):
        x += "."
    return x


def _resolve_image_path(filename: str) -> Path:
    fname = str(filename)
    candidates = [TYPE_C_IMAGES / fname, TYPE_C_IMAGES / f"{fname}.png"]
    if fname.startswith("c_"):
        candidates += [
            TYPE_C_IMAGES / f"type_{fname}.png",
            TYPE_C_IMAGES / f"type_c_{fname[2:]}.png",
            TYPE_C_IMAGES / f"type_c_{fname}.png",
        ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("Image not found. Tried:\n" + "\n".join(str(p) for p in candidates))


# Load test records


def _load_test_records(max_samples: int | None = None) -> list[dict]:
    image_map = pd.read_csv(TYPE_C_IMAGE_MAP)
    sentences_df = pd.read_csv(TYPE_C_SENTENCES)

    image_map["sentence_id"] = image_map["sentence_id"].astype(str)
    sentences_df["sentence_id"] = sentences_df["sentence_id"].astype(str)

    df = image_map.merge(sentences_df, on="sentence_id", how="left")

    if not TYPE_C_SPLIT_MANIFEST.exists():
        raise FileNotFoundError(f"Split manifest not found: {TYPE_C_SPLIT_MANIFEST}")

    split_df = pd.read_csv(TYPE_C_SPLIT_MANIFEST)
    split_df["sentence_id"] = split_df["sentence_id"].astype(str)
    test_sids = split_df[split_df["split"] == "test"]["sentence_id"].tolist()
    order = {sid: i for i, sid in enumerate(test_sids)}

    df = df[df["sentence_id"].isin(test_sids)].copy()
    df["order"] = df["sentence_id"].map(order)
    df = df.sort_values("order").reset_index(drop=True)

    if max_samples is not None:
        df = df.head(max_samples).copy()

    records = []
    for _, row in df.iterrows():
        img_path = _resolve_image_path(row["filename"])
        records.append({
            "img_path": img_path,
            "filename": str(row["filename"]),
            "sentence_id": str(row["sentence_id"]),
            "true_sentence": str(row["sentence"]),
            "notation": str(row.get("notation", "")),
            "n_moves": int(row["n_moves"]) if "n_moves" in row and not pd.isna(row["n_moves"]) else -1,
        })
    return records

# Text embedding corpus


def _load_embedding_corpus(embedding_name: str) -> tuple[torch.Tensor, list[str]]:
    embedding_name = embedding_name.lower()
    sentences_df = pd.read_csv(TYPE_C_SENTENCES)
    all_sentences = sentences_df["sentence"].astype(str).tolist()

    pt_candidates = [
        TYPE_C_EMBED_DIR / f"type_c_{embedding_name}.pt",
        TYPE_C_EMBED_DIR / f"{embedding_name}.pt",
    ]
    for p in pt_candidates:
        if p.exists():
            obj = torch.load(p, map_location="cpu")
            if isinstance(obj, dict) and "embeddings" in obj:
                emb = obj["embeddings"].float()
            elif isinstance(obj, torch.Tensor):
                emb = obj.float()
            else:
                raise ValueError(f"Unsupported PT format: {p}")
            return emb, all_sentences

    if embedding_name == "sbert":
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        emb = model.encode(all_sentences, convert_to_tensor=True).float()
        return emb, all_sentences

    if embedding_name == "tfidf":
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer()
        arr = vectorizer.fit_transform(all_sentences).toarray()
        emb = torch.tensor(arr).float()
        return emb, all_sentences

    raise FileNotFoundError(
        f"No embedding cache found for {embedding_name}. Tried:\n"
        + "\n".join(str(p) for p in pt_candidates)
    )


def _load_text_embedder(embedding_name: str):
    embedding_name = embedding_name.lower()

    if embedding_name == "sbert":
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        def encode(texts: list[str]) -> torch.Tensor:
            return model.encode(texts, convert_to_tensor=True).float()
        print("[openrouter-typec] SBERT loaded")
        return encode

    if embedding_name == "tfidf":
        from sklearn.feature_extraction.text import TfidfVectorizer
        corpus = pd.read_csv(TYPE_C_SENTENCES)["sentence"].astype(str).tolist()
        vectorizer = TfidfVectorizer()
        vectorizer.fit(corpus)
        def encode(texts: list[str]) -> torch.Tensor:
            arr = vectorizer.transform(texts).toarray()
            return torch.tensor(arr).float()
        print("[openrouter-typec] TF-IDF loaded")
        return encode

    raise ValueError(f"Unsupported embedding: {embedding_name}. Use sbert or tfidf.")


# OpenRouter call 

def _openrouter_describe(client, model: str, img_path: Path, prompt: str, retries: int = 5) -> tuple[str, dict]:
    b64 = _img_to_base64(img_path)
    messages = [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": prompt},
        ],
    }]

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(model=model, messages=messages)
            text = response.choices[0].message.content.strip()
            return text, response.model_dump()
        except Exception as exc:
            wait = min(60, 2 ** attempt)
            exc_str = str(exc)
            m = re.search(r"retry.after['\"]?\s*[:\s]+(\d+)", exc_str, re.IGNORECASE)
            if m:
                wait = min(120, int(m.group(1)) + 2)
            if attempt < retries - 1:
                print(f"    [retry {attempt + 1}/{retries - 1}] waiting {wait}s — {exc_str[:160]}")
                time.sleep(wait)
            else:
                raise


def _load_cache(cache_path: Path) -> dict[str, dict]:
    cache = {}
    if not cache_path.exists():
        return cache
    with open(cache_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if obj.get("cache_key") and obj.get("ok"):
                    cache[obj["cache_key"]] = obj
            except json.JSONDecodeError:
                continue
    return cache


def _append_jsonl_threadsafe(path: Path, obj: dict, lock: threading.Lock) -> None:
    with lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# Retrieval


def _retrieve(
    llm_text: str,
    query_encode_fn: Callable[[list[str]], torch.Tensor],
    all_embeddings: torch.Tensor,
    all_sentences: list[str],
    true_sentence: str,
    top_k: tuple[int, ...] = (1, 2, 3, 4, 5),
) -> dict:
    vec = query_encode_fn([llm_text]).float()
    vec_norm = F.normalize(vec, dim=1)
    emb_for_sim = F.normalize(all_embeddings.float(), dim=1)
    sims = F.cosine_similarity(vec_norm, emb_for_sim, dim=1)
    sorted_idx = sims.argsort(descending=True).tolist()

    best_idx = sorted_idx[0]
    pred_sentence = all_sentences[best_idx]
    cosine_sim = float(sims[best_idx].item())

    try:
        true_idx = all_sentences.index(true_sentence)
    except ValueError:
        raise ValueError(f"True sentence not found in corpus: {true_sentence}")

    rank = sorted_idx.index(true_idx) + 1
    return {
        "pred_sentence": pred_sentence,
        "cosine_sim": cosine_sim,
        "true_rank": rank,
        **{f"top_{k}_correct": int(rank <= k) for k in top_k},
    }


def _build_metrics(rows: list[dict], model: str, embedding_name: str) -> dict:
    ranks = np.array([r["true_rank"] for r in rows], dtype=float)
    return {
        "run_id": f"LLM-openrouter-{embedding_name}",
        "model": model,
        "embedding": embedding_name,
        "n_test": len(rows),
        "top_1_acc": float(np.mean([r["top_1_correct"] for r in rows])),
        "top_2_acc": float(np.mean([r["top_2_correct"] for r in rows])),
        "top_3_acc": float(np.mean([r["top_3_correct"] for r in rows])),
        "top_5_acc": float(np.mean([r["top_5_correct"] for r in rows])),
        "mrr": float(np.mean(1.0 / ranks)),
        "mean_cosine_sim": float(np.mean([r["cosine_sim"] for r in rows])),
        "mean_rank": float(np.mean(ranks)),
        "median_rank": float(np.median(ranks)),
    }


def _build_typec_test_results(metrics: dict, embedding_name: str) -> pd.DataFrame:
    dim = 384 if embedding_name == "sbert" else np.nan
    return pd.DataFrame([{
        "run_id": metrics["run_id"],
        "cnn": "llm_openrouter",
        "embedding": embedding_name,
        "raw_text_dim": dim,
        "shared_dim": dim,
        "loss_fn": "none_llm_generation",
        "best_epoch": 0,
        "total_epochs": 0,
        "test_mse_loss": np.nan,
        "test_mean_cosine": metrics["mean_cosine_sim"],
        "test_top1": metrics["top_1_acc"],
        "test_top5": metrics["top_5_acc"],
        "test_mean_rank": metrics["mean_rank"],
        "test_median_rank": metrics["median_rank"],
        "test_mrr": metrics["mrr"],
    }])


# CSV utilities


def _read_done_ids(pred_path: Path, fail_path: Path, retry_failed: bool) -> set[str]:
    done = set()
    if pred_path.exists():
        df = pd.read_csv(pred_path)
        if "sentence_id" in df.columns:
            done.update(df["sentence_id"].astype(str).tolist())
    if fail_path.exists() and not retry_failed:
        df = pd.read_csv(fail_path)
        if "sentence_id" in df.columns:
            done.update(df["sentence_id"].astype(str).tolist())
    return done


def _append_csv_row(row: dict, path: Path) -> None:
    write_header = not path.exists()
    pd.DataFrame([row]).to_csv(path, mode="a", header=write_header, index=False)


def _save_final_outputs(pred_path: Path, summary_path: Path, typec_result_path: Path, by_moves_path: Path, model: str, embedding_name: str) -> dict | None:
    if not pred_path.exists():
        print("[openrouter-typec] No successful predictions yet; final metrics not saved.")
        return None

    pred_df = pd.read_csv(pred_path)
    if pred_df.empty:
        print("[openrouter-typec] Empty predictions; final metrics not saved.")
        return None

    rows = pred_df.to_dict("records")
    metrics = _build_metrics(rows, model, embedding_name)
    pd.DataFrame([metrics]).to_csv(summary_path, index=False)
    _build_typec_test_results(metrics, embedding_name).to_csv(typec_result_path, index=False)

    by_rows = []
    for n_moves, sub in pred_df.groupby("n_moves"):
        ranks = sub["true_rank"].astype(float).to_numpy()
        by_rows.append({
            "run_id": f"LLM-openrouter-{embedding_name}",
            "cnn": "llm_openrouter",
            "embedding": embedding_name,
            "n_moves": int(n_moves),
            "count": len(sub),
            "top1": float(np.mean(ranks <= 1)),
            "top5": float(np.mean(ranks <= 5)),
            "mean_rank": float(np.mean(ranks)),
            "median_rank": float(np.median(ranks)),
            "mrr": float(np.mean(1.0 / ranks)),
            "mean_cosine": float(sub["cosine_sim"].mean()),
        })
    pd.DataFrame(by_rows).sort_values("n_moves").to_csv(by_moves_path, index=False)
    return metrics


def build_comparison_table(results_root: Path, out_path: Path) -> pd.DataFrame:
    files = sorted(results_root.rglob("test_results.csv"))
    frames = []
    for p in files:
        try:
            df = pd.read_csv(p)
            if df.empty:
                continue
            df = df.copy()
            df["source_file"] = str(p)
            frames.append(df)
        except Exception as exc:
            print(f"[compare] skipped {p}: {exc}")

    if not frames:
        empty = pd.DataFrame()
        empty.to_csv(out_path, index=False)
        return empty

    all_df = pd.concat(frames, ignore_index=True, sort=False)

    wanted = [
        "run_id", "cnn", "embedding", "raw_text_dim", "shared_dim", "loss_fn",
        "best_epoch", "test_mse_loss", "test_mean_cosine", "test_top1", "test_top5",
        "test_mean_rank", "test_median_rank", "test_mrr", "source_file",
    ]
    for c in wanted:
        if c not in all_df.columns:
            all_df[c] = np.nan
    all_df = all_df[wanted]

    # Sort: higher MRR / Top-1 / Top-5 better, lower median/mean rank better.
    all_df = all_df.sort_values(
        by=["test_mrr", "test_top1", "test_top5", "test_median_rank", "test_mean_rank"],
        ascending=[False, False, False, True, True],
        na_position="last",
    )
    all_df.to_csv(out_path, index=False)
    return all_df


# Main run


def run_openrouter_typec_comparison(
    model: str = DEFAULT_MODEL,
    embedding_name: str = DEFAULT_EMBEDDING,
    prompt_tag: str = "detailed",
    max_samples: int | None = None,
    api_delay_s: float = 0.0,
    workers: int = 6,
    retries: int = 5,
    retry_failed: bool = False,
    make_compare: bool = True,
) -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(_ROOT / ".env")
    except ImportError:
        pass

    api_key = os.environ.get("OPEN_ROUTER_API_KEY")
    print("API key loaded:", bool(api_key), api_key[:10] if api_key else None)
    if not api_key:
        raise EnvironmentError(
            "OPEN_ROUTER_API_KEY not found. Add it to .env at project root:\n"
            "OPEN_ROUTER_API_KEY=your_key_here"
        )

    from openai import OpenAI

    client = OpenAI(
        base_url=_OPENROUTER_BASE_URL,
        api_key=api_key.strip(),
        default_headers={
            "Authorization": f"Bearer {api_key.strip()}",
            "HTTP-Referer": "http://localhost",
            "X-Title": "Type-C Project",
        },
    )

    print(f"[openrouter-typec] Model: {model}")
    print(f"[openrouter-typec] Embedding: {embedding_name}")
    print(f"[openrouter-typec] Workers: {workers}")
    print(f"[openrouter-typec] Max samples: {'FULL TEST SET' if max_samples is None else max_samples}")

    query_encode_fn = _load_text_embedder(embedding_name)
    all_embeddings, all_sentences = _load_embedding_corpus(embedding_name)

    stem = _build_stem(model=model, embedding_name=embedding_name, prompt_tag=prompt_tag)
    run_dir = OUTPUT_DIR / stem
    run_dir.mkdir(parents=True, exist_ok=True)

    pred_path = run_dir / "test_predictions.csv"
    fail_path = run_dir / "failures.csv"
    cache_path = run_dir / "llm_cache.jsonl"
    summary_path = run_dir / "summary.csv"
    typec_result_path = run_dir / "test_results.csv"
    by_moves_path = run_dir / "by_moves_results.csv"
    compare_path = OUTPUT_DIR / "compare_all_typec_results.csv"

    prompt = _PROMPT_MINIMAL if _normalise_prompt_tag(prompt_tag) == "minimal" else _PROMPT_DETAILED
    test_records = _load_test_records(max_samples=max_samples)
    n_total = len(test_records)
    pad = len(str(n_total))

    done_ids = _read_done_ids(pred_path, fail_path, retry_failed=retry_failed)
    cache = _load_cache(cache_path)
    cache_lock = threading.Lock()

    todo = [r for r in test_records if r["sentence_id"] not in done_ids]
    print(f"[openrouter-typec] Total target records: {n_total}")
    print(f"[openrouter-typec] Already done/skipped: {len(done_ids)}")
    print(f"[openrouter-typec] Remaining this run: {len(todo)}")
    print(f"[openrouter-typec] Cached LLM responses: {len(cache)}")

    def fetch_one(record: dict) -> dict:
        img_path = record["img_path"]
        key = _cache_key(model, prompt_tag, img_path)
        if key in cache:
            entry = cache[key]
            return {"ok": True, "record": record, "llm_raw": entry["llm_raw"], "raw_resp": entry.get("api_response"), "cached": True}

        if api_delay_s > 0:
            time.sleep(api_delay_s)

        try:
            llm_raw, raw_resp = _openrouter_describe(client, model, img_path, prompt, retries=retries)
            cache_entry = {
                "ok": True,
                "cache_key": key,
                "model": model,
                "prompt_tag": prompt_tag,
                "img_path": str(img_path),
                "sentence_id": record["sentence_id"],
                "llm_raw": llm_raw,
                "api_response": raw_resp,
            }
            _append_jsonl_threadsafe(cache_path, cache_entry, cache_lock)
            return {"ok": True, "record": record, "llm_raw": llm_raw, "raw_resp": raw_resp, "cached": False}
        except Exception as exc:
            return {"ok": False, "record": record, "error": str(exc)}

    completed = 0
    success = 0
    failed = 0

    with cf.ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(fetch_one, r): r for r in todo}
        for future in cf.as_completed(futures):
            completed += 1
            out = future.result()
            record = out["record"]
            img_path = record["img_path"]
            true_sentence = record["true_sentence"]

            if not out["ok"]:
                failed += 1
                err_row = {
                    "run_id": f"LLM-openrouter-{embedding_name}",
                    "image_id": img_path.name,
                    "img_path": str(img_path),
                    "sentence_id": record["sentence_id"],
                    "true_sentence": true_sentence,
                    "n_moves": record["n_moves"],
                    "error": out["error"],
                }
                _append_csv_row(err_row, fail_path)
                print(f"  [{completed:{pad}d}/{len(todo)}] FAIL | sid={record['sentence_id']} | {out['error'][:160]}")
                continue

            try:
                llm_raw = out["llm_raw"]
                llm_canon = _canonicalise_typec_output(llm_raw)
                result = _retrieve(llm_canon, query_encode_fn, all_embeddings, all_sentences, true_sentence)

                row = {
                    "run_id": f"LLM-openrouter-{embedding_name}",
                    "image_id": img_path.name,
                    "img_path": str(img_path),
                    "sentence_id": record["sentence_id"],
                    "true_sentence": true_sentence,
                    "sentence": true_sentence,
                    "llm_raw": llm_raw,
                    "llm_canonical": llm_canon,
                    "pred_sentence": result["pred_sentence"],
                    "n_moves": record["n_moves"],
                    "true_rank": result["true_rank"],
                    "cosine_sim": result["cosine_sim"],
                    "correct_cosine": result["cosine_sim"],
                    "is_top1_correct": result["top_1_correct"],
                    "cached": int(out.get("cached", False)),
                    **{k: v for k, v in result.items() if k.startswith("top_") and k.endswith("_correct")},
                }
                _append_csv_row(row, pred_path)
                success += 1
                hit = "HIT " if result["top_1_correct"] else "MISS"
                cache_tag = "cache" if out.get("cached") else "api"
                print(
                    f"  [{completed:{pad}d}/{len(todo)}] {hit} | {cache_tag} | "
                    f"rank={result['true_rank']:>5d} | cos={result['cosine_sim']:.4f} | "
                    f"sid={record['sentence_id']}"
                )
            except Exception as exc:
                failed += 1
                err_row = {
                    "run_id": f"LLM-openrouter-{embedding_name}",
                    "image_id": img_path.name,
                    "img_path": str(img_path),
                    "sentence_id": record["sentence_id"],
                    "true_sentence": true_sentence,
                    "n_moves": record["n_moves"],
                    "error": f"retrieval/save error: {exc}",
                }
                _append_csv_row(err_row, fail_path)
                print(f"  [{completed:{pad}d}/{len(todo)}] FAIL-RETRIEVAL | sid={record['sentence_id']} | {exc}")

    metrics = _save_final_outputs(pred_path, summary_path, typec_result_path, by_moves_path, model, embedding_name)

    if make_compare:
        comp = build_comparison_table(PROJECT_ROOT / "results22", compare_path)
        print(f"[saved] comparison table → {compare_path} ({len(comp)} rows)")

    print("\n[openrouter-typec] Run finished")
    print(f"  Success this run : {success}")
    print(f"  Failed this run  : {failed}")
    print(f"  Predictions      : {pred_path}")
    print(f"  Failures         : {fail_path}")
    print(f"  Cache            : {cache_path}")

    if metrics is not None:
        print(f"  Current n_eval   : {metrics['n_test']}")
        print(f"  Top-1            : {metrics['top_1_acc']:.4f}")
        print(f"  Top-5            : {metrics['top_5_acc']:.4f}")
        print(f"  MRR              : {metrics['mrr']:.4f}")
        print(f"  Mean rank        : {metrics['mean_rank']:.1f}")
        print(f"  Median rank      : {metrics['median_rank']:.1f}")


# CLI


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full-run OpenRouter vision LLM comparison for Type-C")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--embedding", default=DEFAULT_EMBEDDING, choices=["sbert", "tfidf"])
    parser.add_argument("--prompt-tag", default="detailed", choices=["detailed", "minimal", "none"])
    parser.add_argument("--max-samples", type=int, default=None, help="Default None means full test set.")
    parser.add_argument("--api-delay", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=6, help="Concurrent API requests. Use 4-8 normally.")
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--retry-failed", action="store_true", help="Retry samples already listed in failures.csv.")
    parser.add_argument("--no-compare", action="store_true", help="Do not build compare_all_typec_results.csv.")
    args = parser.parse_args()

    run_openrouter_typec_comparison(
        model=args.model,
        embedding_name=args.embedding,
        prompt_tag=args.prompt_tag,
        max_samples=args.max_samples,
        api_delay_s=args.api_delay,
        workers=args.workers,
        retries=args.retries,
        retry_failed=args.retry_failed,
        make_compare=not args.no_compare,
    )
