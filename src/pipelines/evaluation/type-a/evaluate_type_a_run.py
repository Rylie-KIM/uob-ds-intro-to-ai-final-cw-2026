import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from type_a_label_parser import parse_label
from type_a_metrics import (
    exact_sentence_match,
    attribute_overlap,
    relation_match,
    object_pair_match,
    full_structure_match,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--master_csv", required=True)
    parser.add_argument("--predictions_pt", required=True)
    parser.add_argument("--embedding_col", required=True)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--embedding_name", required=True)
    parser.add_argument("--loss_name", required=True)
    parser.add_argument("--test_indices_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def load_tensor(path):
    x = torch.load(path, map_location="cpu")
    if isinstance(x, dict):
        if "embeddings" in x:
            x = x["embeddings"]
        elif "embedding" in x:
            x = x["embedding"]
        elif "predictions" in x:
            x = x["predictions"]
    x = torch.as_tensor(x, dtype=torch.float32)
    if x.ndim == 1:
        x = x.unsqueeze(0)
    return x


def main():
    args = parse_args()

    master_csv = Path(args.master_csv)
    predictions_pt = Path(args.predictions_pt)
    test_indices_csv = Path(args.test_indices_csv)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(master_csv)
    test_df = pd.read_csv(test_indices_csv)

    if "row_index" not in test_df.columns:
        raise ValueError("test_indices_csv must contain a 'row_index' column")

    row_indices = test_df["row_index"].tolist()
    eval_df = df.iloc[row_indices].reset_index(drop=True)

    pred_embeddings = load_tensor(predictions_pt)

    if len(eval_df) != len(pred_embeddings):
        raise ValueError(
            f"Prediction count ({len(pred_embeddings)}) does not match test rows ({len(eval_df)})"
        )

    target_embeddings = []
    for rel_path in eval_df[args.embedding_col]:
        emb = load_tensor(Path(rel_path))
        target_embeddings.append(emb.squeeze(0))

    target_embeddings = torch.stack(target_embeddings)

    pred_embeddings = F.normalize(pred_embeddings, p=2, dim=1)
    target_embeddings = F.normalize(target_embeddings, p=2, dim=1)

    similarity_matrix = pred_embeddings @ target_embeddings.T

    rows = []

    cosine_scores = []
    mse_scores = []
    top1_scores = []
    top5_scores = []
    exact_scores = []
    overlap_counts = []
    overlap_ratios = []
    relation_scores = []
    object_pair_scores = []
    full_structure_scores = []
    mrr_scores = []
    true_ranks = []

    for i in range(len(eval_df)):
        true_label = eval_df.iloc[i]["label"]

        pred_vector = pred_embeddings[i]
        true_vector = target_embeddings[i]

        cosine = F.cosine_similarity(
            pred_vector.unsqueeze(0),
            true_vector.unsqueeze(0)
        ).item()
        mse = F.mse_loss(pred_vector, true_vector).item()

        sim_row = similarity_matrix[i]

        # rank of the correct target within the test retrieval corpus
        sorted_indices = torch.argsort(sim_row, descending=True)
        true_rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item() + 1
        reciprocal_rank = 1.0 / true_rank

        top1 = int(true_rank == 1)
        top5 = int(true_rank <= min(5, len(eval_df)))

        nearest_idx = sorted_indices[0].item()
        predicted_label = eval_df.iloc[nearest_idx]["label"]

        parsed_true = parse_label(true_label)
        parsed_pred = parse_label(predicted_label)

        exact = exact_sentence_match(true_label, predicted_label)
        overlap_count, overlap_ratio = attribute_overlap(parsed_true, parsed_pred)
        rel_match = relation_match(parsed_true, parsed_pred)
        obj_match = object_pair_match(parsed_true, parsed_pred)
        full_match = full_structure_match(parsed_true, parsed_pred)

        cosine_scores.append(cosine)
        mse_scores.append(mse)
        top1_scores.append(top1)
        top5_scores.append(top5)
        exact_scores.append(exact)
        overlap_counts.append(overlap_count)
        overlap_ratios.append(overlap_ratio)
        relation_scores.append(rel_match)
        object_pair_scores.append(obj_match)
        full_structure_scores.append(full_match)
        mrr_scores.append(reciprocal_rank)
        true_ranks.append(true_rank)

        rows.append({
            "row_index": row_indices[i],
            "true_label": true_label,
            "predicted_label": predicted_label,
            "true_rank": true_rank,
            "reciprocal_rank": reciprocal_rank,
            "cosine_similarity": cosine,
            "mse": mse,
            "top1_hit": top1,
            "top5_hit": top5,
            "exact_sentence_match": exact,
            "attribute_overlap_count": overlap_count,
            "attribute_overlap_ratio": overlap_ratio,
            "relation_match": rel_match,
            "object_pair_match": obj_match,
            "full_structure_match": full_match,
        })

    details_df = pd.DataFrame(rows)
    details_path = output_dir / f"{args.run_name}_details.csv"
    details_df.to_csv(details_path, index=False)

    summary_df = pd.DataFrame([{
        "run_name": args.run_name,
        "model_name": args.model_name,
        "embedding_name": args.embedding_name,
        "loss_name": args.loss_name,
        "rows_evaluated": len(eval_df),
        "avg_cosine_similarity": sum(cosine_scores) / len(cosine_scores),
        "avg_mse": sum(mse_scores) / len(mse_scores),
        "top1_accuracy": sum(top1_scores) / len(top1_scores),
        "top5_accuracy": sum(top5_scores) / len(top5_scores),
        "test_mrr": sum(mrr_scores) / len(mrr_scores),
        "test_median_rank": float(pd.Series(true_ranks).median()),
        "exact_sentence_match_rate": sum(exact_scores) / len(exact_scores),
        "avg_attribute_overlap_count": sum(overlap_counts) / len(overlap_counts),
        "avg_attribute_overlap_ratio": sum(overlap_ratios) / len(overlap_ratios),
        "relation_match_rate": sum(relation_scores) / len(relation_scores),
        "object_pair_match_rate": sum(object_pair_scores) / len(object_pair_scores),
        "full_structure_match_rate": sum(full_structure_scores) / len(full_structure_scores),
    }])

    summary_path = output_dir / f"{args.run_name}_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\n=== Evaluation Summary ===")
    print(summary_df.to_string(index=False))
    print(f"\nSaved details to: {details_path}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()