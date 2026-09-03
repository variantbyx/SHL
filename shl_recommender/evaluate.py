"""
Evaluation framework for SHL Assessment Recommendation System.
Measures Mean Recall@10, Precision@10, and Mean Reciprocal Rank (MRR)
against the labeled human ground-truth dataset in data/dataset.xlsx.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from recommender import recommend, ensure_catalog_loaded

ROOT_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(ROOT_DIR, "data")
EXCEL_PATH = os.path.join(DATA_DIR, "dataset.xlsx")
REPORT_PATH = os.path.join(DATA_DIR, "eval_report.json")


def evaluate_system(excel_path: str = EXCEL_PATH, top_k: int = 10) -> Dict[str, Any]:
    """Run full evaluation on the labeled training set."""
    ensure_catalog_loaded()

    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Dataset not found at {excel_path}")

    df_train = pd.read_excel(excel_path, sheet_name="Train-Set")
    grouped = df_train.groupby("Query", sort=False)["Assessment_url"].apply(lambda x: list(set(x))).to_dict()

    recalls = []
    precisions = []
    mrrs = []
    detailed_results = []

    print("\n" + "=" * 75)
    print(f"SHL RECOMMENDER EVALUATION BENCHMARK (Top-K = {top_k})")
    print("=" * 75)

    for idx, (query, true_urls) in enumerate(grouped.items(), 1):
        true_slugs = set(u.rstrip("/").split("/")[-1].lower() for u in true_urls)
        preds = recommend(query, top_k=top_k)
        pred_slugs = [p["url"].rstrip("/").split("/")[-1].lower() for p in preds]

        # Calculate Hits
        hits = sum(1 for s in pred_slugs if s in true_slugs)
        recall = hits / len(true_slugs) if true_slugs else 0.0
        precision = hits / top_k
        
        # Calculate MRR
        mrr = 0.0
        for rank, s in enumerate(pred_slugs, 1):
            if s in true_slugs:
                mrr = 1.0 / rank
                break

        recalls.append(recall)
        precisions.append(precision)
        mrrs.append(mrr)

        query_preview = query.replace("\n", " ")[:60]
        print(f"[{idx:2d}/10] Recall@{top_k}: {recall * 100:5.1f}% | Precision@{top_k}: {precision * 100:5.1f}% | MRR: {mrr:.3f} | {query_preview}...")

        detailed_results.append({
            "query_id": idx,
            "query": query,
            "ground_truth_count": len(true_slugs),
            "ground_truth_slugs": list(true_slugs),
            "predicted_slugs": pred_slugs,
            "hits": hits,
            f"recall@{top_k}": recall,
            f"precision@{top_k}": precision,
            "mrr": mrr
        })

    mean_recall = float(np.mean(recalls))
    mean_precision = float(np.mean(precisions))
    mean_mrr = float(np.mean(mrrs))

    print("-" * 75)
    print(f"BENCHMARK SUMMARY RESULTS:")
    print(f"  -> Mean Recall@{top_k}:    {mean_recall * 100:.2f}%")
    print(f"  -> Mean Precision@{top_k}: {mean_precision * 100:.2f}%")
    print(f"  -> Mean MRR:           {mean_mrr:.4f}")
    print("=" * 75 + "\n")

    report = {
        "metrics": {
            f"mean_recall@{top_k}": mean_recall,
            f"mean_precision@{top_k}": mean_precision,
            "mean_mrr": mean_mrr,
            "total_queries_evaluated": len(grouped)
        },
        "query_details": detailed_results
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Detailed evaluation report saved to {REPORT_PATH}")

    return report


if __name__ == "__main__":
    evaluate_system()
