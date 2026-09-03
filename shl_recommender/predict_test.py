"""
Generate submission predictions for the test dataset in strict Appendix 3 format.
Output CSV format:
Query,Assessment_url
Query 1,https://www.shl.com/solutions/products/product-catalog/view/...
Query 1,https://www.shl.com/solutions/products/product-catalog/view/...
...
"""

import os
import csv
import pandas as pd
from recommender import recommend, ensure_catalog_loaded

ROOT_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(ROOT_DIR, "data")
EXCEL_PATH = os.path.join(DATA_DIR, "dataset.xlsx")
SUBMISSION_CSV_PATH = os.path.join(ROOT_DIR, "submission.csv")


def generate_test_predictions(excel_path: str = EXCEL_PATH, out_csv: str = SUBMISSION_CSV_PATH, top_k: int = 10):
    ensure_catalog_loaded()

    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Dataset not found at {excel_path}")

    df_test = pd.read_excel(excel_path, sheet_name="Test-Set")
    queries = df_test["Query"].dropna().tolist()

    print(f"\nGenerating predictions for {len(queries)} test queries...")

    rows = []
    for idx, query in enumerate(queries, 1):
        preds = recommend(query, top_k=top_k)
        for p in preds:
            rows.append({
                "Query": query,
                "Assessment_url": p["url"]
            })
        print(f"[{idx}/{len(queries)}] Generated {len(preds)} recommendations for: {query[:50]}...")

    # Write strictly formatted CSV
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Query", "Assessment_url"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSuccessfully wrote {len(rows)} prediction rows to {out_csv}")


if __name__ == "__main__":
    generate_test_predictions()
