import pyarrow.parquet as pq
import pandas as pd

base_dir = "data/yambda/flat/50m"

for fname in ["listens.parquet", "likes.parquet", "dislikes.parquet"]:
    fpath = f"{base_dir}/{fname}"
    meta = pq.read_metadata(fpath)
    print(f"=== {fname} (Rows: {meta.num_rows:,}) ===")
    print(pq.read_schema(fpath))
    print()

sample = pq.read_table(f"{base_dir}/listens.parquet").slice(0, 10_000).to_pandas()
print("=== Sample Rows (listens.parquet) ===")
print(sample.head())
print("\nTimestamp min / max:", sample["timestamp"].min(), "/", sample["timestamp"].max())
print("\nTimestamp diff quantiles (within same uid):")
diffs = sample.sort_values(["uid", "timestamp"]).groupby("uid")["timestamp"].diff().dropna()
print(diffs.quantile([0.1, 0.25, 0.5, 0.75, 0.9]))
print("\nOrganic split:\n", sample["is_organic"].value_counts(normalize=True))