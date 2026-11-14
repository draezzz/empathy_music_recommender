import numpy as np
import pandas as pd
import pyarrow.parquet as pq

BASE_DIR = "data/yambda/flat/50m"

print("Loading listens.parquet...")
df = pq.read_table(f"{BASE_DIR}/listens.parquet").to_pandas()

# 1. Sample 2,500 whole users deterministically (keeps complete user histories intact)
rng = np.random.default_rng(42)
unique_uids = df["uid"].unique()
sampled_uids = rng.choice(unique_uids, size=min(2500, len(unique_uids)), replace=False)
df = df[df["uid"].isin(sampled_uids)].copy()

# 2. Cast unsigned columns to signed int64 before diff/math to prevent uint32 underflow
df["uid"] = df["uid"].astype("int64")
df["timestamp"] = df["timestamp"].astype("int64")
df["item_id"] = df["item_id"].astype("int64")
df["track_length_seconds"] = df["track_length_seconds"].astype("int64")

# Sort chronologically per user
df = df.sort_values(["uid", "timestamp"], kind="mergesort").reset_index(drop=True)

# 3. Reconstruct sessions (>30 min = 1,800s inactivity gap starts a new session)
SESSION_GAP_SEC = 30 * 60
time_diff = df.groupby("uid")["timestamp"].diff()
df["new_session"] = time_diff.isna() | (time_diff > SESSION_GAP_SEC)
df["session_id"] = df.groupby("uid")["new_session"].cumsum()

# 4. Clean play ratio & define skip targets (primary: <50%, sensitivity: <30%)
df["played_pct"] = df["played_ratio_pct"].clip(lower=0, upper=100).astype("float32")
df["skip_50"] = (df["played_pct"] < 50).astype("int8")
df["skip_30"] = (df["played_pct"] < 30).astype("int8")

# 5. Causally attach prior likes & dislikes (only feedback recorded at or before current listen)
def attach_prior_feedback(main_df, fb_path, col_name):
    fb = pq.read_table(fb_path).to_pandas()
    fb = fb[fb["uid"].isin(sampled_uids)][["uid", "item_id", "timestamp"]].copy()
    fb["uid"] = fb["uid"].astype("int64")
    fb["item_id"] = fb["item_id"].astype("int64")
    fb["timestamp"] = fb["timestamp"].astype("int64")
    # Earliest feedback timestamp per (uid, item_id)
    fb_first = fb.groupby(["uid", "item_id"], as_index=False)["timestamp"].min()
    fb_first = fb_first.rename(columns={"timestamp": f"{col_name}_ts"})
    merged = main_df.merge(fb_first, on=["uid", "item_id"], how="left")
    merged[col_name] = (merged[f"{col_name}_ts"].notna() & (merged[f"{col_name}_ts"] <= merged["timestamp"])).astype("int8")
    return merged.drop(columns=[f"{col_name}_ts"])

df = attach_prior_feedback(df, f"{BASE_DIR}/likes.parquet", "prior_like")
df = attach_prior_feedback(df, f"{BASE_DIR}/dislikes.parquet", "prior_dislike")

out_path = "data/yambda/sessions_clean.parquet"
df.to_parquet(out_path, index=False)

print(f"Saved {out_path}")
print(f"Users: {df['uid'].nunique():,} | Sessions: {df.groupby(['uid', 'session_id']).ngroups:,} | Listens: {len(df):,}")
print(f"Skip (<50%) rate: {df['skip_50'].mean():.4f} | Early skip (<30%) rate: {df['skip_30'].mean():.4f}")
print(f"Prior like rate: {df['prior_like'].mean():.4f} | Prior dislike rate: {df['prior_dislike'].mean():.4f}")