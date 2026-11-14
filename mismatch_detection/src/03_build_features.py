import time
import numpy as np
import pandas as pd

t0 = time.time()
print("Loading sessions_clean.parquet...")
df = pd.read_parquet("data/yambda/sessions_clean.parquet")

# Identify session and user boundaries via fast boolean masks (data is sorted by uid, timestamp)
same_sess_1 = (df["uid"] == df["uid"].shift(1)) & (df["session_id"] == df["session_id"].shift(1))
same_sess_2 = same_sess_1 & (df["uid"] == df["uid"].shift(2)) & (df["session_id"] == df["session_id"].shift(2))
same_sess_3 = same_sess_2 & (df["uid"] == df["uid"].shift(3)) & (df["session_id"] == df["session_id"].shift(3))
same_user_1 = df["uid"] == df["uid"].shift(1)

# 1. Session context features (fully Cython-vectorized)
g_sess = df.groupby(["uid", "session_id"], sort=False)
df["session_position"] = (g_sess.cumcount() + 1).astype("int32")
df["track_played_sec"] = ((df["played_pct"] / 100.0) * df["track_length_seconds"]).astype("float32")
df["session_elapsed_sec"] = g_sess["track_played_sec"].cumsum().astype("float32")

gap = (df["timestamp"] - df["timestamp"].shift(1)).where(same_sess_1, 0).clip(lower=0)
df["prev_gap_sec"] = gap.astype("float32")

# 2. Vectorized rolling 4-track window within session: tracks [t, t-1, t-2, t-3]
s0 = df["skip_50"].astype("float32")
s1 = s0.shift(1).where(same_sess_1, np.nan)
s2 = s0.shift(2).where(same_sess_2, np.nan)
s3 = s0.shift(3).where(same_sess_3, np.nan)

valid_count = 1 + same_sess_1.astype("float32") + same_sess_2.astype("float32") + same_sess_3.astype("float32")
skip_sum4 = s0 + s1.fillna(0) + s2.fillna(0) + s3.fillna(0)

df["skip_rate_last4"] = (skip_sum4 / valid_count).astype("float32")
df["skip_burst"] = (skip_sum4 >= 2).astype("int8")

e0 = df["skip_30"].astype("float32")
e1 = e0.shift(1).where(same_sess_1, 0)
e2 = e0.shift(2).where(same_sess_2, 0)
e3 = e0.shift(3).where(same_sess_3, 0)
df["early_skip_ratio_last4"] = ((e0 + e1 + e2 + e3) / valid_count).astype("float32")

p0 = df["played_pct"].astype("float32")
p1 = p0.shift(1).where(same_sess_1, 0)
p2 = p0.shift(2).where(same_sess_2, 0)
p3 = p0.shift(3).where(same_sess_3, 0)
df["mean_played_pct_last4"] = ((p0 + p1 + p2 + p3) / valid_count).astype("float32")

df["curr_skip_50"] = df["skip_50"].astype("int8")
df["curr_skip_30"] = df["skip_30"].astype("int8")
df["curr_played_pct"] = df["played_pct"].astype("float32")

# 3. Vectorized causal per-user baseline (expanding mean & std shifted by 1 listen)
g_user = df.groupby("uid", sort=False)
global_skip_mean = float(df["skip_50"].mean())

user_cum_count = g_user.cumcount().astype("float64")  # 0 on user's 1st listen, 1 on 2nd, etc.
user_cum_sum = g_user["skip_50"].cumsum().astype("float64") - df["skip_50"].astype("float64")

user_exp_mean = np.where(user_cum_count > 0, user_cum_sum / user_cum_count, global_skip_mean)
# For binary 0/1 variable, sample variance = (sum_x - (sum_x^2)/n) / (n - 1)
user_var = np.where(
    user_cum_count > 1,
    (user_cum_sum - (user_cum_sum ** 2) / np.maximum(user_cum_count, 1)) / np.maximum(user_cum_count - 1, 1),
    0.25 ** 2,
)
user_exp_std = np.sqrt(np.clip(user_var, 1e-6, None))

df["user_baseline_skip"] = user_exp_mean.astype("float32")
df["skip_rate_zscore"] = ((df["skip_rate_last4"] - user_exp_mean) / (user_exp_std + 1e-4)).clip(-10, 10).astype("float32")

# 4. Proxy target: next track (t+1) in the same session
next_same_sess = (df["uid"] == df["uid"].shift(-1)) & (df["session_id"] == df["session_id"].shift(-1))
df["target_next_skip"] = df["skip_50"].shift(-1).where(next_same_sess, np.nan)
df["target_next_skip_30"] = df["skip_30"].shift(-1).where(next_same_sess, np.nan)
df["next_is_organic"] = df["is_organic"].shift(-1).where(next_same_sess, np.nan)

model_df = df.dropna(subset=["target_next_skip"]).copy()
model_df["target_next_skip"] = model_df["target_next_skip"].astype("int8")
model_df["target_next_skip_30"] = model_df["target_next_skip_30"].astype("int8")
model_df["next_is_organic"] = model_df["next_is_organic"].astype("int8")

keep_cols = [
    "uid", "session_id", "is_organic", "next_is_organic",
    "session_position", "session_elapsed_sec", "prev_gap_sec",
    "curr_skip_50", "curr_skip_30", "curr_played_pct",
    "skip_rate_last4", "early_skip_ratio_last4", "skip_burst", "mean_played_pct_last4",
    "user_baseline_skip", "skip_rate_zscore",
    "prior_like", "prior_dislike",
    "target_next_skip", "target_next_skip_30",
]
model_df = model_df[keep_cols]

out_path = "data/yambda/features_ready.parquet"
model_df.to_parquet(out_path, index=False)

print(f"Done in {time.time() - t0:.1f}s -> Saved {out_path}")
print(f"Modeling rows (non-terminal tracks): {len(model_df):,}")
print(f"Target next-skip (<50%) rate: {model_df['target_next_skip'].mean():.4f}")
print(f"Target next-skip (<30%) rate: {model_df['target_next_skip_30'].mean():.4f}")
print(f"Skip burst rate (>=2 skips in last 4): {model_df['skip_burst'].mean():.4f}")