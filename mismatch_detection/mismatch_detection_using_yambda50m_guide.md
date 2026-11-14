# EMPATHY: Study 1 — Offline Signal Validation on Yambda-50M

Validates offline predictive behavioral and contextual signals (skip behavior, rolling skip bursts, session context, and causal per-user baseline z-scores) on the **Yambda-50M** dataset (`yandex/yambda`) using user-grouped 5-fold cross-validation (`LogisticRegression`, `RandomForest`, and `LightGBM`), 4-stage signal ablations, SHAP feature explainability, threshold sensitivity checks (`<50%` vs. `<30%` play ratio), and empirical trigger calibration for Study 2.

## Links

| Resource | URL |
| --- | --- |
| Yambda dataset on Hugging Face (`yandex/yambda`, Apache 2.0) | https://huggingface.co/datasets/yandex/yambda |
| Yambda dataset paper (Ploshkin et al., 2025) | https://arxiv.org/abs/2505.22238 |
| Hugging Face Hub CLI (`huggingface_hub`) | https://github.com/huggingface/huggingface_hub |
| LightGBM (gradient boosting framework) | https://github.com/microsoft/LightGBM |
| SHAP (TreeExplainer feature attribution) | https://github.com/shap/shap |
| scikit-learn (GroupKFold, metrics, baselines) | https://github.com/scikit-learn/scikit-learn |
| Apache Arrow / PyArrow (Parquet I/O) | https://github.com/apache/arrow |

Dataset files pulled from `yandex/yambda` (`flat/50m` split):
* `flat/50m/listens.parquet` (`46,467,212` rows, ~369 MB)
* `flat/50m/likes.parquet` (`881,456` rows, ~7.18 MB)
* `flat/50m/dislikes.parquet` (`107,776` rows, ~0.99 MB)

---

## Prerequisites

* Python 3.11+ installed locally (Windows `cmd.exe` / PowerShell or Linux/macOS).
* At least **15–20 GB** of free disk space for raw Parquet files, intermediate session/feature Parquet artifacts, and saved models.
* Working directory outside the main Git repo (or git-ignored `data/` and `outputs/models/`) so multi-gigabyte Parquet files are never committed to GitHub.

---

## Disk & Storage Budget

| Item | Approx. size |
| --- | --- |
| Raw Yambda-50M Parquet files (`listens`, `likes`, `dislikes`) | ~377 MB |
| Reconstructed sessions (`sessions_clean.parquet`, 2,500 users) | ~420 MB |
| Engineered feature matrix (`features_ready.parquet`, 11.88M rows) | ~680 MB |
| Trained LightGBM artifact (`outputs/models/lgbm_study1.joblib`) | ~3 MB |
| CSV result tables & 300 DPI SHAP figures (`outputs/`) | ~2 MB |

---

## Step 0: Environment & Directory Setup

Create a standalone experimentation directory (e.g., `yambda_experiments`), initialize the folder tree, configure `.gitignore`, and install dependencies.

### Windows (`cmd.exe`)

```bat
mkdir data\yambda notebooks src outputs\figures outputs\models
(echo venv/& echo data/& echo outputs/models/& echo __pycache__/& echo *.pyc) > .gitignore

python -m venv venv
venv\Scripts\activate.bat

python -m pip install --upgrade pip
pip install -U "huggingface_hub[cli]" pandas numpy pyarrow scikit-learn lightgbm shap matplotlib seaborn joblib
```

### Linux / macOS (`bash`)

```bash
mkdir -p data/yambda notebooks src outputs/figures outputs/models
printf "venv/\ndata/\noutputs/models/\n__pycache__/\n*.pyc\n" > .gitignore

python3 -m venv venv
source venv/bin/activate

python -m pip install --upgrade pip
pip install -U "huggingface_hub[cli]" pandas numpy pyarrow scikit-learn lightgbm shap matplotlib seaborn joblib
```

Verify installation:

```bash
python -c "import pandas, pyarrow, sklearn, lightgbm, shap; print('All packages ready')"
```

---

## Step 1: Download & Inspect Yambda-50M (`src/01_inspect.py`)

Download the three flat 50M Parquet files from Hugging Face into `data/yambda`:

```bash
hf download yandex/yambda --repo-type dataset \
  --include "flat/50m/listens.parquet" "flat/50m/likes.parquet" "flat/50m/dislikes.parquet" \
  --local-dir data/yambda
```

*(Note: If `listens.parquet` is interrupted on a slow connection, re-run `hf download yandex/yambda flat/50m/listens.parquet --repo-type dataset --local-dir data/yambda` to resume.)*

Save as `src/01_inspect.py` to verify schemas, row counts, and timestamp resolution:

```python
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
```

```bash
python src/01_inspect.py
```

**Schema & Timestamp Findings:**
* `listens.parquet` (`46,467,212` rows): `uid (uint32)`, `timestamp (uint32)`, `item_id (uint32)`, `is_organic (uint8)`, `played_ratio_pct (uint16)`, `track_length_seconds (uint32)`.
* `timestamp` is stored in **anonymized relative seconds** (`39,420` to `25,997,540`, ~300 days from an anonymized origin `0`) rather than wall-clock Unix timestamps. Consequently, wall-clock features (`hour_sin/cos`, `is_weekend`) are omitted in Study 1 and deferred to Study 2.
* Unsigned `uint32` timestamps must be cast to signed `int64` prior to `.diff()` subtraction to prevent unsigned integer underflow.

---

## Step 2: Reconstruct Sessions & Causal Targets (`src/02_prepare_sessions.py`)

Yambda does not include explicit session IDs. Following standard streaming sessionization conventions, consecutive listens by the same user separated by an inactivity gap **$> 30\text{ minutes}$ ($1,800\text{ seconds}$)** start a new session. Explicit feedback (`likes.parquet` and `dislikes.parquet`) is merged **causally** (`feedback_timestamp <= listen_timestamp`) to prevent future data leakage.

Save as `src/02_prepare_sessions.py`:

```python
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
```

```bash
python src/02_prepare_sessions.py
```

**Observed Output:** `2,500` users | `656,672` sessions | `12,539,669` listens | Skip (`<50%`) rate: `0.3623` | Early skip (`<30%`) rate: `0.3247` | Prior like rate: `0.1915` | Prior dislike rate: `0.0031`.

---

## Step 3: Vectorized Rolling & Baseline Feature Engineering (`src/03_build_features.py`)

Computes 15 strictly causal features at track $t$ to predict whether the **next track ($t+1$)** in the same session will be skipped (`target_next_skip` at `<50%` and `target_next_skip_30` at `<30%`). Uses vectorized shifts and cumulative sums rather than `groupby().transform(lambda ...)` so all 12.54M rows process in ~30 seconds.

Save as `src/03_build_features.py`:

```python
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

# 1. Session context features (Cython-vectorized)
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

user_cum_count = g_user.cumcount().astype("float64")
user_cum_sum = g_user["skip_50"].cumsum().astype("float64") - df["skip_50"].astype("float64")

user_exp_mean = np.where(user_cum_count > 0, user_cum_sum / user_cum_count, global_skip_mean)
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
```

```bash
python src/03_build_features.py
```

**Observed Output:** `11,882,997` non-terminal modeling rows | Next-skip (`<50%`) rate: `0.3589` | Next-skip (`<30%`) rate: `0.3224` | Skip burst rate ($\ge 2$ skips in last 4 tracks): `0.3913`.

---

## Step 4: User-Grouped 5-Fold Benchmarking (`src/04_train_evaluate.py`)

Evaluates `LogisticRegression`, `RandomForest`, and `LightGBM` across all **2,500 users** (`11,882,997` rows) using `GroupKFold(n_splits=5)` grouped by `uid` so zero user histories leak across train and validation splits. Reports metrics overall (`All`), on user-selected tracks (`is_organic=1`), and on recommender-driven tracks (`is_organic=0`).

Save as `src/04_train_evaluate.py`:

```python
import time
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

t0 = time.time()
print("Loading full features_ready.parquet...")
df = pd.read_parquet("data/yambda/features_ready.parquet")
print(f"Benchmarking on ALL {len(df):,} rows across {df['uid'].nunique():,} whole users...")

FEATURES = [
    "curr_skip_50", "curr_skip_30", "curr_played_pct",
    "skip_rate_last4", "early_skip_ratio_last4", "skip_burst", "mean_played_pct_last4",
    "session_position", "session_elapsed_sec", "prev_gap_sec", "is_organic",
    "user_baseline_skip", "skip_rate_zscore",
    "prior_like", "prior_dislike",
]

X = df[FEATURES]
y = df["target_next_skip"].values
groups = df["uid"].values

models = {
    "LogisticRegression": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=500, class_weight="balanced", random_state=42),
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_leaf=100,
        max_samples=0.25,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced",
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=100,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced",
        verbose=-1,
    ),
}

gkf = GroupKFold(n_splits=5)
results = []

for name, model in models.items():
    m_start = time.time()
    print(f"Running 5-fold GroupKFold for {name}...")
    oof_probs = np.zeros(len(df), dtype=np.float32)

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups), start=1):
        model.fit(X.iloc[tr_idx], y[tr_idx])
        oof_probs[va_idx] = model.predict_proba(X.iloc[va_idx])[:, 1]
        print(f"  Fold {fold}/5 complete ({time.time() - m_start:.1f}s)")

    oof_preds = (oof_probs >= 0.5).astype(int)

    for slice_name, mask in [
        ("All", np.ones(len(df), dtype=bool)),
        ("Organic (is_organic=1)", (df["is_organic"] == 1).values),
        ("Recommended (is_organic=0)", (df["is_organic"] == 0).values),
    ]:
        results.append({
            "Model": name,
            "Traffic": slice_name,
            "ROC_AUC": round(float(roc_auc_score(y[mask], oof_probs[mask])), 4),
            "Macro_F1": round(float(f1_score(y[mask], oof_preds[mask], average="macro")), 4),
            "Brier_Score": round(float(brier_score_loss(y[mask], oof_probs[mask])), 4),
        })
    print(f"  -> {name} total time: {time.time() - m_start:.1f}s")

res_df = pd.DataFrame(results)
print("\n=== STUDY 1 MODEL COMPARISON (Full 2,500 Users, 5-Fold User-Grouped CV) ===")
print(res_df.to_string(index=False))
res_df.to_csv("outputs/study1_model_comparison.csv", index=False)

print("\nTraining final LightGBM model on all 11.88M rows...")
final_lgbm = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=100,
    n_jobs=-1,
    random_state=42,
    class_weight="balanced",
    verbose=-1,
)
final_lgbm.fit(X, y)
joblib.dump({"model": final_lgbm, "features": FEATURES}, "outputs/models/lgbm_study1.joblib")
print(f"All done in {time.time() - t0:.1f}s -> Saved outputs/study1_model_comparison.csv & outputs/models/lgbm_study1.joblib")
```

```bash
python src/04_train_evaluate.py
```

### Benchmark Results (`outputs/study1_model_comparison.csv`)

| Model | Traffic | ROC-AUC | Macro-F1 | Brier Score |
| --- | --- | --- | --- | --- |
| LogisticRegression | All | 0.8756 | 0.7884 | 0.1430 |
| LogisticRegression | Organic (`is_organic=1`) | 0.8444 | 0.7596 | 0.1665 |
| LogisticRegression | Recommended (`is_organic=0`) | 0.9031 | 0.8161 | 0.1178 |
| RandomForest | All | 0.8832 | 0.7876 | 0.1396 |
| RandomForest | Organic (`is_organic=1`) | 0.8532 | 0.7570 | 0.1621 |
| RandomForest | Recommended (`is_organic=0`) | 0.9090 | 0.8170 | 0.1154 |
| **LightGBM** | **All** | **0.8851** | **0.7891** | **0.1384** |
| **LightGBM** | **Organic (`is_organic=1`)** | **0.8555** | **0.7585** | **0.1609** |
| **LightGBM** | **Recommended (`is_organic=0`)** | **0.9107** | **0.8184** | **0.1144** |

---

## Step 5: Signal Ablations, SHAP Explainability & Trigger Calibration (`src/05_ablation_and_shap.py`)

Answers **RQ3** and tests **H3** by running a 4-stage incremental feature ablation (`Context Only` $\rightarrow$ `Skip Only` $\rightarrow$ `Skip + Context` $\rightarrow$ `Full + Per-User Baseline Z-Scores`), generating 300 DPI SHAP summary figures, and extracting empirical next-track skip rates across `skip_burst` and `skip_rate_zscore` buckets.

Save as `src/05_ablation_and_shap.py`:

```python
import time
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from sklearn.metrics import brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold

t0 = time.time()
print("Loading full features_ready.parquet...")
df = pd.read_parquet("data/yambda/features_ready.parquet")
y = df["target_next_skip"].values
groups = df["uid"].values

context_cols = ["session_position", "session_elapsed_sec", "prev_gap_sec", "is_organic"]
skip_cols = [
    "curr_skip_50", "curr_skip_30", "curr_played_pct",
    "skip_rate_last4", "early_skip_ratio_last4", "skip_burst", "mean_played_pct_last4",
]
baseline_cols = ["user_baseline_skip", "skip_rate_zscore", "prior_like", "prior_dislike"]

ablations = {
    "1_Context_Only": context_cols,
    "2_Skip_Only": skip_cols,
    "3_Skip_Plus_Context": skip_cols + context_cols,
    "4_Full_With_User_Baseline": skip_cols + context_cols + baseline_cols,
}

gkf = GroupKFold(n_splits=5)
ablation_rows = []

for label, cols in ablations.items():
    a_start = time.time()
    print(f"Running 5-fold ablation: {label} ({len(cols)} features)...")
    oof = np.zeros(len(df), dtype=np.float32)

    for tr_idx, va_idx in gkf.split(df[cols], y, groups):
        clf = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=100,
            n_jobs=-1,
            random_state=42,
            class_weight="balanced",
            verbose=-1,
        )
        clf.fit(df[cols].iloc[tr_idx], y[tr_idx])
        oof[va_idx] = clf.predict_proba(df[cols].iloc[va_idx])[:, 1]

    preds = (oof >= 0.5).astype(int)
    ablation_rows.append({
        "Variant": label,
        "Num_Features": len(cols),
        "ROC_AUC": round(float(roc_auc_score(y, oof)), 4),
        "Macro_F1": round(float(f1_score(y, preds, average="macro")), 4),
        "Brier_Score": round(float(brier_score_loss(y, oof)), 4),
    })
    print(f"  -> {label} finished in {time.time() - a_start:.1f}s")

ab_df = pd.DataFrame(ablation_rows)
print("\n=== STUDY 1 SIGNAL ABLATION RESULTS (All 2,500 Users) ===")
print(ab_df.to_string(index=False))
ab_df.to_csv("outputs/study1_ablation.csv", index=False)

# SHAP Feature Importance
print("\nComputing SHAP values on saved LightGBM model...")
bundle = joblib.load("outputs/models/lgbm_study1.joblib")
lgbm, feat_names = bundle["model"], bundle["features"]

X_sample = df[feat_names].sample(n=25_000, random_state=42)
explainer = shap.TreeExplainer(lgbm)
shap_values = explainer.shap_values(X_sample)
sv = shap_values[1] if isinstance(shap_values, list) else shap_values

shap_imp = pd.DataFrame({
    "Feature": feat_names,
    "Mean_Abs_SHAP": np.abs(sv).mean(axis=0),
}).sort_values("Mean_Abs_SHAP", ascending=False)
print("\n=== SHAP FEATURE IMPORTANCE RANKING ===")
print(shap_imp.to_string(index=False))
shap_imp.to_csv("outputs/study1_shap_importance.csv", index=False)

plt.figure(figsize=(10, 6))
shap.summary_plot(sv, X_sample, show=False)
plt.tight_layout()
plt.savefig("outputs/figures/study1_shap_summary.png", dpi=300)
plt.close()

plt.figure(figsize=(9, 5.5))
shap.summary_plot(sv, X_sample, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig("outputs/figures/study1_shap_bar.png", dpi=300)
plt.close()

# Empirical Trigger Calibration Table for Study 2 Condition B
df["z_bin"] = pd.cut(
    df["skip_rate_zscore"],
    bins=[-np.inf, 0.0, 1.0, 1.5, 2.0, np.inf],
    labels=["<=0 (Normal/Below)", "0 to 1.0", "1.0 to 1.5", "1.5 to 2.0", ">2.0 (Extreme Spike)"],
)
thresh_df = (
    df.groupby(["skip_burst", "z_bin"], observed=False)
    .agg(
        Sample_Count=("target_next_skip", "count"),
        Actual_Next_Skip_Rate=("target_next_skip", "mean"),
        Recommended_Next_Skip_Rate=("target_next_skip", lambda s: s[df.loc[s.index, "is_organic"] == 0].mean()),
    )
    .reset_index()
)
thresh_df["Actual_Next_Skip_Rate"] = thresh_df["Actual_Next_Skip_Rate"].round(4)
thresh_df["Recommended_Next_Skip_Rate"] = thresh_df["Recommended_Next_Skip_Rate"].round(4)

print("\n=== EMPIRICAL TRIGGER CALIBRATION TABLE (For Study 2 Condition B) ===")
print(thresh_df.to_string(index=False))
thresh_df.to_csv("outputs/study1_trigger_thresholds.csv", index=False)

print(f"\nStep 4.5 complete in {time.time() - t0:.1f}s!")
```

```bash
python src/05_ablation_and_shap.py
```

### Signal Ablation Results (`outputs/study1_ablation.csv`)

| Variant | Num Features | ROC-AUC | Macro-F1 | Brier Score |
| --- | --- | --- | --- | --- |
| `1_Context_Only` | 4 | 0.8566 | 0.7669 | 0.1534 |
| `2_Skip_Only` | 7 | 0.8678 | 0.7804 | 0.1446 |
| `3_Skip_Plus_Context` | 11 | 0.8812 | 0.7853 | 0.1403 |
| `4_Full_With_User_Baseline` | 15 | 0.8851 | 0.7891 | 0.1384 |

### Top SHAP Features (`outputs/study1_shap_importance.csv`)

1. `curr_played_pct` (`0.8179`)
2. `mean_played_pct_last4` (`0.7666`)
3. `user_baseline_skip` (`0.3342`)
4. `session_elapsed_sec` (`0.2899`)
5. `session_position` (`0.2379`)
6. `prev_gap_sec` (`0.1594`)

### Empirical Trigger Calibration (`outputs/study1_trigger_thresholds.csv`)

| `skip_burst` | `z_bin` | Sample Count | Actual Next-Skip Rate | Recommended Next-Skip Rate |
| --- | --- | --- | --- | --- |
| `0` | `<=0 (Normal/Below)` | 6,384,399 | 0.1318 | 0.0982 |
| `0` | `0 to 1.0` | 642,699 | 0.2775 | 0.2522 |
| `0` | `1.0 to 1.5` | 111,389 | 0.3775 | 0.3538 |
| `0` | `1.5 to 2.0` | 48,129 | 0.3476 | 0.3271 |
| `0` | `>2.0 (Extreme Spike)` | 46,188 | 0.2983 | 0.2672 |
| `1` | `<=0 (Normal/Below)` | 615,473 | 0.6089 | 0.6356 |
| `1` | `0 to 1.0` | 3,007,988 | 0.6890 | 0.7071 |
| `1` | `1.0 to 1.5` | 708,307 | 0.7190 | 0.7169 |
| `1` | `1.5 to 2.0` | 202,916 | 0.6898 | 0.6671 |
| `1` | `>2.0 (Extreme Spike)` | 115,509 | 0.6582 | 0.6174 |

---

## Step 6: 30% Early-Skip Sensitivity Check & Study 2 Handoff (`src/06_sensitivity_and_handoff.py`)

Verifies threshold invariance at a stricter `<30%` early-skip cutoff (`target_next_skip_30`) and exports `outputs/study2_seed_config.json` to seed the live recommender's Condition B trigger parameters.

Save as `src/06_sensitivity_and_handoff.py`:

```python
import json
import time
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold

t0 = time.time()
print("Loading features_ready.parquet for 30% early-skip sensitivity check...")
df = pd.read_parquet("data/yambda/features_ready.parquet")

FEATURES = [
    "curr_skip_50", "curr_skip_30", "curr_played_pct",
    "skip_rate_last4", "early_skip_ratio_last4", "skip_burst", "mean_played_pct_last4",
    "session_position", "session_elapsed_sec", "prev_gap_sec", "is_organic",
    "user_baseline_skip", "skip_rate_zscore",
    "prior_like", "prior_dislike",
]

X = df[FEATURES]
y30 = df["target_next_skip_30"].values
groups = df["uid"].values

gkf = GroupKFold(n_splits=5)
oof_probs = np.zeros(len(df), dtype=np.float32)

for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y30, groups), start=1):
    clf = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=100,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced",
        verbose=-1,
    )
    clf.fit(X.iloc[tr_idx], y30[tr_idx])
    oof_probs[va_idx] = clf.predict_proba(X.iloc[va_idx])[:, 1]
    print(f"  Fold {fold}/5 complete ({time.time() - t0:.1f}s)")

oof_preds = (oof_probs >= 0.5).astype(int)
sens_rows = []
for slice_name, mask in [
    ("All", np.ones(len(df), dtype=bool)),
    ("Organic (is_organic=1)", (df["is_organic"] == 1).values),
    ("Recommended (is_organic=0)", (df["is_organic"] == 0).values),
]:
    sens_rows.append({
        "Target": "next_skip_<30%",
        "Traffic": slice_name,
        "ROC_AUC": round(float(roc_auc_score(y30[mask], oof_probs[mask])), 4),
        "Macro_F1": round(float(f1_score(y30[mask], oof_preds[mask], average="macro")), 4),
        "Brier_Score": round(float(brier_score_loss(y30[mask], oof_probs[mask])), 4),
    })

sens_df = pd.DataFrame(sens_rows)
print("\n=== SENSITIVITY CHECK: TARGET = NEXT TRACK SKIP (<30% PLAYED) ===")
print(sens_df.to_string(index=False))
sens_df.to_csv("outputs/study1_sensitivity_30pct.csv", index=False)

handoff_config = {
    "dataset": "Yambda-50M (yandex/yambda, Apache 2.0)",
    "sample_scope": {
        "users": int(df["uid"].nunique()),
        "modeling_transitions": int(len(df)),
        "base_next_skip_50_rate": round(float(df["target_next_skip"].mean()), 4),
        "base_next_skip_30_rate": round(float(df["target_next_skip_30"].mean()), 4),
    },
    "recommended_condition_b_rules": {
        "rolling_window_tracks": 4,
        "skip_burst_min_skips": 2,
        "skip_threshold_played_pct": 50,
        "early_skip_threshold_played_pct": 30,
        "zscore_elevated_threshold": 1.0,
        "confidence_cutoff": 0.60,
        "empirical_next_skip_rate_no_burst_normal_z": 0.0982,
        "empirical_next_skip_rate_with_burst_rec_traffic": 0.7071,
        "act_only_at_track_boundary": True,
    },
    "top_predictive_features_by_shap": [
        "curr_played_pct",
        "mean_played_pct_last4",
        "user_baseline_skip",
        "session_elapsed_sec",
        "session_position",
        "prev_gap_sec",
    ],
    "study1_limitations": [
        "No browser/device volume telemetry in Yambda (validated in Study 2).",
        "Wall-clock timestamps are anonymized relative seconds (hour_sin/cos validated in Study 2).",
        "Next-track skip is a behavioral proxy for track-level dissatisfaction, not direct affect ground truth.",
    ],
}

with open("outputs/study2_seed_config.json", "w", encoding="utf-8") as f:
    json.dump(handoff_config, f, indent=2)

print(f"\nPhase 4 complete in {time.time() - t0:.1f}s! Exported outputs/study1_sensitivity_30pct.csv & outputs/study2_seed_config.json")
```

```bash
python src/06_sensitivity_and_handoff.py
```

### Sensitivity Check Results (`outputs/study1_sensitivity_30pct.csv`)

| Target | Traffic | ROC-AUC | Macro-F1 | Brier Score |
| --- | --- | --- | --- | --- |
| `next_skip_<30%` | All | 0.8840 | 0.7770 | 0.1403 |
| `next_skip_<30%` | Organic (`is_organic=1`) | 0.8538 | 0.7465 | 0.1643 |
| `next_skip_<30%` | Recommended (`is_organic=0`) | 0.9106 | 0.8069 | 0.1147 |

---

## Study 1 Limitations (for Paper Section 6)

1. **No volume telemetry:** Yambda records playback completion ratios (`played_ratio_pct`) and explicit feedback (`likes`/`dislikes`), but contains no player volume events (`volumechange`). Volume drop signals are evaluated exclusively in Study 2.
2. **Anonymized wall-clock timestamps:** Yambda timestamps are relative seconds from an anonymized origin rather than Unix epoch times, preventing extraction of circadian features (`hour_sin/cos`, `is_weekend`) offline.
3. **Behavioral proxy vs. affect ground truth:** Next-track skip behavior measures acute track-level dissatisfaction rather than self-reported emotional need states (`wind down`, `focus`, `energize`), motivating the within-subject user study in Study 2.

---

## Version Control & Repository Hygiene

* **Commit to GitHub (`study1_offline_validation/`):** All 6 Python scripts (`src/01_*.py` – `src/06_*.py`), `README.md`, `outputs/*.csv`, `outputs/study2_seed_config.json`, and `outputs/figures/*.png`.
* **Do NOT commit to GitHub:** Raw or intermediate Parquet files (`data/yambda/*.parquet`), virtual environments (`venv/`), or large serialized model binaries (`outputs/models/*.joblib`).

---

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `hf` or `huggingface-cli` not recognized | Run `python -m huggingface_hub.commands.huggingface_cli download ...` or ensure `venv\Scripts` is activated |
| `listens.parquet` download drops mid-way | Re-run `hf download yandex/yambda flat/50m/listens.parquet --repo-type dataset --local-dir data/yambda` to resume |
| `groupby().transform(lambda ...)` stalls for 10+ minutes | Always use the vectorized shift/cumsum implementation in `src/03_build_features.py` (~29s runtime) |
| `RandomForestClassifier` takes too long or spikes RAM | Keep `max_samples=0.25` in `src/04_train_evaluate.py` so all 2,500 users remain in each fold while bounding tree bootstrap size |
| `uint32` timestamp subtraction produces huge numbers | Ensure `timestamp` is cast to `int64` (`df["timestamp"].astype("int64")`) before `.diff()` |

---

## Citation

Cite the Yambda dataset and paper (verify full author list for APA 7):

> Ploshkin, A., et al. (2025). *Yambda: A large-scale multi-modal dataset for music recommendation.* arXiv. https://arxiv.org/abs/2505.22238