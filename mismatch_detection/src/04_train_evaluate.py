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
        max_samples=0.25,  # Uses all 2,500 users per fold while keeping tree-building fast & RAM-safe
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