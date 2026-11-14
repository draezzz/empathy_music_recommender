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

# Export calibrated rule parameters & findings for Member 2 (Study 2 Recommender)
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