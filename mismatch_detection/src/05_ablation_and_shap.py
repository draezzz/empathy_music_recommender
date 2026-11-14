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

# Define the 4 ablation feature subsets
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

# --- SHAP Feature Importance ---
print("\nComputing SHAP values on saved LightGBM model...")
bundle = joblib.load("outputs/models/lgbm_study1.joblib")
lgbm, feat_names = bundle["model"], bundle["features"]

X_sample = df[feat_names].sample(n=25_000, random_state=42)
explainer = shap.TreeExplainer(lgbm)
shap_values = explainer.shap_values(X_sample)
sv = shap_values[1] if isinstance(shap_values, list) else shap_values

# Mean absolute SHAP ranking table
shap_imp = pd.DataFrame({
    "Feature": feat_names,
    "Mean_Abs_SHAP": np.abs(sv).mean(axis=0),
}).sort_values("Mean_Abs_SHAP", ascending=False)
print("\n=== SHAP FEATURE IMPORTANCE RANKING ===")
print(shap_imp.to_string(index=False))
shap_imp.to_csv("outputs/study1_shap_importance.csv", index=False)

# Beeswarm summary plot
plt.figure(figsize=(10, 6))
shap.summary_plot(sv, X_sample, show=False)
plt.tight_layout()
plt.savefig("outputs/figures/study1_shap_summary.png", dpi=300)
plt.close()

# Clean bar chart for paper
plt.figure(figsize=(9, 5.5))
shap.summary_plot(sv, X_sample, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig("outputs/figures/study1_shap_bar.png", dpi=300)
plt.close()

# --- Empirical Trigger Calibration Table for Member 2 ---
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