# %% [markdown]
# # Phase 13 (v2): Evaluation with enriched features

# %%
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
    confusion_matrix, precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt

IN_CSV = r"C:\SysmonLab\data\csv\session_features_v2.csv"
MODEL_DIR = r"C:\SysmonLab\ml\models"
PLOT_DIR = r"C:\SysmonLab\ml\plots"
os.makedirs(PLOT_DIR, exist_ok=True)

df = pd.read_csv(IN_CSV)

NUMERIC_FEATURES = [
    "num_events", "num_event_types", "max_cmdline_entropy",
    "max_parent_cmdline_entropy", "duration_seconds", "events_per_second",
    "pc_pair_rarity",
]
BOOL_FEATURES = [
    "has_registry_event", "has_network_event", "has_file_create",
    "is_run_key", "is_runonce", "is_ifeo", "is_winlogon",
    "is_services_key", "is_shell_folder", "has_parent_in_dataset",
    "is_novel_pc_pair",
    "has_encoded_flag", "has_frombase64", "has_iex", "has_download_cradle",
    "has_hidden_window", "has_bypass", "has_certutil_decode", "has_long_base64_blob",
]
FEATURES = NUMERIC_FEATURES + BOOL_FEATURES
for c in BOOL_FEATURES:
    df[c] = df[c].astype(int)

scaler = joblib.load(os.path.join(MODEL_DIR, "scaler_v2.joblib"))
clf = joblib.load(os.path.join(MODEL_DIR, "isolation_forest_v2.joblib"))

# %%
test_df = df[
    (df["run_file"] == "benign_run3.evtx") | (df["run_file"].str.startswith("attack_run"))
].copy()

X_test = scaler.transform(test_df[FEATURES])
y_true = test_df["label"].values
raw_scores = clf.score_samples(X_test)
anomaly_scores = -raw_scores
test_df["anomaly_score"] = anomaly_scores

print(f"Test set size: {len(test_df)} (benign={sum(y_true==0)}, attack={sum(y_true==1)})")

# %%
roc_auc = roc_auc_score(y_true, anomaly_scores)
pr_auc = average_precision_score(y_true, anomaly_scores)
print(f"ROC AUC: {roc_auc:.4f}")
print(f"PR AUC:  {pr_auc:.4f}")

fpr, tpr, _ = roc_curve(y_true, anomaly_scores)
precision, recall, _ = precision_recall_curve(y_true, anomaly_scores)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}")
axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3)
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curve - Isolation Forest (v2, enriched features)")
axes[0].legend()

axes[1].plot(recall, precision, label=f"PR AUC = {pr_auc:.3f}")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall Curve - Isolation Forest (v2)")
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "roc_pr_curves_v2.png"), dpi=150)
plt.show()
print(f"Saved plot to {PLOT_DIR}\\roc_pr_curves_v2.png")

# %%
train_df = df[df["run_file"].isin(["benign_run1.evtx", "benign_run2.evtx"])].copy()
X_train = scaler.transform(train_df[FEATURES])
train_anomaly_scores = -clf.score_samples(X_train)

for pct in [95, 99, 99.9]:
    T = np.percentile(train_anomaly_scores, pct)
    y_pred = (anomaly_scores >= T).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print(f"\n--- Threshold at {pct}th percentile of benign train scores (T={T:.4f}) ---")
    print(f"Confusion matrix [[TN FP][FN TP]]:\n{cm}")
    print(f"Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}")

# %%
T95 = np.percentile(train_anomaly_scores, 95)
test_df["flagged"] = (test_df["anomaly_score"] >= T95).astype(int)

attack_only = test_df[test_df["label"] == 1]
print("\nDetection rate by attack run:")
print(attack_only.groupby("run_file")["flagged"].mean())

print("\nDetection rate by ASEP category (attack sessions only):")
for flag in ["is_run_key", "is_runonce", "is_ifeo", "is_winlogon", "is_services_key", "is_shell_folder"]:
    subset = attack_only[attack_only[flag] == 1]
    if len(subset) > 0:
        print(f"  {flag}: n={len(subset)}, detection_rate={subset['flagged'].mean():.3f}")
    else:
        print(f"  {flag}: n=0")

print("\nDetection rate by evasion-keyword category (attack sessions only):")
for flag in ["has_encoded_flag", "has_frombase64", "has_iex", "has_download_cradle",
             "has_hidden_window", "has_bypass", "has_certutil_decode", "has_long_base64_blob",
             "is_novel_pc_pair"]:
    subset = attack_only[attack_only[flag] == 1]
    if len(subset) > 0:
        print(f"  {flag}: n={len(subset)}, detection_rate={subset['flagged'].mean():.3f}")
    else:
        print(f"  {flag}: n=0")

benign_only = test_df[test_df["label"] == 0]
print(f"\nFalse positive rate on held-out benign (run3): {benign_only['flagged'].mean():.4f}")

# %%
test_df.to_csv(r"C:\SysmonLab\data\csv\test_results_v2.csv", index=False)
print("Saved per-session results to C:\\SysmonLab\\data\\csv\\test_results_v2.csv")