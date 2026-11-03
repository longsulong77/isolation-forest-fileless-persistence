# %% [markdown]
# # Phase 13: Evaluation
# ROC AUC, PR AUC, confusion matrix at an operational threshold, and a
# per-technique (per run_file / per ASEP-flag) breakdown.
#
# Score convention: scikit-learn's score_samples() returns HIGHER = more
# normal. We flip the sign so HIGHER = more anomalous, matching the thesis's
# s -> 1 means anomaly convention (Section 2.4.3).

# %%
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
    confusion_matrix, precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt

IN_CSV = r"C:\SysmonLab\data\csv\session_features.csv"
MODEL_DIR = r"C:\SysmonLab\ml\models"
PLOT_DIR = r"C:\SysmonLab\ml\plots"
import os
os.makedirs(PLOT_DIR, exist_ok=True)

df = pd.read_csv(IN_CSV)

NUMERIC_FEATURES = [
    "num_events", "num_event_types", "max_cmdline_entropy",
    "max_parent_cmdline_entropy", "duration_seconds", "events_per_second",
]
BOOL_FEATURES = [
    "has_registry_event", "has_network_event", "has_file_create",
    "is_run_key", "is_runonce", "is_ifeo", "is_winlogon",
    "is_services_key", "is_shell_folder", "has_parent_in_dataset",
]
FEATURES = NUMERIC_FEATURES + BOOL_FEATURES
for c in BOOL_FEATURES:
    df[c] = df[c].astype(int)

scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
clf = joblib.load(os.path.join(MODEL_DIR, "isolation_forest.joblib"))

# %% [markdown]
# ## Build the held-out test set (benign_run3 + all attack runs)

# %%
test_df = df[
    (df["run_file"] == "benign_run3.evtx") | (df["run_file"].str.startswith("attack_run"))
].copy()

X_test = scaler.transform(test_df[FEATURES])
y_true = test_df["label"].values  # 1 = attack, 0 = benign

raw_scores = clf.score_samples(X_test)
anomaly_scores = -raw_scores  # flip: higher = more anomalous

test_df["anomaly_score"] = anomaly_scores

print(f"Test set size: {len(test_df)} (benign={sum(y_true==0)}, attack={sum(y_true==1)})")

# %% [markdown]
# ## ROC AUC and PR AUC (thesis Section 2.6.1 / 3.5.1)

# %%
roc_auc = roc_auc_score(y_true, anomaly_scores)
pr_auc = average_precision_score(y_true, anomaly_scores)

print(f"ROC AUC: {roc_auc:.4f}")
print(f"PR AUC:  {pr_auc:.4f}")

fpr, tpr, roc_thresholds = roc_curve(y_true, anomaly_scores)
precision, recall, pr_thresholds = precision_recall_curve(y_true, anomaly_scores)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}")
axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3)
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curve - Isolation Forest")
axes[0].legend()

axes[1].plot(recall, precision, label=f"PR AUC = {pr_auc:.3f}")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall Curve - Isolation Forest")
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "roc_pr_curves.png"), dpi=150)
plt.show()
print(f"Saved plot to {PLOT_DIR}\\roc_pr_curves.png")

# %% [markdown]
# ## Operational threshold: Nth percentile of BENIGN TRAINING scores
# Thesis Section 3.1.1 / 3.5.2: threshold (T) derived from Benign Training
# Corpus score distribution, targeting a low operational FPR.
# We compute this against the TRAIN set (benign_run1+2) scores, not test,
# to keep the threshold a property of "normal" rather than tuned on the
# test set.

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

# %% [markdown]
# ## Per-technique detection rate
# Breaks down recall by which attack_run file and which ASEP flag fired,
# using the 95th-percentile threshold as the operating point.

# %%
T95 = np.percentile(train_anomaly_scores, 95)
test_df["flagged"] = (test_df["anomaly_score"] >= T95).astype(int)

attack_only = test_df[test_df["label"] == 1]
print("Detection rate by attack run:")
print(attack_only.groupby("run_file")["flagged"].mean())

print("\nDetection rate by ASEP category (attack sessions only):")
for flag in ["is_run_key", "is_runonce", "is_ifeo", "is_winlogon", "is_services_key", "is_shell_folder"]:
    subset = attack_only[attack_only[flag] == 1]
    if len(subset) > 0:
        print(f"  {flag}: n={len(subset)}, detection_rate={subset['flagged'].mean():.3f}")
    else:
        print(f"  {flag}: n=0 (no attack sessions matched this ASEP pattern)")

benign_only = test_df[test_df["label"] == 0]
print(f"\nFalse positive rate on held-out benign (run3): {benign_only['flagged'].mean():.4f}")

# %%
test_df.to_csv(r"C:\SysmonLab\data\csv\test_results.csv", index=False)
print("Saved per-session results to C:\\SysmonLab\\data\\csv\\test_results.csv")