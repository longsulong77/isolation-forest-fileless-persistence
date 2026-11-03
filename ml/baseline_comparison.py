# %% [markdown]
# # Phase 12: Baseline Algorithm Comparison
# Compares Isolation Forest against One-Class SVM, Local Outlier Factor,
# Elliptic Envelope (all unsupervised, trained on benign only - same protocol
# as IF), plus a supervised Random Forest as an upper-bound reference (sees
# actual attack labels during training, which IF/OCSVM/LOF/EE do not).
#
# Unsupervised models: same train/test split as Isolation Forest
#   (train=benign_run1+2, test=benign_run3 + all attack runs)
# Random Forest (supervised): needs attack examples to train on, so uses a
#   DIFFERENT split to avoid training on the same attacks it's tested on:
#   RF train = benign_run1+2 + attack_run1
#   RF test  = benign_run3 + attack_run2 + attack_run3
# This makes the RF comparison illustrative (what's achievable WITH labels)
# rather than a strictly apples-to-apples unsupervised comparison - noted
# explicitly in the output and should be called out the same way in the
# thesis results chapter.

# %%
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import matplotlib.pyplot as plt

IN_CSV = r"C:\SysmonLab\data\csv\session_features_v2.csv"
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

# %% [markdown]
# ## Unsupervised split (same as Phase 11/13)

# %%
train_df = df[df["run_file"].isin(["benign_run1.evtx", "benign_run2.evtx"])].copy()
test_df = df[(df["run_file"] == "benign_run3.evtx") | (df["run_file"].str.startswith("attack_run"))].copy()

scaler = MinMaxScaler()
X_train = scaler.fit_transform(train_df[FEATURES])
X_test = scaler.transform(test_df[FEATURES])
y_test = test_df["label"].values

results = {}

# %% [markdown]
# ## Isolation Forest (reference, re-fit here for a fair identical-split comparison)

# %%
clf_if = IsolationForest(n_estimators=100, max_samples=256, contamination="auto", random_state=42, n_jobs=-1)
clf_if.fit(X_train)
scores_if = -clf_if.score_samples(X_test)
results["Isolation Forest"] = (roc_auc_score(y_test, scores_if), average_precision_score(y_test, scores_if))

# %% [markdown]
# ## One-Class SVM

# %%
clf_svm = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale")
clf_svm.fit(X_train)
scores_svm = -clf_svm.decision_function(X_test)
results["One-Class SVM"] = (roc_auc_score(y_test, scores_svm), average_precision_score(y_test, scores_svm))

# %% [markdown]
# ## Local Outlier Factor (novelty mode, so it can score new/test data)

# %%
clf_lof = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=0.05)
clf_lof.fit(X_train)
scores_lof = -clf_lof.decision_function(X_test)
results["Local Outlier Factor"] = (roc_auc_score(y_test, scores_lof), average_precision_score(y_test, scores_lof))

# %% [markdown]
# ## Elliptic Envelope
# Can fail with singular covariance when many boolean features are
# near-constant - wrapped defensively.

# %%
try:
    clf_ee = EllipticEnvelope(contamination=0.05, random_state=42)
    clf_ee.fit(X_train)
    scores_ee = -clf_ee.decision_function(X_test)
    results["Elliptic Envelope"] = (roc_auc_score(y_test, scores_ee), average_precision_score(y_test, scores_ee))
except Exception as e:
    print(f"Elliptic Envelope failed: {e}")
    results["Elliptic Envelope"] = (None, None)

# %% [markdown]
# ## Random Forest (supervised upper-bound reference - different split, see note above)

# %%
rf_train_df = pd.concat([train_df, df[df["run_file"] == "attack_run1.evtx"]])
rf_test_df = df[(df["run_file"] == "benign_run3.evtx") | (df["run_file"].isin(["attack_run2.evtx", "attack_run3.evtx"]))]

rf_scaler = MinMaxScaler()
X_rf_train = rf_scaler.fit_transform(rf_train_df[FEATURES])
X_rf_test = rf_scaler.transform(rf_test_df[FEATURES])
y_rf_train = rf_train_df["label"].values
y_rf_test = rf_test_df["label"].values

clf_rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced")
clf_rf.fit(X_rf_train, y_rf_train)
scores_rf = clf_rf.predict_proba(X_rf_test)[:, 1]
results["Random Forest (supervised, diff. split)"] = (
    roc_auc_score(y_rf_test, scores_rf), average_precision_score(y_rf_test, scores_rf)
)

# %% [markdown]
# ## Summary

# %%
print(f"{'Model':<40} {'ROC AUC':>10} {'PR AUC':>10}")
print("-" * 62)
for name, (roc, pr) in results.items():
    roc_str = f"{roc:.4f}" if roc is not None else "FAILED"
    pr_str = f"{pr:.4f}" if pr is not None else "FAILED"
    print(f"{name:<40} {roc_str:>10} {pr_str:>10}")

valid = {k: v for k, v in results.items() if v[0] is not None}
fig, ax = plt.subplots(figsize=(10, 5))
names = list(valid.keys())
roc_vals = [v[0] for v in valid.values()]
pr_vals = [v[1] for v in valid.values()]
x = np.arange(len(names))
width = 0.35
ax.bar(x - width/2, roc_vals, width, label="ROC AUC")
ax.bar(x + width/2, pr_vals, width, label="PR AUC")
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=20, ha="right")
ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Random baseline")
ax.set_ylabel("AUC")
ax.set_title("Algorithm Comparison")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "algorithm_comparison.png"), dpi=150)
plt.show()
print(f"\nSaved plot to {PLOT_DIR}\\algorithm_comparison.png")