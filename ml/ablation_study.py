# %% [markdown]
# # Ablation Study: Isolation Forest with/without each feature category
# Directly answers RQ2 (does Shannon Entropy actually discriminate?) and
# quantifies the contribution of each feature category added across
# Phases 9-10b: Content-Based Anomaly (entropy), Targeting Indicators (ASEP
# flags), Behavioural Context (parent-child rarity), and the evasion-keyword
# flags added during enrichment.
#
# Same train/test split as Phase 11/13 throughout (train=benign_run1+2,
# test=benign_run3 + all attack runs) - only the FEATURE SET changes between
# runs, so differences in ROC/PR AUC are attributable to the features
# removed, not to a different split.

# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
import os

IN_CSV = r"C:\SysmonLab\data\csv\session_features_v2.csv"
PLOT_DIR = r"C:\SysmonLab\ml\plots"
os.makedirs(PLOT_DIR, exist_ok=True)

df = pd.read_csv(IN_CSV)

ALL_NUMERIC = [
    "num_events", "num_event_types", "max_cmdline_entropy",
    "max_parent_cmdline_entropy", "duration_seconds", "events_per_second",
    "pc_pair_rarity",
]
ALL_BOOL = [
    "has_registry_event", "has_network_event", "has_file_create",
    "is_run_key", "is_runonce", "is_ifeo", "is_winlogon",
    "is_services_key", "is_shell_folder", "has_parent_in_dataset",
    "is_novel_pc_pair",
    "has_encoded_flag", "has_frombase64", "has_iex", "has_download_cradle",
    "has_hidden_window", "has_bypass", "has_certutil_decode", "has_long_base64_blob",
]
ALL_FEATURES = ALL_NUMERIC + ALL_BOOL
for c in ALL_BOOL:
    df[c] = df[c].astype(int)

ENTROPY_FEATURES = ["max_cmdline_entropy", "max_parent_cmdline_entropy"]
ASEP_FEATURES = ["is_run_key", "is_runonce", "is_ifeo", "is_winlogon", "is_services_key", "is_shell_folder"]
RARITY_FEATURES = ["pc_pair_rarity", "is_novel_pc_pair", "has_parent_in_dataset"]
EVASION_FEATURES = ["has_encoded_flag", "has_frombase64", "has_iex", "has_download_cradle",
                     "has_hidden_window", "has_bypass", "has_certutil_decode", "has_long_base64_blob"]

VARIANTS = {
    "Full feature set (v2)": ALL_FEATURES,
    "Entropy only (2 features)": ENTROPY_FEATURES,
    "No entropy": [f for f in ALL_FEATURES if f not in ENTROPY_FEATURES],
    "No Targeting Indicators (ASEP)": [f for f in ALL_FEATURES if f not in ASEP_FEATURES],
    "No Behavioural Context (rarity)": [f for f in ALL_FEATURES if f not in RARITY_FEATURES],
    "No evasion-keyword flags": [f for f in ALL_FEATURES if f not in EVASION_FEATURES],
    "Original v1 (pre-enrichment)": [f for f in ALL_FEATURES if f not in RARITY_FEATURES[:2] + EVASION_FEATURES],
}

# %%
train_df = df[df["run_file"].isin(["benign_run1.evtx", "benign_run2.evtx"])].copy()
test_df = df[(df["run_file"] == "benign_run3.evtx") | (df["run_file"].str.startswith("attack_run"))].copy()
y_test = test_df["label"].values

results = {}
for name, feats in VARIANTS.items():
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(train_df[feats])
    X_test = scaler.transform(test_df[feats])

    clf = IsolationForest(n_estimators=100, max_samples=256, contamination="auto", random_state=42, n_jobs=-1)
    clf.fit(X_train)
    scores = -clf.score_samples(X_test)

    roc = roc_auc_score(y_test, scores)
    pr = average_precision_score(y_test, scores)
    results[name] = (roc, pr, len(feats))

# %%
print(f"{'Variant':<35} {'#Feat':>6} {'ROC AUC':>10} {'PR AUC':>10}")
print("-" * 63)
for name, (roc, pr, n) in results.items():
    print(f"{name:<35} {n:>6} {roc:>10.4f} {pr:>10.4f}")

# %%
names = list(results.keys())
roc_vals = [v[0] for v in results.values()]
pr_vals = [v[1] for v in results.values()]
x = np.arange(len(names))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x - width/2, roc_vals, width, label="ROC AUC")
ax.bar(x + width/2, pr_vals, width, label="PR AUC")
ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Random baseline")
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=25, ha="right")
ax.set_ylabel("AUC")
ax.set_title("Ablation: Isolation Forest Feature Category Contribution")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "ablation_study.png"), dpi=150)
plt.show()
print(f"\nSaved plot to {PLOT_DIR}\\ablation_study.png")