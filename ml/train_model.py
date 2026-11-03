# %% [markdown]
# # Phase 11: Train/Test Split, Scaling, Isolation Forest Training
#
# Thesis Section 3.4: 80/20 split on Benign Training Corpus, Min-Max scaling
# fit on benign TRAIN only (to avoid leakage), IsolationForest with
# n_estimators=100, max_samples=256 (Section 3.4.3).
#
# Split strategy note: rather than a random 80/20 row split (which would mix
# benign_run1/2/3 together and let the model see near-duplicate sessions from
# the same run in both train and test), this splits BY RUN FILE:
#   - Train: benign_run1 + benign_run2
#   - Test (benign side): benign_run3  (held-out normal traffic -> false alarm rate)
#   - Test (attack side): attack_run1 + attack_run2 + attack_run3 (all of it -> recall)
# This is stricter than a literal 80/20 row split but avoids a common leakage
# mistake and gives an honest false-positive estimate on genuinely unseen
# benign behavior.

# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
import joblib

IN_CSV = r"C:\SysmonLab\data\csv\session_features.csv"
MODEL_DIR = r"C:\SysmonLab\ml\models"
import os
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(IN_CSV)
print(df.shape)
print(df["run_file"].value_counts())

# %% [markdown]
# ## Feature selection
# Numeric features for the model. Categorical/high-cardinality fields
# (Image, ParentImage, ProcessGuid) are excluded from the numeric matrix but
# kept in the dataframe for later per-technique / per-process inspection.

# %%
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

# %% [markdown]
# ## Split by run file

# %%
train_df = df[df["run_file"].isin(["benign_run1.evtx", "benign_run2.evtx"])].copy()
test_benign_df = df[df["run_file"] == "benign_run3.evtx"].copy()
test_attack_df = df[df["run_file"].str.startswith("attack_run")].copy()

print(f"Train (benign only): {len(train_df)}")
print(f"Test benign (held-out): {len(test_benign_df)}")
print(f"Test attack: {len(test_attack_df)}")

# %% [markdown]
# ## Scale: fit on TRAIN (benign) only, apply everywhere

# %%
scaler = MinMaxScaler()
X_train = scaler.fit_transform(train_df[FEATURES])
X_test_benign = scaler.transform(test_benign_df[FEATURES])
X_test_attack = scaler.transform(test_attack_df[FEATURES])

joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))

# %% [markdown]
# ## Train Isolation Forest
# contamination left at 'auto' here rather than the thesis draft's
# inconsistent 0.001 / 1-10% values (see thesis review note) - 'auto' avoids
# committing to a number that contradicts itself across the draft. The actual
# decision threshold for alerting is set later (Phase 13) from the benign
# score distribution directly, which is what the thesis's 99.999th-percentile
# approach (Section 3.1.1) actually describes - contamination mainly affects
# the model's internal offset_, not what we'll use operationally.

# %%
clf = IsolationForest(
    n_estimators=100,
    max_samples=256,
    contamination="auto",
    random_state=42,
    n_jobs=-1,
)
clf.fit(X_train)
joblib.dump(clf, os.path.join(MODEL_DIR, "isolation_forest.joblib"))
print("Model trained and saved.")

# %% [markdown]
# ## Quick sanity check: score distributions
# Lower score() = more anomalous in scikit-learn's convention.
# We'll flip sign in Phase 13 so higher = more anomalous, matching the
# thesis's s -> 1 means anomaly convention.

# %%
train_scores = clf.score_samples(X_train)
test_benign_scores = clf.score_samples(X_test_benign)
test_attack_scores = clf.score_samples(X_test_attack)

print("Train (benign) score stats:", train_scores.mean(), train_scores.std())
print("Test benign score stats:   ", test_benign_scores.mean(), test_benign_scores.std())
print("Test attack score stats:   ", test_attack_scores.mean(), test_attack_scores.std())
print()
print("If the model learned anything useful, test attack scores should be")
print("noticeably LOWER (more negative / more anomalous) than benign scores.")