# %% [markdown]
# # Phase 11 (v2): Train/Test Split, Scaling, Isolation Forest Training
# Same as train_model.py but using session_features_v2.csv (enriched with
# parent-child rarity + evasion keyword flags).

# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
import joblib
import os

IN_CSV = r"C:\SysmonLab\data\csv\session_features_v2.csv"
MODEL_DIR = r"C:\SysmonLab\ml\models"
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(IN_CSV)
print(df.shape)
print(df["run_file"].value_counts())

# %%
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

# %%
train_df = df[df["run_file"].isin(["benign_run1.evtx", "benign_run2.evtx"])].copy()
test_benign_df = df[df["run_file"] == "benign_run3.evtx"].copy()
test_attack_df = df[df["run_file"].str.startswith("attack_run")].copy()

print(f"Train (benign only): {len(train_df)}")
print(f"Test benign (held-out): {len(test_benign_df)}")
print(f"Test attack: {len(test_attack_df)}")

# %%
scaler = MinMaxScaler()
X_train = scaler.fit_transform(train_df[FEATURES])
X_test_benign = scaler.transform(test_benign_df[FEATURES])
X_test_attack = scaler.transform(test_attack_df[FEATURES])

joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler_v2.joblib"))

# %%
clf = IsolationForest(
    n_estimators=100,
    max_samples=256,
    contamination="auto",
    random_state=42,
    n_jobs=-1,
)
clf.fit(X_train)
joblib.dump(clf, os.path.join(MODEL_DIR, "isolation_forest_v2.joblib"))
print("Model trained and saved (v2).")

# %%
train_scores = clf.score_samples(X_train)
test_benign_scores = clf.score_samples(X_test_benign)
test_attack_scores = clf.score_samples(X_test_attack)

print("Train (benign) score stats:", train_scores.mean(), train_scores.std())
print("Test benign score stats:   ", test_benign_scores.mean(), test_benign_scores.std())
print("Test attack score stats:   ", test_attack_scores.mean(), test_attack_scores.std())