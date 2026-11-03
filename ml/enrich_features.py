# %% [markdown]
# # Phase 10b: Feature Enrichment
# Fixes the gap diagnosed from Phase 13 results:
#   1. Parent-child pair rarity - properly operationalizes "Behavioural
#      Context" (thesis Section 2.5.1 / 3.3.2) instead of a single boolean.
#      Frequency table is built from TRAIN (benign_run1+2) sessions ONLY,
#      then applied to all sessions - avoids leaking test-set info into the
#      "what counts as normal" reference.
#   2. Evasion/LOLBIN keyword flags on CommandLine - direct markers of
#      encoding, download cradles, and execution-policy bypass, which the
#      current entropy-only approach was missing for non-Base64 evasion.
#
# Reads dataset.csv (Phase 9, raw events) + session_features.csv (Phase 10,
# sessions) and writes session_features_v2.csv.

# %%
import pandas as pd
import numpy as np
import re

DATASET_CSV = r"C:\SysmonLab\data\csv\dataset.csv"
SESSIONS_CSV = r"C:\SysmonLab\data\csv\session_features.csv"
OUT_CSV = r"C:\SysmonLab\data\csv\session_features_v2.csv"

TRAIN_RUNS = ["benign_run1.evtx", "benign_run2.evtx"]

# %% [markdown]
# ## Part 1: Parent-child pair rarity (fit on TRAIN sessions only)

# %%
sessions = pd.read_csv(SESSIONS_CSV)
sessions["pc_pair"] = sessions["ParentImage"].fillna("") + " -> " + sessions["Image"].fillna("")

train_sessions = sessions[sessions["run_file"].isin(TRAIN_RUNS)]
pc_pair_counts = train_sessions["pc_pair"].value_counts().to_dict()
train_total = len(train_sessions)

def pc_rarity(pair):
    count = pc_pair_counts.get(pair, 0)
    # count=0 (never seen in training) -> rarity=1.0 (max rare/novel)
    # count=train_total (seen every time) -> rarity approaches 0
    return 1.0 - (count / train_total if train_total > 0 else 0)

sessions["pc_pair_count_in_train"] = sessions["pc_pair"].map(lambda p: pc_pair_counts.get(p, 0))
sessions["pc_pair_rarity"] = sessions["pc_pair"].map(pc_rarity)
sessions["is_novel_pc_pair"] = (sessions["pc_pair_count_in_train"] == 0).astype(int)

print("Parent-child rarity added.")
print(f"Unique (parent,child) pairs seen in training: {len(pc_pair_counts)}")
print(f"Novel pairs in test attack sessions: "
      f"{sessions[sessions['run_file'].str.startswith('attack_run')]['is_novel_pc_pair'].sum()} "
      f"/ {len(sessions[sessions['run_file'].str.startswith('attack_run')])}")
print(f"Novel pairs in held-out benign (run3): "
      f"{sessions[sessions['run_file'] == 'benign_run3.evtx']['is_novel_pc_pair'].sum()} "
      f"/ {len(sessions[sessions['run_file'] == 'benign_run3.evtx'])}")

# %% [markdown]
# ## Part 2: Evasion / LOLBIN keyword flags
# Scan CommandLine (and ParentCommandLine) text from the raw event-level
# dataset, aggregated per ProcessGuid (flag = True if ANY event in the
# session matches).

# %%
EVASION_PATTERNS = {
    "has_encoded_flag": r"-enc(odedcommand)?\b",
    "has_frombase64": r"frombase64string",
    "has_iex": r"\biex\b|invoke-expression",
    "has_download_cradle": r"downloadstring|downloadfile|net\.webclient|invoke-webrequest",
    "has_hidden_window": r"-w(indowstyle)?\s+hidden|-nop\b|-noni\b",
    "has_bypass": r"bypass",
    "has_certutil_decode": r"certutil.*(-decode|/decode)",
    "has_long_base64_blob": r"[A-Za-z0-9+/]{40,}={0,2}",
}

df = pd.read_csv(DATASET_CSV, dtype=str, keep_default_na=False)
df["combined_cmdline"] = (df["CommandLine"].fillna("") + " " + df["ParentCommandLine"].fillna("")).str.lower()

for flag, pattern in EVASION_PATTERNS.items():
    df[flag] = df["combined_cmdline"].str.contains(pattern, regex=True, na=False)

agg_cols = {flag: "any" for flag in EVASION_PATTERNS}
evasion_by_session = df[df["ProcessGuid"] != ""].groupby("ProcessGuid").agg(agg_cols).reset_index()

sessions = sessions.merge(evasion_by_session, on="ProcessGuid", how="left")
for flag in EVASION_PATTERNS:
    sessions[flag] = sessions[flag].fillna(False).astype(int)

print("\nEvasion keyword flags added.")
attack_sessions = sessions[sessions["run_file"].str.startswith("attack_run")]
for flag in EVASION_PATTERNS:
    print(f"  {flag}: {attack_sessions[flag].sum()} / {len(attack_sessions)} attack sessions")

# %%
sessions.to_csv(OUT_CSV, index=False)
print(f"\nSaved to: {OUT_CSV}")
print(f"Shape: {sessions.shape}")