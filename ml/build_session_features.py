"""
Phase 10: Transactional Grouping + Feature Engineering
Reads C:\\SysmonLab\\data\\csv\\dataset.csv (Phase 9 output, one row per raw event)
and produces C:\\SysmonLab\\data\\csv\\session_features.csv (one row per
ProcessGuid "session", per thesis Section 3.3).

Feature categories (thesis Section 3.3.2):
  - Behavioural Context: Image, ParentImage, whether parent also appears in dataset
  - Content-Based Anomaly: max Shannon entropy of CommandLine / ParentCommandLine in session
  - Targeting Indicators: ASEP registry path flags (Run, RunOnce, IFEO, Winlogon, Services, Shell Folders)
  - Temporal Analytics: event count, distinct event types, session duration, events/sec

Note on standardization (thesis Section 3.3.4): Min-Max scaling is intentionally
NOT applied here. It is applied in Phase 11 using a scaler fit ONLY on the
benign training split, then applied to all data (train + test, benign + attack).
Fitting the scaler on the full dataset here would leak attack-set statistics
into training, which the thesis's own 80/20 benign train/test design (Section
3.4.2) implies should not happen.
"""
import pandas as pd
import numpy as np

IN_CSV = r"C:\SysmonLab\data\csv\dataset.csv"
OUT_CSV = r"C:\SysmonLab\data\csv\session_features.csv"

ASEP_PATTERNS = {
    "is_run_key": r"CurrentVersion\\Run\b",
    "is_runonce": r"RunOnce",
    "is_ifeo": r"Image File Execution Options",
    "is_winlogon": r"Winlogon",
    "is_services_key": r"CurrentControlSet\\Services",
    "is_shell_folder": r"User Shell Folders",
}


def main():
    df = pd.read_csv(IN_CSV, dtype=str, keep_default_na=False)
    df["cmdline_entropy"] = pd.to_numeric(df["cmdline_entropy"], errors="coerce").fillna(0.0)
    df["parent_cmdline_entropy"] = pd.to_numeric(df["parent_cmdline_entropy"], errors="coerce").fillna(0.0)
    df["UtcTime_parsed"] = pd.to_datetime(df["UtcTime"], errors="coerce")
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)

    all_guids = set(df["ProcessGuid"].dropna()) - {""}

    for flag, pattern in ASEP_PATTERNS.items():
        df[flag] = df["TargetObject"].str.contains(pattern, case=False, regex=True, na=False)

    sessions = []
    grouped = df[df["ProcessGuid"] != ""].groupby("ProcessGuid")

    for guid, g in grouped:
        row = {
            "ProcessGuid": guid,
            "run_file": g["run_file"].iloc[0],
            "label": int(g["label"].max()),
            "Image": next((v for v in g["Image"] if v), ""),
            "ParentImage": next((v for v in g["ParentImage"] if v), ""),
            "ParentProcessGuid": next((v for v in g["ParentProcessGuid"] if v), ""),
            "num_events": len(g),
            "num_event_types": g["EventID"].nunique(),
            "max_cmdline_entropy": g["cmdline_entropy"].max(),
            "max_parent_cmdline_entropy": g["parent_cmdline_entropy"].max(),
            "has_registry_event": bool((g["EventID"] == "13").any() or (g["EventID"] == "12").any()),
            "has_network_event": bool((g["EventID"] == "3").any()),
            "has_file_create": bool((g["EventID"] == "11").any()),
        }
        for flag in ASEP_PATTERNS:
            row[flag] = bool(g[flag].any())

        times = g["UtcTime_parsed"].dropna()
        if len(times) >= 2:
            duration = (times.max() - times.min()).total_seconds()
        else:
            duration = 0.0
        row["duration_seconds"] = duration
        row["events_per_second"] = (row["num_events"] / duration) if duration > 0 else float(row["num_events"])

        parent_guid = row["ParentProcessGuid"]
        row["has_parent_in_dataset"] = bool(parent_guid and parent_guid in all_guids)

        sessions.append(row)

    out = pd.DataFrame(sessions)
    out.to_csv(OUT_CSV, index=False)

    print(f"Sessions (ProcessGuid groups): {len(out)}")
    print(out["label"].value_counts())
    print(f"Saved to: {OUT_CSV}")


if __name__ == "__main__":
    main()