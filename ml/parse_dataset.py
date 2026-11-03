"""
Preview/reference version of the Phase 9 parser.
This is what will be placed at C:\\SysmonLab\\ml\\parse_dataset.py on the host.
Not run here - no .evtx files in this environment. Shown for review before
the user copies it over.
"""
import re
import csv
import glob
import math
import os
from collections import Counter
from Evtx.Evtx import Evtx
import xml.etree.ElementTree as ET

RAW_DIR = r"C:\SysmonLab\data\raw"
OUT_CSV = r"C:\SysmonLab\data\csv\dataset.csv"

NS = "{http://schemas.microsoft.com/win/2004/08/events/event}"

FIELDS = [
    "run_file", "label", "EventID", "UtcTime", "ProcessGuid", "ParentProcessGuid",
    "ProcessId", "Image", "ParentImage", "CommandLine", "ParentCommandLine",
    "User", "TargetObject", "Details", "EventType",
]


def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def parse_event_xml(xml_str: str) -> dict:
    root = ET.fromstring(xml_str)
    system = root.find(f"{NS}System")
    event_id = system.findtext(f"{NS}EventID", default="")
    time_elem = system.find(f"{NS}TimeCreated")
    utc_time = time_elem.get("SystemTime") if time_elem is not None else ""

    data = {}
    event_data = root.find(f"{NS}EventData")
    if event_data is not None:
        for d in event_data.findall(f"{NS}Data"):
            name = d.get("Name")
            data[name] = d.text or ""

    return {
        "EventID": event_id,
        "UtcTime": data.get("UtcTime", utc_time),
        "ProcessGuid": data.get("ProcessGuid", ""),
        "ParentProcessGuid": data.get("ParentProcessGuid", ""),
        "ProcessId": data.get("ProcessId", ""),
        "Image": data.get("Image", ""),
        "ParentImage": data.get("ParentImage", ""),
        "CommandLine": data.get("CommandLine", ""),
        "ParentCommandLine": data.get("ParentCommandLine", ""),
        "User": data.get("User", ""),
        "TargetObject": data.get("TargetObject", ""),
        "Details": data.get("Details", ""),
        "EventType": data.get("EventType", ""),
    }


def label_for_file(filename: str) -> int:
    name = filename.lower()
    if name.startswith("benign_run"):
        return 0
    if name.startswith("attack_run"):
        return 1
    return -1  # unknown, will be skipped


def main():
    files = sorted(glob.glob(os.path.join(RAW_DIR, "*.evtx")))
    files = [f for f in files if os.path.basename(f).lower().startswith(("benign_run", "attack_run"))]

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

    total_rows = 0
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=FIELDS + ["cmdline_entropy", "parent_cmdline_entropy"])
        writer.writeheader()

        for path in files:
            fname = os.path.basename(path)
            label = label_for_file(fname)
            if label == -1:
                continue

            print(f"Parsing {fname} (label={label}) ...")
            count = 0
            with Evtx(path) as log:
                for record in log.records():
                    try:
                        xml_str = record.xml()
                        parsed = parse_event_xml(xml_str)
                    except Exception:
                        continue

                    row = {f: parsed.get(f, "") for f in FIELDS if f not in ("run_file", "label")}
                    row["run_file"] = fname
                    row["label"] = label
                    row["cmdline_entropy"] = round(shannon_entropy(parsed.get("CommandLine", "")), 4)
                    row["parent_cmdline_entropy"] = round(shannon_entropy(parsed.get("ParentCommandLine", "")), 4)
                    writer.writerow(row)
                    count += 1
            print(f"  -> {count} rows")
            total_rows += count

    print(f"\nDone. Total rows: {total_rows}")
    print(f"Saved to: {OUT_CSV}")


if __name__ == "__main__":
    main()
