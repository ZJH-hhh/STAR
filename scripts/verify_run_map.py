"""Check the files referenced by electronic table S29 in an extracted revision archive."""
import argparse
import csv
import json
import math
from pathlib import Path


def verify(root):
    fields = ("eval_manifest", "protocol_file", "episode_metrics_file", "assay_metrics_file",
              "summary_file", "screening_top100_file")
    table = root / "Supplementary_Table_Run_Map.csv"
    with table.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        missing_columns = set(fields + ("job_key", "AUROC_pct", "AP_pct")) - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError("Missing S29 columns: " + ", ".join(sorted(missing_columns)))
        rows = list(reader)
    missing = set()
    for row in rows:
        for field in fields:
            value = row[field]
            if not value or not (root / value).is_file():
                missing.add(f"{field}: {value!r}")
    if len(rows) != 624:
        raise ValueError(f"Expected 624 evaluation records in this revision, found {len(rows)}")
    if missing:
        raise ValueError(f"{len(missing)} unresolved file references; first entries: "
                         + ", ".join(sorted(missing)[:10]))
    for row in rows:
        with (root / row["summary_file"]).open(encoding="utf-8-sig") as stream:
            summary = json.load(stream)
        for field, metric in [("AUROC_pct", "macro_auroc"), ("AP_pct", "macro_auprc")]:
            if not math.isclose(float(row[field]), 100 * summary[metric], rel_tol=0, abs_tol=1e-6):
                raise ValueError(f"{row['job_key']}: {field} differs from {row['summary_file']}")
    return len(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("revision_root", type=Path)
    print(f"Verified file references and summary values in {verify(parser.parse_args().revision_root)} S29 records.")
