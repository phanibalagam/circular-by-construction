"""01 — Profile the two real-data anchors.

Anchor A: SECOM (UCI ML Repository, id 179). 1,567 real production runs from a
semiconductor fab, 590 sensor signals, pass/fail disposition labels, July-October
2008. This is the only genuinely REAL process-data-with-disposition-label corpus
available to this study.

Anchor B: openFDA drug/enforcement. Real FDA drug recall records. Used only to
anchor the *failure-mode class mix and severity prior* of the simulator -- it
carries no trajectories, no CQA values, and cannot anchor fidelity.

Outputs: results/real_data_profile.json, results/failure_mode_prior.json

Independent work. Carried out on personal time and equipment, not connected to
the author's employment. No proprietary, confidential or internal data of any
organization was used. All data is public.
"""
from __future__ import annotations

import json
import re
import zipfile
from collections import Counter

import numpy as np
import pandas as pd

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))  # run from anywhere

from common import RAW, INTERIM, write_json

# --------------------------------------------------------------------------
# Anchor A: SECOM
# --------------------------------------------------------------------------


def load_secom() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    X = pd.read_csv(RAW / "secom" / "secom.data", sep=r"\s+", header=None,
                    na_values=["NaN"])
    lab = pd.read_csv(RAW / "secom" / "secom_labels.data", sep=r"\s+", header=None,
                      names=["label", "ts"])
    ts = pd.to_datetime(lab["ts"], format="%d/%m/%Y %H:%M:%S")
    # SECOM encodes fail as +1, pass as -1. Recode to 1 = fail (the minority,
    # release-relevant class) and 0 = pass.
    y = (lab["label"] == 1).astype(int)
    return X, y, ts


def profile_secom() -> dict:
    X, y, ts = load_secom()
    miss = X.isna().mean()
    nun = X.nunique(dropna=True)
    constant = nun <= 1
    # Analysis matrix: drop constant columns and columns missing more than half
    # their values, then median-impute. Both thresholds are reported.
    keep = (~constant) & (miss <= 0.5)
    Xk = X.loc[:, keep]
    Xi = Xk.fillna(Xk.median())
    var = Xi.var()
    Xi = Xi.loc[:, var > 0]

    INTERIM.mkdir(parents=True, exist_ok=True)
    Xi.to_csv(INTERIM / "secom_X.csv.gz")
    y.to_frame("fail").assign(ts=ts).to_csv(INTERIM / "secom_y.csv.gz")

    return {
        "source": "UCI ML Repository, SECOM, id 179",
        "n_runs": int(X.shape[0]),
        "n_signals_raw": int(X.shape[1]),
        "n_fail": int(y.sum()),
        "n_pass": int((1 - y).sum()),
        "fail_rate": float(y.mean()),
        "time_start": str(ts.min()),
        "time_end": str(ts.max()),
        "span_days": int((ts.max() - ts.min()).days),
        "missing": {
            "cell_fraction": float(miss.mean()),
            "max_column_fraction": float(miss.max()),
            "cols_complete": int((miss == 0).sum()),
            "cols_over_50pct_missing": int((miss > 0.5).sum()),
            "mean_missing_per_row": float(X.isna().sum(1).mean()),
            "max_missing_per_row": int(X.isna().sum(1).max()),
        },
        "quality_problems": {
            "constant_columns": int(constant.sum()),
            "duplicate_rows": int(X.duplicated().sum()),
            "scale_min": float(X.min().min()),
            "scale_max": float(X.max().max()),
        },
        "analysis_matrix": {
            "rule": "drop constant columns and columns with >50% missing; median-impute; drop zero-variance",
            "n_signals": int(Xi.shape[1]),
            "n_runs": int(Xi.shape[0]),
            "path": "data/interim/secom_X.csv.gz",
        },
    }


# --------------------------------------------------------------------------
# Anchor B: openFDA drug enforcement -> failure-mode prior
# --------------------------------------------------------------------------

# Rule-based mapping from recall reason text to manufacturing failure modes that
# a batch-release decision would have to catch. Rules are ordered; first match
# wins. Coverage (fraction of records matched) is reported, not assumed.
FAILURE_MODE_RULES: list[tuple[str, str]] = [
    ("sterility_microbial", r"steril|microb|bacteri|fungal|mold|endotoxin|pyrogen|bioburden|contaminat.{0,20}(microb|organism)"),
    ("potency_assay", r"\bsubpotent|superpotent|\bpotency|\bassay\b|out of specification.{0,30}(assay|potency|content)|content uniformity"),
    ("impurity_degradant", r"impurit|degrad|nitrosamin|ndma|ndea|oxidation|related substance"),
    ("cross_contamination_foreign", r"cross[- ]contamin|foreign (matter|material|substance)|particulate|glass|metal|black specks"),
    ("dissolution_physical", r"dissolution|disintegrat|friabilit|hardness|tablet (defect|split|chip)|capsule (defect|split)"),
    ("stability_shelf_life", r"stabilit|shelf[- ]life|fail(?:ed|ure).{0,20}(specification).{0,20}(stability)|expir"),
    ("container_closure", r"container|closure|seal|leak|packag(?:e|ing) (defect|integrity)|cap\b"),
    ("labeling_mixup", r"label|mislabel|misbrand|incorrect (product|strength|barcode|ndc)|mix[- ]?up|wrong (drug|product|strength)"),
    ("cgmp_process_deviation", r"cgmp|gmp deviation|process (deviation|control)|manufacturing (deviation|process)|equipment failure|validation"),
    ("unapproved_undeclared", r"unapproved|without an approved|undeclared|marketed without"),
]

_COMPILED = [(k, re.compile(p, re.I)) for k, p in FAILURE_MODE_RULES]


def classify_reason(text: str) -> str:
    for key, rx in _COMPILED:
        if rx.search(text or ""):
            return key
    return "unclassified"


def profile_openfda() -> dict:
    zpath = RAW / "openfda_enforcement" / "drug-enforcement-0001-of-0001.json.zip"
    with zipfile.ZipFile(zpath) as z:
        name = z.namelist()[0]
        payload = json.loads(z.read(name))
    res = payload["results"]
    last_updated = payload.get("meta", {}).get("last_updated")

    rows = []
    for r in res:
        rows.append({
            "classification": r.get("classification"),
            "product_type": r.get("product_type"),
            "report_date": str(r.get("report_date", ""))[:4],
            "reason": r.get("reason_for_recall", "") or "",
            "voluntary_mandated": r.get("voluntary_mandated"),
        })
    df = pd.DataFrame(rows)
    df = df[df["product_type"] == "Drugs"].copy()
    df["mode"] = df["reason"].map(classify_reason)

    # Severity prior: P(Class | failure mode), from real records.
    cls_order = ["Class I", "Class II", "Class III"]
    df = df[df["classification"].isin(cls_order)]
    mode_counts = df["mode"].value_counts()
    sev = (df.groupby("mode")["classification"]
             .value_counts(normalize=True)
             .unstack(fill_value=0.0)
             .reindex(columns=cls_order, fill_value=0.0))

    coverage = float((df["mode"] != "unclassified").mean())

    prior = {
        "source": "openFDA drug/enforcement",
        "export_last_updated": last_updated,
        "n_records_total": len(res),
        "n_drug_records_classified": int(len(df)),
        "rule_coverage": coverage,
        "n_unclassified": int((df["mode"] == "unclassified").sum()),
        "severity_mix_overall": df["classification"].value_counts(normalize=True).to_dict(),
        "mode_frequency": (mode_counts / mode_counts.sum()).to_dict(),
        "mode_counts": mode_counts.to_dict(),
        "severity_given_mode": sev.to_dict(orient="index"),
        "year_range": [df["report_date"].min(), df["report_date"].max()],
        "caveat": (
            "Recall records describe failures that ESCAPED release and were later "
            "detected in the market. They are a biased sample of manufacturing "
            "failure: modes that batch release reliably catches are under-represented. "
            "This prior anchors the failure-mode MIX of the simulator, not its "
            "absolute failure rate, and not any trajectory."
        ),
    }
    return prior


def main() -> None:
    secom = profile_secom()
    prior = profile_openfda()
    write_json("real_data_profile.json", {"secom": secom})
    write_json("failure_mode_prior.json", prior)

    print("\nSECOM:", secom["n_runs"], "runs,", secom["analysis_matrix"]["n_signals"],
          "usable signals, fail rate %.4f" % secom["fail_rate"])
    print("openFDA: %d drug recalls, rule coverage %.3f" % (
        prior["n_drug_records_classified"], prior["rule_coverage"]))
    for k, v in sorted(prior["mode_frequency"].items(), key=lambda kv: -kv[1]):
        print("   %-32s %.4f" % (k, v))


if __name__ == "__main__":
    main()
