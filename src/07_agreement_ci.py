"""07 — Uncertainty on the cross-corpus agreement.

The summary in script 03 computes one Spearman correlation per metric, over the
six method means. With n=6 that single coefficient carries no usable uncertainty,
which is the weakest point of the central claim.

Here we instead compute the correlation *per seed*: for seed s, rank the six
methods by their value on the simulated corpus at seed s and by their value on
the real corpus at seed s, and correlate. That gives ten independent estimates
per metric, hence a mean, a standard deviation and a percentile interval. It also
allows a direct paired test of the paper's actual claim --- that fidelity and
privacy metrics agree across corpora while utility metrics do not --- by
comparing the two groups of per-seed correlations with a Mann-Whitney U test.

Outputs: results/agreement_ci.json

Independent work. Carried out on personal time and equipment, not connected to
the author's employment. No proprietary, confidential or internal data of any
organization was used. All data is public.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))  # run from anywhere

from common import RESULTS, SEEDS, write_json
from generators import METHODS

PRIMARY = {
    "fid_c2st_auc": "fidelity", "fid_ks_mean": "fidelity",
    "fid_corr_mae": "fidelity", "fid_pca_overlap": "fidelity",
    "util_auroc": "utility", "util_auprc": "utility",
    "priv_dcr_ratio": "privacy", "priv_mia_auc": "privacy",
}


def main() -> None:
    df = pd.read_csv(RESULTS / "evaluation_raw.csv")
    df = df[df.generator.isin(METHODS)]

    out = {"n_seeds": len(SEEDS), "n_methods": len(METHODS), "per_metric": {}}
    groups = {"fidelity": [], "utility": [], "privacy": []}

    for metric, axis in PRIMARY.items():
        rhos = []
        for s in SEEDS:
            a, b = [], []
            for g in METHODS:
                va = df[(df.corpus == "SIM") & (df.seed == s) & (df.generator == g)][metric]
                vb = df[(df.corpus == "REAL") & (df.seed == s) & (df.generator == g)][metric]
                if va.empty or vb.empty or va.isna().all() or vb.isna().all():
                    a, b = [], []
                    break
                a.append(float(va.iloc[0])); b.append(float(vb.iloc[0]))
            if len(a) == len(METHODS) and not (np.isnan(a).any() or np.isnan(b).any()):
                r, _ = spearmanr(a, b)
                if r == r:
                    rhos.append(float(r))
        if not rhos:
            out["per_metric"][metric] = {"axis": axis, "n_usable_seeds": 0,
                                         "note": "no seed produced a complete ranking"}
            continue
        arr = np.asarray(rhos)
        # Between-method spread relative to seed noise: a metric on which the six
        # methods barely differ produces a ranking of noise, and its correlation
        # should not be read as agreement about anything.
        spread = []
        for corpus in ("SIM", "REAL"):
            sub = df[df.corpus == corpus]
            means = sub.groupby("generator")[metric].mean().reindex(METHODS)
            sds = sub.groupby("generator")[metric].std().reindex(METHODS)
            spread.append(float(means.std() / max(sds.mean(), 1e-12)))
        out["per_metric"][metric] = {
            "axis": axis,
            "n_usable_seeds": len(rhos),
            "rho_mean": float(arr.mean()),
            "rho_sd": float(arr.std(ddof=1)),
            "rho_p05": float(np.percentile(arr, 5)),
            "rho_p95": float(np.percentile(arr, 95)),
            "separation_sim": spread[0],
            "separation_real": spread[1],
            "values": rhos,
        }
        groups[axis].extend(rhos)

    fid_priv = groups["fidelity"] + groups["privacy"]
    util = groups["utility"]
    if fid_priv and util:
        u, p = mannwhitneyu(fid_priv, util, alternative="greater")
        out["contrast"] = {
            "test": "Mann-Whitney U, one-sided (fidelity+privacy > utility)",
            "n_fidelity_privacy": len(fid_priv), "n_utility": len(util),
            "median_fidelity_privacy": float(np.median(fid_priv)),
            "median_utility": float(np.median(util)),
            "U": float(u), "p": float(p),
            "note": ("Per-seed correlations within a metric are not independent of "
                     "each other in the strict sense -- they share the same six "
                     "methods -- so this p-value should be read as a descriptive "
                     "contrast, not as a confirmatory test."),
        }
    write_json("agreement_ci.json", out)

    for m, d in out["per_metric"].items():
        if d.get("n_usable_seeds"):
            print("%-18s %-9s rho %6.3f +- %.3f  [%.2f, %.2f]  sep sim/real %.2f/%.2f"
                  % (m, d["axis"], d["rho_mean"], d["rho_sd"], d["rho_p05"],
                     d["rho_p95"], d["separation_sim"], d["separation_real"]))
    if "contrast" in out:
        c = out["contrast"]
        print("contrast: median fid+priv %.3f vs utility %.3f, U=%.0f p=%.4f"
              % (c["median_fidelity_privacy"], c["median_utility"], c["U"], c["p"]))


if __name__ == "__main__":
    main()
