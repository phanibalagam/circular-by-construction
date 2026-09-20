"""03 — Run the identical fidelity / utility / privacy protocol on both corpora.

Two corpora:
  REAL : SECOM, 1,567 real production runs with real pass/fail disposition.
  SIM  : the mechanistic simulator of script 02.

Six generators plus a bootstrap reference are fitted on each corpus, over ten
seeds. The same metrics are computed in both cases.

The primary question is NOT "which generator wins". It is whether the answer to
"which generator wins", and the absolute metric values, are the same on simulated
data as on real data. If they are not, then evaluating a synthetic-data generator
on a simulator -- which is what a self-validating digital twin does -- tells you
nothing about how it behaves on a real process.

Outputs: results/evaluation_raw.json, results/evaluation_summary.json

Independent work. Carried out on personal time and equipment, not connected to
the author's employment. No proprietary, confidential or internal data of any
organization was used. All data is public.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))  # run from anywhere

from common import INTERIM, RESULTS, SEEDS, write_json, mean_sd
from generators import GENERATORS, METHODS
import metrics as M

N_SYNTH_MULT = 1.0    # fidelity/privacy: synthetic sample size = training size
UTIL_MULT = 5.0       # utility: draw 5x so a rare failure class has a chance to
                      # appear. A generator that still yields <5 positives at 5x
                      # has failed to represent the minority class at all, which
                      # is reported rather than patched around.


def load_real() -> tuple[np.ndarray, np.ndarray, list[str]]:
    X = pd.read_csv(INTERIM / "secom_X.csv.gz", index_col=0)
    y = pd.read_csv(INTERIM / "secom_y.csv.gz", index_col=0)["fail"].to_numpy()
    return X.to_numpy(float), y.astype(int), list(map(str, X.columns))


def load_sim() -> tuple[np.ndarray, np.ndarray, list[str]]:
    d = pd.read_csv(INTERIM / "sim_batch_records.csv.gz")
    y = d["disposition"].astype(int).to_numpy()
    drop = ["batch_id", "disposition", "failure_mode"]
    X = d.drop(columns=drop)
    X = X.loc[:, X.var() > 0]
    X = X.fillna(X.median())
    return X.to_numpy(float), y, list(X.columns)


def standardise(train, *others):
    mu, sd = train.mean(0), train.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    out = [(train - mu) / sd] + [(o - mu) / sd for o in others]
    return out, (mu, sd)


def evaluate_corpus(name: str, X: np.ndarray, y: np.ndarray) -> list[dict]:
    rows = []
    for seed in SEEDS:
        # Three-way split. `train` fits the generator; `holdout` is the
        # non-member set for the privacy tests; `test` is the untouched real
        # test set for the utility test. No set is used for two purposes.
        idx = np.arange(len(X))
        tr_idx, rest = train_test_split(idx, test_size=0.5, random_state=seed,
                                        stratify=y)
        ho_idx, te_idx = train_test_split(rest, test_size=0.5, random_state=seed,
                                          stratify=y[rest])
        (Xtr, Xho, Xte), _ = standardise(X[tr_idx], X[ho_idx], X[te_idx])
        ytr, yho, yte = y[tr_idx], y[ho_idx], y[te_idx]

        # The label is modelled jointly with the features: the generator sees
        # the disposition column, exactly as a batch-record generator would.
        Dtr = np.column_stack([Xtr, ytr.astype(float)])

        base = M.trtr(Xtr, ytr, Xte, yte, seed=seed)

        for gname in ["bootstrap"] + METHODS:
            t0 = time.time()
            gen = GENERATORS[gname](seed=seed)
            gen.fit(Dtr)
            S = gen.sample(int(len(Dtr) * N_SYNTH_MULT))
            fit_s = time.time() - t0

            SU = gen.sample(int(len(Dtr) * UTIL_MULT))
            Xs, ys_raw = SU[:, :-1], SU[:, -1]
            # Continuous generators emit a real-valued label column; threshold at
            # the midpoint of the two label values. Reported as a design choice.
            ys = (ys_raw > 0.5).astype(int)

            fid_ks = M.marginal_ks(Dtr, S)
            fid = {
                **fid_ks,
                "wasserstein_mean": M.marginal_wasserstein(Dtr, S),
                **M.correlation_delta(Dtr, S),
                "pca_overlap": M.pca_subspace_overlap(Dtr, S, k=10),
                "c2st_auc": M.c2st_auc(Dtr, S, seed=seed),
            }
            util = M.tstr(Xs, ys, Xte, yte, seed=seed)
            priv = M.dcr_nndr(S, Dtr, np.column_stack([Xho, yho.astype(float)]))
            priv["mia_auc"] = M.membership_inference_auc(
                S, Dtr, np.column_stack([Xho, yho.astype(float)]))

            rows.append({
                "corpus": name, "generator": gname, "seed": seed,
                "fit_seconds": fit_s,
                "synth_positive_rate": float((S[:, -1] > 0.5).mean()),
                "synth_positive_rate_util": float(ys.mean()),
                "n_synth_positive_util": int(ys.sum()),
                "train_positive_rate": float(ytr.mean()),
                **{f"fid_{k}": v for k, v in fid.items()},
                **{f"util_{k}": v for k, v in util.items()},
                **{f"priv_{k}": v for k, v in priv.items()},
                "util_trtr_auroc": base["auroc"], "util_trtr_auprc": base["auprc"],
            })
            print(f"  [{name}] seed {seed} {gname:>9}  "
                  f"c2st {fid['c2st_auc']:.3f}  ks {fid['ks_mean']:.3f}  "
                  f"tstr-auprc {util['auprc']:.3f}  dcr-ratio {priv['dcr_ratio']:.3f}  "
                  f"({fit_s:.1f}s)", flush=True)
    return rows


HIGHER_IS_BETTER = {"fid_pca_overlap", "util_auroc", "util_auprc",
                    "priv_dcr_ratio", "fid_c2st_auc_inv"}


def summarise(df: pd.DataFrame) -> dict:
    """Per-corpus, per-generator mean and sd, plus the cross-corpus agreement
    that the paper turns on."""
    metric_cols = [c for c in df.columns
                   if c.startswith(("fid_", "util_", "priv_"))
                   and df[c].dtype != object and c != "util_degenerate"]
    summary = {}
    for corpus, gdf in df.groupby("corpus"):
        summary[corpus] = {}
        for gen, sdf in gdf.groupby("generator"):
            summary[corpus][gen] = {c: mean_sd(sdf[c].dropna())
                                    for c in metric_cols if sdf[c].notna().any()}

    # Cross-corpus agreement: for each metric, rank the six methods on each
    # corpus by their mean, and compare the two rankings.
    from scipy.stats import spearmanr, kendalltau
    agreement = {}
    for c in metric_cols:
        try:
            a = np.array([summary["SIM"][g][c]["mean"] for g in METHODS])
            b = np.array([summary["REAL"][g][c]["mean"] for g in METHODS])
        except KeyError:
            continue
        if np.isnan(a).any() or np.isnan(b).any():
            continue
        rho, p_rho = spearmanr(a, b)
        tau, p_tau = kendalltau(a, b)
        agreement[c] = {
            "spearman_rho": float(rho), "spearman_p": float(p_rho),
            "kendall_tau": float(tau), "kendall_p": float(p_tau),
            "sim_best_method": METHODS[int(np.argmin(a))],
            "real_best_method": METHODS[int(np.argmin(b))],
            "sim_values": a.tolist(), "real_values": b.tolist(),
            "note": "argmin reported; interpret direction per metric",
        }
    return {"per_corpus": summary, "cross_corpus_agreement": agreement}


def main() -> None:
    Xr, yr, _ = load_real()
    Xs, ys, _ = load_sim()
    print(f"REAL {Xr.shape} pos={yr.mean():.4f}   SIM {Xs.shape} pos={ys.mean():.4f}")

    rows = evaluate_corpus("REAL", Xr, yr) + evaluate_corpus("SIM", Xs, ys)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "evaluation_raw.csv", index=False)
    write_json("evaluation_summary.json", summarise(df))
    print("done")


if __name__ == "__main__":
    main()
