"""04 — Online process monitoring and release-readiness on the simulated corpus.

This is the task the workbook framed the paper around: detect abnormal process
trajectories and predict release readiness. It can only be run on the SIMULATED
corpus, because no real batch trajectories are available to this study. Every
number in this script therefore describes the simulator, not a real process, and
is labelled as such in the paper.

Method: multiway (batch-wise unfolded) PCA monitoring in the Nomikos-MacGregor
tradition. A PCA model is fitted on the unfolded trajectories of nominal training
batches; online, Hotelling T-squared and squared prediction error are evaluated
on the partial trajectory at each sampling instant and compared with control
limits calibrated on nominal training batches at a fixed false-alarm budget.

Two results matter more than the headline detection rate:
  1. Detection is reported separately for failure modes that leave a trajectory
     signature and for those that do not. The latter set a hard ceiling that no
     amount of modelling can raise.
  2. The false-alarm budget is swept, so detection is reported as a curve rather
     than at one operating point chosen after the fact.

Outputs: results/monitoring.json

Independent work. Carried out on personal time and equipment, not connected to
the author's employment. No proprietary, confidential or internal data of any
organization was used. All data is public.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))  # run from anywhere

from common import INTERIM, SEEDS, write_json, mean_sd

OBSERVABLE = {"sterility_microbial", "potency_assay", "impurity_degradant",
              "cgmp_process_deviation"}
UNOBSERVABLE = {"cross_contamination_foreign", "container_closure"}
ALPHA_GRID = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10]
N_PC = 12


def load():
    z = np.load(INTERIM / "sim_trajectories.npz", allow_pickle=True)
    A = z["traj"].astype(float)           # (n_batches, n_time, n_vars)
    t = z["t"].astype(float)
    disp = z["disposition"].astype(int)
    mode = z["mode"].astype(str)
    # Sensor dropouts: forward-fill along time, then fill any leading gap with
    # the column median. This is the imputation an on-line system would do.
    n, T, V = A.shape
    for i in range(n):
        for j in range(V):
            col = A[i, :, j]
            bad = np.isnan(col)
            if bad.any():
                good = np.where(~bad)[0]
                if good.size == 0:
                    col[:] = 0.0
                else:
                    col[:] = np.interp(np.arange(T), good, col[good])
    return A, t, disp, mode


def unfold(A, upto):
    """Batch-wise unfolding: each batch becomes one row of (time x variable)."""
    return A[:, :upto, :].reshape(A.shape[0], -1)


def monitor(seed: int, A, t, disp, mode) -> dict:
    idx = np.arange(len(A))
    tr, te = train_test_split(idx, test_size=0.4, random_state=seed, stratify=disp)
    # The monitoring model is fitted on nominal training batches only: an in-
    # control reference set, as MSPC requires.
    nominal_tr = tr[(disp[tr] == 0) & (mode[tr] == "none")]

    T = A.shape[1]
    checkpoints = np.arange(8, T + 1, 4)   # evaluate every 8 h of process time

    # Per checkpoint: fit a PCA on the nominal reference and record the two
    # statistics for every test batch.
    stat_T2 = np.full((len(te), len(checkpoints)), np.nan)
    stat_Q = np.full((len(te), len(checkpoints)), np.nan)
    lim_T2 = np.zeros((len(checkpoints), len(ALPHA_GRID)))
    lim_Q = np.zeros((len(checkpoints), len(ALPHA_GRID)))

    for ci, c in enumerate(checkpoints):
        Xn = unfold(A[nominal_tr], c)
        mu, sd = Xn.mean(0), Xn.std(0)
        sd = np.where(sd > 1e-9, sd, 1.0)
        Zn = (Xn - mu) / sd
        k = min(N_PC, Zn.shape[0] - 1, Zn.shape[1])
        p = PCA(n_components=k, random_state=seed).fit(Zn)

        def stats(Z):
            S = p.transform(Z)
            lam = np.maximum(p.explained_variance_, 1e-12)
            t2 = (S ** 2 / lam).sum(1)
            R = Z - p.inverse_transform(S)
            q = (R ** 2).sum(1)
            return t2, q

        t2n, qn = stats(Zn)
        for ai, a in enumerate(ALPHA_GRID):
            lim_T2[ci, ai] = np.quantile(t2n, 1 - a)
            lim_Q[ci, ai] = np.quantile(qn, 1 - a)

        Zt = (unfold(A[te], c) - mu) / sd
        stat_T2[:, ci], stat_Q[:, ci] = stats(Zt)

    out = {"alpha": ALPHA_GRID, "per_alpha": []}
    y = disp[te]
    m = mode[te]
    for ai, a in enumerate(ALPHA_GRID):
        alarm = (stat_T2 > lim_T2[:, ai]) | (stat_Q > lim_Q[:, ai])
        ever = alarm.any(1)
        first = np.where(ever, alarm.argmax(1), -1)
        tt = np.where(first >= 0, t[checkpoints[np.clip(first, 0, None)] - 1], np.nan)

        nominal = (m == "none") & (y == 0)
        obs_fail = np.isin(m, list(OBSERVABLE)) & (y == 1)
        unobs_fail = np.isin(m, list(UNOBSERVABLE)) & (y == 1)
        any_fail = y == 1

        def rate(mask):
            return float(ever[mask].mean()) if mask.sum() else float("nan")

        out["per_alpha"].append({
            "alpha": a,
            "false_alarm_rate_nominal": rate(nominal),
            "detection_rate_all_rejected": rate(any_fail),
            "detection_rate_observable_modes": rate(obs_fail),
            "detection_rate_unobservable_modes": rate(unobs_fail),
            "median_time_to_detection_h_observable": (
                float(np.nanmedian(tt[obs_fail])) if obs_fail.sum() else float("nan")),
            "median_time_to_detection_h_all": (
                float(np.nanmedian(tt[any_fail])) if any_fail.sum() else float("nan")),
            "n_nominal": int(nominal.sum()), "n_observable_fail": int(obs_fail.sum()),
            "n_unobservable_fail": int(unobs_fail.sum()),
        })

    # --- release-readiness: predict end-of-batch disposition from the trajectory
    # summary available at the end of the production phase (t = 200 h), i.e.
    # before harvest, which is when the prediction would be useful.
    c_ready = int(np.searchsorted(t, 200.0))
    Xr_tr = unfold(A[tr], c_ready); Xr_te = unfold(A[te], c_ready)
    mu, sd = Xr_tr.mean(0), Xr_tr.std(0); sd = np.where(sd > 1e-9, sd, 1.0)
    clf = HistGradientBoostingClassifier(max_iter=250, random_state=seed)
    clf.fit((Xr_tr - mu) / sd, disp[tr])
    p_te = clf.predict_proba((Xr_te - mu) / sd)[:, 1]
    prevalence = float(disp[te].mean())
    out["readiness"] = {
        "cutoff_h": 200.0,
        "auroc": float(roc_auc_score(disp[te], p_te)),
        "auprc": float(average_precision_score(disp[te], p_te)),
        "auprc_baseline_prevalence": prevalence,
        "auprc_lift_over_prevalence": float(
            average_precision_score(disp[te], p_te) / prevalence),
    }
    return out


def main() -> None:
    A, t, disp, mode = load()
    runs = [monitor(s, A, t, disp, mode) for s in SEEDS]

    agg = {"alpha_grid": ALPHA_GRID, "n_pc": N_PC, "n_seeds": len(SEEDS),
           "corpus": "SIMULATED ONLY -- no real trajectories exist in this study",
           "per_alpha": [], "readiness": {}}
    keys = [k for k in runs[0]["per_alpha"][0] if k != "alpha"]
    for ai, a in enumerate(ALPHA_GRID):
        rec = {"alpha": a}
        for k in keys:
            vals = [r["per_alpha"][ai][k] for r in runs]
            vals = [v for v in vals if v == v]
            rec[k] = mean_sd(vals) if vals else None
        agg["per_alpha"].append(rec)
    for k in runs[0]["readiness"]:
        agg["readiness"][k] = mean_sd([r["readiness"][k] for r in runs])

    write_json("monitoring.json", agg)
    for rec in agg["per_alpha"]:
        print("alpha %.3f  FA %.3f+-%.3f  det(obs) %.3f+-%.3f  det(unobs) %.3f  "
              "TTD %.0fh" % (
                  rec["alpha"], rec["false_alarm_rate_nominal"]["mean"],
                  rec["false_alarm_rate_nominal"]["sd"],
                  rec["detection_rate_observable_modes"]["mean"],
                  rec["detection_rate_observable_modes"]["sd"],
                  rec["detection_rate_unobservable_modes"]["mean"],
                  rec["median_time_to_detection_h_observable"]["mean"]))
    r = agg["readiness"]
    print("readiness AUROC %.3f+-%.3f  AUPRC %.3f+-%.3f (prevalence %.3f)" % (
        r["auroc"]["mean"], r["auroc"]["sd"], r["auprc"]["mean"], r["auprc"]["sd"],
        r["auprc_baseline_prevalence"]["mean"]))


if __name__ == "__main__":
    main()
