"""06 — Emit numbers.tex and the three tables.

Every numeral that appears in the paper is a LaTeX macro defined here and read
from results/*.json. The paper body contains no hand-typed results. If a number
in the paper looks wrong, this file and the JSON it reads are the only places to
look.

Independent work. Carried out on personal time and equipment, not connected to
the author's employment. No proprietary, confidential or internal data of any
organization was used. All data is public.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))  # run from anywhere

from common import RESULTS, ROOT
from generators import METHODS

OUT_NUM = ROOT / "numbers.tex"
OUT_MAIN = ROOT / "table_main.tex"
OUT_AGREE = ROOT / "table_agreement.tex"
OUT_MON = ROOT / "table_monitoring.tex"

# The metrics the paper reports agreement over. Deliberately one metric per
# construct: reporting ks_mean, ks_median, ks_p90 and ks_max separately would
# count the same evidence four times.
PRIMARY = [
    ("fid_c2st_auc",    "C2ST AUC",   "fidelity", True),
    ("fid_ks_mean",     "KS",         "fidelity", True),
    ("fid_corr_mae",    "corr MAE",   "fidelity", True),
    ("fid_pca_overlap", "PCA ovl",    "fidelity", False),
    ("util_auroc",      "TSTR AUROC", "utility",  False),
    ("util_auprc",      "TSTR AUPRC", "utility",  False),
    ("priv_dcr_ratio",  "DCR ratio",  "privacy",  False),
    ("priv_mia_auc",    "MIA AUC",    "privacy",  True),
]

_macros: list[str] = []


def mac(name: str, value: str) -> None:
    assert name.isalpha(), name
    _macros.append(rf"\newcommand{{\{name}}}{{{value}}}")


def fmt(x: float, nd: int = 3) -> str:
    return f"{x:.{nd}f}"


def pm(d: dict, nd: int = 3) -> str:
    return rf"{d['mean']:.{nd}f}\,$\pm$\,{d['sd']:.{nd}f}"


def num(x: float) -> str:
    return f"{int(x):,}".replace(",", r"{,}")


def main() -> None:
    prof = json.loads((RESULTS / "real_data_profile.json").read_text())["secom"]
    prior = json.loads((RESULTS / "failure_mode_prior.json").read_text())
    sim = json.loads((RESULTS / "simulator_spec.json").read_text())
    mon = json.loads((RESULTS / "monitoring.json").read_text())
    ev = json.loads((RESULTS / "evaluation_summary.json").read_text())
    S, AGREE = ev["per_corpus"], ev["cross_corpus_agreement"]
    raw = pd.read_csv(RESULTS / "evaluation_raw.csv")
    ci = json.loads((RESULTS / "agreement_ci.json").read_text())
    CI, CON = ci["per_metric"], ci["contrast"]

    # ---- corpora ---------------------------------------------------------
    mac("secomRuns", num(prof["n_runs"]))
    mac("secomSignalsRaw", num(prof["n_signals_raw"]))
    mac("secomSignals", num(prof["analysis_matrix"]["n_signals"]))
    mac("secomFail", num(prof["n_fail"]))
    mac("secomFailRate", fmt(100 * prof["fail_rate"], 2))
    mac("secomSpanDays", num(prof["span_days"]))
    mac("secomMissingPct", fmt(100 * prof["missing"]["cell_fraction"], 2))
    mac("secomMaxColMissingPct", fmt(100 * prof["missing"]["max_column_fraction"], 1))
    mac("secomConstantCols", num(prof["quality_problems"]["constant_columns"]))

    mac("recallRecords", num(prior["n_drug_records_classified"]))
    mac("recallCoveragePct", fmt(100 * prior["rule_coverage"], 1))
    mac("recallExport", prior["export_last_updated"])
    mac("recallSterilityPct", fmt(100 * prior["mode_frequency"]["sterility_microbial"], 1))
    mac("recallClassOnePct", fmt(100 * prior["severity_mix_overall"]["Class I"], 1))

    # ---- simulator -------------------------------------------------------
    r = sim["realised"]
    mac("simBatches", num(sim["n_batches"]))
    mac("simFeatures", num(sim["n_record_features"]))
    mac("simDurationH", fmt(sim["duration_h"], 0))
    mac("simSampleH", fmt(sim["sample_interval_h"], 0))
    mac("simInjected", num(r["n_injected_failures"]))
    mac("simRejected", num(r["n_rejected_by_spec"]))
    mac("simRejectRatePct", fmt(100 * r["reject_rate"], 2))
    mac("simObservableFail", num(r["n_trajectory_observable_failures"]))
    mac("simUnobservableFail", num(r["n_trajectory_unobservable_failures"]))
    mac("simSpecPercentile", fmt(sim["cqa_spec"]["_calibration"]["tail_percentile"], 2))
    mac("simSpecCalBatches", num(sim["cqa_spec"]["_calibration"]["n_nominal_batches"]))

    # ---- monitoring ------------------------------------------------------
    rows = mon["per_alpha"]
    pick = min(rows, key=lambda x: abs(x["false_alarm_rate_nominal"]["mean"] - 0.07))
    mac("monAlpha", fmt(pick["alpha"], 3))
    mac("monFA", pm(pick["false_alarm_rate_nominal"]))
    mac("monFAfold", fmt(pick["false_alarm_rate_nominal"]["mean"] / pick["alpha"], 0))
    mac("monDetObs", pm(pick["detection_rate_observable_modes"]))
    mac("monDetUnobs", pm(pick["detection_rate_unobservable_modes"]))
    mac("monTTD", pm(pick["median_time_to_detection_h_observable"], 0))
    mac("monNObsFail", num(pick["n_observable_fail"]["mean"]))
    mac("monNUnobsFail", num(pick["n_unobservable_fail"]["mean"]))
    rd = mon["readiness"]
    mac("readyAUROC", pm(rd["auroc"]))
    mac("readyAUPRC", pm(rd["auprc"]))
    mac("readyPrev", fmt(rd["auprc_baseline_prevalence"]["mean"], 3))
    mac("readyLift", pm(rd["auprc_lift_over_prevalence"], 1))
    mac("nSeeds", num(mon["n_seeds"]))
    mac("nMethods", num(len(METHODS)))

    # ---- cross-corpus agreement over the primary metric set --------------
    def a(metric, field, nd=2):
        return fmt(AGREE[metric][field], nd) if metric in AGREE else "n/a"

    short = {"fid_c2st_auc": "Ctst", "fid_ks_mean": "Ks", "fid_corr_mae": "Corr",
             "fid_pca_overlap": "Pca", "util_auroc": "Auroc", "util_auprc": "Auprc",
             "priv_dcr_ratio": "Dcr", "priv_mia_auc": "Mia"}
    for k, sh in short.items():
        mac(f"agree{sh}Rho", a(k, "spearman_rho"))
        mac(f"agree{sh}P", a(k, "spearman_p"))

    # per-seed correlations: ten independent estimates per metric
    for k, sh in short.items():
        d = CI.get(k, {})
        if d.get("n_usable_seeds"):
            mac(f"ci{sh}Rho", rf"{d['rho_mean']:.2f}\,$\pm$\,{d['rho_sd']:.2f}")
            mac(f"ci{sh}Lo", fmt(d["rho_p05"], 2))
            mac(f"ci{sh}Hi", fmt(d["rho_p95"], 2))
            mac(f"sep{sh}Sim", fmt(d["separation_sim"], 1))
            mac(f"sep{sh}Real", fmt(d["separation_real"], 1))
    fp = [v["rho_mean"] for v in CI.values() if v.get("axis") in ("fidelity", "privacy")]
    ut = [v["rho_mean"] for v in CI.values() if v.get("axis") == "utility"]
    mac("ciFidPrivLo", fmt(min(fp), 2))
    mac("ciFidPrivHi", fmt(max(fp), 2))
    mac("ciUtilLo", fmt(min(ut), 2))
    mac("ciUtilHi", fmt(max(ut), 2))
    mac("ciContrastMedFid", fmt(CON["median_fidelity_privacy"], 2))
    mac("ciContrastMedUtil", fmt(CON["median_utility"], 2))
    mac("ciContrastP", ("<0.001" if CON["p"] < 1e-3 else fmt(CON["p"], 3)))
    mac("ciNFidPriv", num(CON["n_fidelity_privacy"]))
    mac("ciNUtil", num(CON["n_utility"]))

    rho = {k: AGREE[k]["spearman_rho"] for k, *_ in PRIMARY if k in AGREE}
    fid_priv = [v for k, v in rho.items() if not k.startswith("util_")]
    util = [v for k, v in rho.items() if k.startswith("util_")]
    mac("agreeMedianRho", fmt(float(np.median(list(rho.values()))), 2))
    mac("agreeNMetrics", num(len(rho)))
    mac("agreeFidPrivMin", fmt(min(fid_priv), 2))
    mac("agreeFidPrivMax", fmt(max(fid_priv), 2))
    mac("agreeFidPrivN", num(len(fid_priv)))
    mac("agreeUtilMax", fmt(max(util), 2))
    mac("agreeNSignif", num(int(sum(1 for k in rho if AGREE[k]["spearman_p"] < 0.05))))

    # ---- per-generator headline values -----------------------------------
    def val(corpus, gen, metric, nd=3):
        d = S.get(corpus, {}).get(gen, {}).get(metric)
        return pm(d, nd) if d else r"\textemdash"

    def mean(corpus, gen, metric):
        d = S.get(corpus, {}).get(gen, {}).get(metric)
        return d["mean"] if d else float("nan")

    mac("bootDcrReal", val("REAL", "bootstrap", "priv_dcr_ratio"))
    mac("bootMiaReal", val("REAL", "bootstrap", "priv_mia_auc"))
    mac("bootCopyReal", val("REAL", "bootstrap", "priv_exact_copy_fraction"))
    mac("bootCtstReal", val("REAL", "bootstrap", "fid_c2st_auc"))

    # train-real/test-real baselines: the ceiling each corpus offers
    for corpus, sh in [("REAL", "Real"), ("SIM", "Sim")]:
        sub = raw[raw.corpus == corpus]
        mac(f"trtrAuprc{sh}", rf"{sub.util_trtr_auprc.mean():.3f}\,$\pm$\,{sub.util_trtr_auprc.std():.3f}")
        mac(f"trtrAuroc{sh}", rf"{sub.util_trtr_auroc.mean():.3f}\,$\pm$\,{sub.util_trtr_auroc.std():.3f}")
        mac(f"prev{sh}", fmt(sub.train_positive_rate.mean(), 3))

    def best(corpus, metric, lower=True):
        v = {g: mean(corpus, g, metric) for g in METHODS}
        v = {k: x for k, x in v.items() if x == x}
        return (min if lower else max)(v, key=v.get)

    for metric, sh, lower in [("fid_c2st_auc", "Ctst", True),
                              ("fid_corr_mae", "Corr", True),
                              ("util_auprc", "Util", False),
                              ("fid_pca_overlap", "Pca", False)]:
        for corpus, cs in [("REAL", "Real"), ("SIM", "Sim")]:
            g = best(corpus, metric, lower)
            mac(f"best{sh}{cs}", g)
            mac(f"best{sh}{cs}Val", val(corpus, g, metric))

    # utility of the best method relative to its corpus's own real-data ceiling
    for corpus, cs in [("REAL", "Real"), ("SIM", "Sim")]:
        g = best(corpus, "util_auprc", lower=False)
        sub = raw[raw.corpus == corpus]
        mac(f"bestUtilRetain{cs}",
            fmt(mean(corpus, g, "util_auprc") / sub.util_trtr_auprc.mean(), 2))

    # ---- minority-class reproduction -------------------------------------
    spr = raw.groupby(["corpus", "generator"])["synth_positive_rate"].mean()
    for corpus, cs in [("REAL", "Real"), ("SIM", "Sim")]:
        tp = raw[raw.corpus == corpus].train_positive_rate.mean()
        for g in METHODS:
            mac(f"spr{g.capitalize()}{cs}", fmt(spr[(corpus, g)], 4))
            mac(f"sprRatio{g.capitalize()}{cs}", fmt(spr[(corpus, g)] / tp, 2))
    deg = raw.groupby(["corpus", "generator"])["util_degenerate"].mean()
    mac("vaeDegenRealPct", fmt(100 * deg[("REAL", "vae")], 0))
    mac("vaeDegenRealSeeds", num(int(raw[(raw.corpus == "REAL") &
                                         (raw.generator == "vae")].util_auprc.notna().sum())))

    OUT_NUM.write_text("% AUTO-GENERATED by src/06_tables.py -- do not edit.\n"
                       + "\n".join(_macros) + "\n", encoding="utf-8")
    print(f"wrote {OUT_NUM} ({len(_macros)} macros)")

    # ---- tables ----------------------------------------------------------
    HDR = "% AUTO-GENERATED by src/06_tables.py -- do not edit."

    T = [HDR, r"\begin{tabular}{ll" + "c" * len(PRIMARY) + "}", r"\toprule"]
    T.append(" & ".join(["corpus", "generator"] + [p[1] for p in PRIMARY]) + r" \\")
    T.append(r"\midrule")
    for corpus, label in [("REAL", "SECOM (real)"), ("SIM", "simulated")]:
        for i, g in enumerate(["bootstrap"] + METHODS):
            cells = [val(corpus, g, m) for m, *_ in PRIMARY]
            gl = rf"\textit{{{g}}}$^\dagger$" if g == "bootstrap" else g
            T.append(" & ".join([label if i == 0 else "", gl] + cells) + r" \\")
        T.append(r"\midrule" if corpus == "REAL" else r"\bottomrule")
    T.append(r"\end{tabular}")
    OUT_MAIN.write_text("\n".join(T) + "\n", encoding="utf-8")
    print("wrote", OUT_MAIN)

    T = [HDR, r"\begin{tabular}{llccrl}", r"\toprule",
         r"axis & metric & $\rho$ (per seed) & 5--95\% & sep. & "
         r"selected method (sim / real) \\",
         r"\midrule"]
    for metric, label, axis, lower in PRIMARY:
        if metric not in AGREE or not CI.get(metric, {}).get("n_usable_seeds"):
            continue
        g, d = AGREE[metric], CI[metric]
        sv = dict(zip(METHODS, g["sim_values"]))
        rv = dict(zip(METHODS, g["real_values"]))
        ps = (min if lower else max)(sv, key=sv.get)
        pr = (min if lower else max)(rv, key=rv.get)
        tick = r"\;\checkmark" if ps == pr else ""
        T.append(f"{axis} & {label} & "
                 f"{d['rho_mean']:.2f}\\,$\\pm$\\,{d['rho_sd']:.2f} & "
                 f"[{d['rho_p05']:.2f}, {d['rho_p95']:.2f}] & "
                 f"{min(d['separation_sim'], d['separation_real']):.1f} & "
                 f"{ps} / {pr}{tick} " + r"\\")
    T += [r"\bottomrule", r"\end{tabular}"]
    OUT_AGREE.write_text("\n".join(T) + "\n", encoding="utf-8")
    print("wrote", OUT_AGREE)

    T = [HDR, r"\begin{tabular}{rcccc}", r"\toprule",
         r"nominal $\alpha$ & achieved false-alarm rate & detection: signature "
         r"& detection: none & median TTD (h) \\", r"\midrule"]
    for rec in rows:
        T.append(" & ".join([
            fmt(rec["alpha"], 3),
            pm(rec["false_alarm_rate_nominal"]),
            pm(rec["detection_rate_observable_modes"]),
            pm(rec["detection_rate_unobservable_modes"]),
            pm(rec["median_time_to_detection_h_observable"], 0),
        ]) + r" \\")
    T += [r"\bottomrule", r"\end{tabular}"]
    OUT_MON.write_text("\n".join(T) + "\n", encoding="utf-8")
    print("wrote", OUT_MON)


if __name__ == "__main__":
    main()
