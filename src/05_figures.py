"""05 — All figures. Vector PDF, Okabe-Ito palette, grayscale-safe.

Reads only from results/ and data/interim/, so every figure is reproducible from
the recorded numbers without re-running any experiment.

Independent work. Carried out on personal time and equipment, not connected to
the author's employment. No proprietary, confidential or internal data of any
organization was used. All data is public.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))  # run from anywhere

from common import FIGURES, INTERIM, RESULTS
from generators import METHODS

# Okabe-Ito, colour-vision-deficiency safe.
OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
      "vermillion": "#D55E00", "purple": "#CC79A7", "sky": "#56B4E9",
      "yellow": "#F0E442", "black": "#000000", "grey": "#7F7F7F"}
GEN_COLOR = dict(zip(METHODS, [OI["grey"], OI["blue"], OI["green"],
                               OI["orange"], OI["purple"], OI["vermillion"]]))

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def save(fig, name):
    p = FIGURES / f"{name}.pdf"
    fig.savefig(p)
    plt.close(fig)
    print("wrote", p)


# --------------------------------------------------------------------------
def fig1_design():
    """Two-layer design, and exactly where the circularity enters."""
    fig, ax = plt.subplots(figsize=(6.6, 2.7))
    ax.set_xlim(0, 100); ax.set_ylim(0, 44); ax.axis("off")
    ax.grid(False)

    def box(x, y, w, h, text, fc, ec="#333333", ls="-"):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.6,rounding_size=1.4",
                                    fc=fc, ec=ec, lw=1.0, ls=ls))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.2)

    def arrow(x1, y1, x2, y2, style="-|>", color="#333333", ls="-"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                     mutation_scale=9, lw=1.0, color=color,
                                     linestyle=ls, shrinkA=2, shrinkB=2))

    box(1, 26, 20, 13, "Real anchors\nSECOM (1,567 runs)\nopenFDA recalls (17,898)", "#DCEAF7")
    box(28, 26, 20, 13, "Mechanistic\nbatch simulator\n(ground-truth process)", "#E8F4EC")
    box(55, 26, 20, 13, "Generative twin\ncopula / GMM /\nVAE / WGAN", "#FBEEDC")
    box(81, 26, 17, 13, "Synthetic\nbatch records", "#F6E4EE")

    arrow(21, 32.5, 28, 32.5); ax.text(24.5, 23.5, "failure-mode\nprior", ha="center", fontsize=6.2)
    arrow(48, 32.5, 55, 32.5); ax.text(51.5, 34.6, "trains", ha="center", fontsize=6.2)
    arrow(75, 32.5, 81, 32.5)

    box(28, 4, 47, 12,
        "Evaluation: fidelity  |  utility  |  privacy\n"
        "run identically on the real corpus and on the simulated corpus",
        "#F2F2F2")
    arrow(89, 26, 89, 10.5); arrow(89, 10.5, 75, 10.5)
    arrow(11, 26, 11, 10.5); arrow(11, 10.5, 28, 10.5)

    # the circular path
    arrow(38, 26, 38, 16.5, color=OI["vermillion"], ls="--")
    ax.text(40.5, 20.5, "the circular path: a twin scored only against the simulator\n"
                        "that produced its training data", color=OI["vermillion"],
            fontsize=6.4, va="center")
    save(fig, "fig1_design")


# --------------------------------------------------------------------------
def fig2_failure_prior():
    prior = json.loads((RESULTS / "failure_mode_prior.json").read_text())
    spec = json.loads((RESULTS / "simulator_spec.json").read_text())
    emp = prior["mode_frequency"]
    renorm = spec["mode_prior_renormalised"]
    order = sorted(emp, key=emp.get, reverse=True)
    kept = set(renorm)

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5),
                             gridspec_kw={"width_ratios": [1.55, 1]})
    ax = axes[0]
    vals = [emp[k] for k in order]
    cols = [OI["blue"] if k in kept else OI["grey"] for k in order]
    ax.barh(range(len(order)), vals, color=cols, height=0.7)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([k.replace("_", " ") for k in order])
    ax.invert_yaxis()
    ax.set_xlabel("share of 17,898 real FDA drug recalls")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(fc=OI["blue"], label="retained in the simulator"),
                       Patch(fc=OI["grey"], label="excluded: not a process failure")],
              frameon=False, loc="lower right", fontsize=6.2)

    ax = axes[1]
    ks = sorted(renorm, key=renorm.get, reverse=True)
    obs = spec["process_modes"]
    cols = [OI["green"] if obs[k]["observable"] else OI["vermillion"] for k in ks]
    ax.barh(range(len(ks)), [renorm[k] for k in ks], color=cols, height=0.7)
    ax.set_yticks(range(len(ks)))
    ax.set_yticklabels([k.replace("_", " ") for k in ks])
    ax.invert_yaxis()
    ax.set_xlabel("renormalised prior")
    ax.legend(handles=[Patch(fc=OI["green"], label="leaves a trajectory signature"),
                       Patch(fc=OI["vermillion"], label="leaves none")],
              frameon=False, loc="center right", fontsize=6.2)
    save(fig, "fig2_failure_prior")


# --------------------------------------------------------------------------
def fig3_trajectories():
    z = np.load(INTERIM / "sim_trajectories.npz", allow_pickle=True)
    A, t, mode = z["traj"], z["t"], z["mode"].astype(str)
    names = ["biomass (g/L)", "substrate (g/L)", "product (g/L)", "DO (mmol/L)"]
    fig, axes = plt.subplots(1, 4, figsize=(6.6, 1.9))
    nom = np.where(mode == "none")[0][:60]
    con = np.where(mode == "sterility_microbial")[0][:12]
    for j, ax in enumerate(axes):
        for i in nom:
            ax.plot(t, A[i, :, j], color=OI["grey"], lw=0.3, alpha=0.35)
        for i in con:
            ax.plot(t, A[i, :, j], color=OI["vermillion"], lw=0.6, alpha=0.9)
        ax.set_xlabel("h"); ax.set_ylabel(names[j])
    axes[0].plot([], [], color=OI["grey"], lw=1, label="nominal")
    axes[0].plot([], [], color=OI["vermillion"], lw=1, label="contaminated")
    axes[0].legend(frameon=False, loc="upper left")
    save(fig, "fig3_trajectories")


# --------------------------------------------------------------------------
def _summary():
    return json.loads((RESULTS / "evaluation_summary.json").read_text())


PANELS = [
    ("fid_c2st_auc",   "C2ST AUC  (0.5 = ideal)", False),
    ("fid_ks_mean",    "mean KS  (lower better)", False),
    ("fid_corr_mae",   "correlation MAE  (lower better)", False),
    ("fid_pca_overlap","PCA subspace overlap  (higher better)", True),
    ("util_auprc",     "TSTR AUPRC on real test  (higher better)", True),
    ("priv_dcr_ratio", "DCR ratio train/holdout  (1 = no leakage)", True),
]


def fig4_cross_corpus():
    S = _summary()["per_corpus"]
    fig, axes = plt.subplots(2, 3, figsize=(6.6, 4.0))
    for ax, (metric, label, _hib) in zip(axes.ravel(), PANELS):
        xs, ys, cs, ok = [], [], [], []
        for g in METHODS:
            a = S.get("SIM", {}).get(g, {}).get(metric)
            b = S.get("REAL", {}).get(g, {}).get(metric)
            if not a or not b:
                continue
            xs.append(a); ys.append(b); cs.append(GEN_COLOR[g]); ok.append(g)
        if not xs:
            ax.set_visible(False); continue
        for a, b, c, g in zip(xs, ys, cs, ok):
            ax.errorbar(a["mean"], b["mean"], xerr=a["sd"], yerr=b["sd"],
                        fmt="o", ms=4, color=c, ecolor=c, elinewidth=0.8,
                        capsize=1.5, label=g)
        lo = min(min(a["mean"] for a in xs), min(b["mean"] for b in ys))
        hi = max(max(a["mean"] for a in xs), max(b["mean"] for b in ys))
        pad = 0.08 * (hi - lo + 1e-9)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=OI["black"],
                lw=0.6, ls=":", zorder=0)
        ax.set_xlabel("simulated corpus"); ax.set_ylabel("real corpus (SECOM)")
        ax.set_title(label, fontsize=7)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, ncol=6, frameon=False, loc="lower center",
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout()
    save(fig, "fig4_cross_corpus")


def fig5_tradeoff():
    S = _summary()["per_corpus"]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6), sharey=True)
    for ax, corpus in zip(axes, ["REAL", "SIM"]):
        for g in ["bootstrap"] + METHODS:
            d = S.get(corpus, {}).get(g, {})
            if "fid_c2st_auc" not in d or "priv_dcr_ratio" not in d:
                continue
            c = GEN_COLOR.get(g, OI["black"])
            mk = "*" if g == "bootstrap" else "o"
            ax.errorbar(d["fid_c2st_auc"]["mean"], d["priv_dcr_ratio"]["mean"],
                        xerr=d["fid_c2st_auc"]["sd"], yerr=d["priv_dcr_ratio"]["sd"],
                        fmt=mk, ms=7 if mk == "*" else 4.5, color=c, ecolor=c,
                        elinewidth=0.8, capsize=1.5, label=g)
            ax.annotate(g, (d["fid_c2st_auc"]["mean"], d["priv_dcr_ratio"]["mean"]),
                        textcoords="offset points", xytext=(4, 3), fontsize=6)
        ax.axhline(1.0, color=OI["black"], lw=0.6, ls=":")
        ax.set_xlabel("C2ST AUC   (0.5 = indistinguishable)")
        ax.set_title(("real corpus (SECOM)" if corpus == "REAL"
                      else "simulated corpus"), fontsize=7.5)
    axes[0].set_ylabel("DCR ratio  (1 = no memorisation)")
    fig.tight_layout()
    save(fig, "fig5_tradeoff")


# --------------------------------------------------------------------------
def fig6_monitoring():
    m = json.loads((RESULTS / "monitoring.json").read_text())
    fa = [r["false_alarm_rate_nominal"] for r in m["per_alpha"]]
    obs = [r["detection_rate_observable_modes"] for r in m["per_alpha"]]
    un = [r["detection_rate_unobservable_modes"] for r in m["per_alpha"]]
    ttd = [r["median_time_to_detection_h_observable"] for r in m["per_alpha"]]

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5))
    ax = axes[0]
    x = [v["mean"] for v in fa]
    ax.errorbar(x, [v["mean"] for v in obs],
                xerr=[v["sd"] for v in fa], yerr=[v["sd"] for v in obs],
                fmt="o-", ms=3.5, lw=1.0, color=OI["green"], capsize=1.5,
                label="modes with a trajectory signature")
    ax.errorbar(x, [v["mean"] for v in un],
                xerr=[v["sd"] for v in fa], yerr=[v["sd"] for v in un],
                fmt="s--", ms=3.5, lw=1.0, color=OI["vermillion"], capsize=1.5,
                label="modes without one")
    ax.plot([0, max(x)], [0, max(x)], color=OI["black"], lw=0.6, ls=":",
            label="chance")
    ax.set_xlabel("batch-level false-alarm rate on nominal batches")
    ax.set_ylabel("detection rate")
    ax.legend(frameon=False, loc="upper left")

    ax = axes[1]
    ax.errorbar(x, [v["mean"] for v in ttd],
                xerr=[v["sd"] for v in fa], yerr=[v["sd"] for v in ttd],
                fmt="o-", ms=3.5, lw=1.0, color=OI["blue"], capsize=1.5)
    ax.axhline(230, color=OI["grey"], lw=0.6, ls=":")
    ax.text(max(x) * 0.55, 233, "batch end (230 h)", fontsize=6, color=OI["grey"])
    ax.set_xlabel("batch-level false-alarm rate")
    ax.set_ylabel("median time to first alarm (h)")
    fig.tight_layout()
    save(fig, "fig6_monitoring")


def main():
    fig1_design()
    fig2_failure_prior()
    fig3_trajectories()
    fig6_monitoring()
    if (RESULTS / "evaluation_summary.json").exists():
        fig4_cross_corpus()
        fig5_tradeoff()
    else:
        print("evaluation_summary.json missing -- skipping figs 4 and 5")


if __name__ == "__main__":
    main()
