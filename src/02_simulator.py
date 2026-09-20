"""02 — Mechanistic fed-batch GMP batch simulator.

This is the *ground-truth process* layer of the study. It is a deterministic
mechanistic model with stochastic batch-to-batch parameter variation, a sensor
layer, and injected failure modes. It is NOT a learned model and NOT the digital
twin: the twin (script 03) is a generative model trained on this simulator's
output. Keeping the two layers separate is what makes the circularity in the
literature explicit rather than hidden.

Kinetic structure follows the standard Bajpai-Reuss / PenSim family of fed-batch
penicillin models. Parameter values are taken from published ranges; every value
carries a `src` tag in PARAMS and is reproduced in the paper's parameter table.
No proprietary process data, parameter, or specification was used.

Outputs:
  data/interim/sim_batch_records.csv.gz  one row per batch (batch-record view)
  data/interim/sim_trajectories.npz      full sampled trajectories
  results/simulator_spec.json            parameters, spec limits, failure prior

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

from common import INTERIM, RESULTS, RAW, write_json, rng

# --------------------------------------------------------------------------
# Process parameters. `src` records where the nominal value or range comes from;
# `cv` is the batch-to-batch coefficient of variation applied to that parameter.
# --------------------------------------------------------------------------
PARAMS = {
    "mu_max":  dict(value=0.092, unit="1/h",      cv=0.08, src="fed-batch penicillin, Bajpai-Reuss family"),
    "Ks":      dict(value=0.15,  unit="g/L",      cv=0.10, src="Monod half-saturation, substrate"),
    "Ko":      dict(value=0.02,  unit="mmol/L",   cv=0.10, src="Monod half-saturation, dissolved oxygen"),
    "Yxs":     dict(value=0.45,  unit="g/g",      cv=0.06, src="biomass yield on substrate"),
    "Yps":     dict(value=0.90,  unit="g/g",      cv=0.08, src="product yield on substrate"),
    "qp_max":  dict(value=0.0062, unit="g/g/h",   cv=0.10, src="specific production rate"),
    "Kp":      dict(value=0.0002, unit="g/L",      cv=0.12, src="production half-saturation (substrate)"),
    "Ki_p":    dict(value=8.0,    unit="g/L",      cv=0.12, src="substrate inhibition of production"),
    "Kd":      dict(value=0.012, unit="1/h",      cv=0.10, src="biomass decay"),
    "m_s":     dict(value=0.014, unit="g/g/h",    cv=0.10, src="maintenance coefficient"),
    "K_hyd":   dict(value=0.0035, unit="1/h",     cv=0.12, src="product hydrolysis / degradation"),
    "kLa":     dict(value=85.0,  unit="1/h",      cv=0.12, src="oxygen transfer, large-scale stirred tank"),
    "DO_star": dict(value=0.25,  unit="mmol/L",   cv=0.03, src="saturation DO at operating T, P"),
    "Sf":      dict(value=420.0, unit="g/L",      cv=0.05, src="feed substrate concentration"),
    "V0":      dict(value=58000.0, unit="L",      cv=0.02, src="initial working volume, industrial scale"),
    "X0":      dict(value=0.12,  unit="g/L",      cv=0.20, src="inoculum density (seed-train variability)"),
    "S0":      dict(value=18.0,  unit="g/L",      cv=0.08, src="initial substrate"),
}

# Control setpoints and the alarm band a conventional univariate control chart
# would use. These are the *process* limits, distinct from the CQA specification
# limits below.
SETPOINTS = {"pH": 6.5, "T": 298.15, "DO_frac": 0.30}
T_END_H = 230.0
DT_H = 0.25          # integration step
SAMPLE_EVERY_H = 2.0  # sensor sampling interval

PHASES = [("lag", 0.0, 12.0), ("growth", 12.0, 45.0),
          ("production", 45.0, 200.0), ("harvest", 200.0, 230.0)]

# Critical quality attributes measured at release, with two-sided or one-sided
# specification limits. A batch is REJECTED if any CQA falls outside spec.
# Critical quality attributes measured at release. `side` gives which
# specification limit applies. IMPORTANT: no realism claim is made for this
# simulator. Its model form follows the published fed-batch structure cited in
# the paper; no parameter was fitted to plant data and its absolute titer scale
# is arbitrary. Specification limits are therefore
# derived from the simulator's OWN nominal (no-failure) distribution at the
# +/-3 sigma equivalent percentile, not asserted as real product specifications.
# See the Limitations section: this is a deliberate consequence of having no real
# batch data to calibrate against.
CQA_SIDE = {
    "titer_g_L":        dict(side="lo", unit="g/L",    observable=True),
    "purity_pct":       dict(side="lo", unit="%",      observable=True),
    "degradant_pct":    dict(side="hi", unit="%",      observable=True),
    "endotoxin_EU_mL":  dict(side="hi", unit="EU/mL",  observable=True),
    "bioburden_CFU_mL": dict(side="hi", unit="CFU/mL", observable=True),
    "particulates_ct":  dict(side="hi", unit="count",  observable=False),
    "water_pct":        dict(side="hi", unit="%",      observable=False),
    "ccit_leak":        dict(side="hi", unit="binary", observable=False),
}
SPEC_PERCENTILE = 0.27  # % in each tail; +/-3 sigma equivalent for a normal
CQA_SPEC: dict = {}     # filled by calibrate_specs()


def calibrate_specs(g, n_cal: int = 500) -> dict:
    """Set specification limits from the simulator's own nominal distribution."""
    vals = {k: [] for k in CQA_SIDE}
    for _ in range(n_cal):
        sim = simulate_batch(0, None, g)
        for k in CQA_SIDE:
            vals[k].append(sim["cqa"][k])
    spec = {}
    for k, meta in CQA_SIDE.items():
        a = np.asarray(vals[k])
        if k == "ccit_leak":
            spec[k] = dict(lo=None, hi=0.5, **meta)
            continue
        if meta["side"] == "lo":
            spec[k] = dict(lo=float(np.percentile(a, SPEC_PERCENTILE)), hi=None, **meta)
        else:
            spec[k] = dict(lo=None, hi=float(np.percentile(a, 100 - SPEC_PERCENTILE)), **meta)
    spec["_calibration"] = dict(n_nominal_batches=n_cal, tail_percentile=SPEC_PERCENTILE)
    return spec

# Failure modes the simulator can inject. `observable` records whether the mode
# leaves a signature in the process trajectory at all -- two modes deliberately
# do not, so that trajectory-based detection has a real, principled ceiling
# rather than an artificially perfect one.
PROCESS_MODES = {
    "sterility_microbial":         dict(observable=True),
    "potency_assay":               dict(observable=True),
    "impurity_degradant":          dict(observable=True),
    "cgmp_process_deviation":      dict(observable=True),
    "cross_contamination_foreign": dict(observable=False),
    "container_closure":           dict(observable=False),
}
# openFDA modes excluded from the simulator, with the reason. Recorded so the
# renormalisation of the empirical prior is auditable.
EXCLUDED_MODES = {
    "labeling_mixup": "packaging/labelling error, not a bioreactor process failure",
    "unapproved_undeclared": "regulatory/marketing status, not a process failure",
    "stability_shelf_life": "manifests post-release over shelf life, not at release",
    "dissolution_physical": "downstream solid-dose attribute, out of scope for this fed-batch model",
    "unclassified": "reason text not matched by the rule set",
}

# Nominal batch failure rate. NOT taken from the recall corpus: recall records
# count escapes, not batch rejections, so they cannot estimate a rejection rate.
# Set to match SECOM's observed fail rate so the two datasets are comparable in
# class balance, and reported as an assumption.
NOMINAL_FAILURE_RATE = 0.0664


def load_mode_prior() -> dict:
    """Renormalised failure-mode prior over process-relevant modes only."""
    prior = json.loads((RESULTS / "failure_mode_prior.json").read_text())
    freq = prior["mode_frequency"]
    kept = {k: freq[k] for k in PROCESS_MODES if k in freq}
    tot = sum(kept.values())
    return {k: v / tot for k, v in kept.items()}


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------

def draw_params(g: np.random.Generator) -> dict:
    p = {}
    for k, spec in PARAMS.items():
        p[k] = spec["value"] * (1.0 + spec["cv"] * g.standard_normal())
        p[k] = max(p[k], 1e-6)
    return p


def feed_profile(t: float, p: dict) -> float:
    """Exponential feed during the production phase, zero elsewhere (L/h)."""
    if t < 45.0 or t > 200.0:
        return 0.0
    return 150.0 * np.exp(0.0032 * (t - 45.0))


def simulate_batch(seed: int, mode: str | None, g: np.random.Generator) -> dict:
    p = draw_params(g)
    n = int(T_END_H / DT_H) + 1
    t = np.arange(n) * DT_H

    X = np.zeros(n); S = np.zeros(n); P = np.zeros(n); V = np.zeros(n)
    DO = np.zeros(n); pH = np.zeros(n); Temp = np.zeros(n); CO2 = np.zeros(n)
    Cont = np.zeros(n)  # contaminant biomass

    X[0], S[0], V[0] = p["X0"], p["S0"], p["V0"]
    DO[0] = p["DO_star"] * SETPOINTS["DO_frac"] / 0.30 * 0.30
    pH[0], Temp[0] = SETPOINTS["pH"], SETPOINTS["T"]

    # --- failure-mode setup -------------------------------------------------
    mu_scale, qp_scale, temp_excursion, ph_excursion = 1.0, 1.0, 0.0, 0.0
    contam_t0 = None
    if mode == "potency_assay":
        # chronic under-performance: low-yielding seed or degraded medium
        qp_scale = g.uniform(0.35, 0.65)
        mu_scale = g.uniform(0.80, 0.95)
    elif mode == "sterility_microbial":
        contam_t0 = g.uniform(25.0, 110.0)
    elif mode == "impurity_degradant":
        # sustained temperature excursion accelerating hydrolysis
        temp_excursion = g.uniform(1.6, 3.4)
    elif mode == "cgmp_process_deviation":
        ph_excursion = g.uniform(0.35, 0.75) * g.choice([-1.0, 1.0])

    ctrl_int_ph = 0.0
    ctrl_int_do = 0.0

    for i in range(n - 1):
        ti = t[i]
        # --- controllers (PI with actuator noise) --------------------------
        ph_sp = SETPOINTS["pH"] + (ph_excursion if 80.0 < ti < 140.0 else 0.0)
        err_ph = ph_sp - pH[i]
        ctrl_int_ph += err_ph * DT_H
        pH[i + 1] = pH[i] + (0.55 * err_ph + 0.02 * ctrl_int_ph) * DT_H \
            + 0.012 * g.standard_normal() * np.sqrt(DT_H)

        t_sp = SETPOINTS["T"] + (temp_excursion if 60.0 < ti < 190.0 else 0.0)
        Temp[i + 1] = Temp[i] + 0.45 * (t_sp - Temp[i]) * DT_H \
            + 0.02 * g.standard_normal() * np.sqrt(DT_H)

        # --- growth kinetics ------------------------------------------------
        f_ph = np.exp(-0.5 * ((pH[i] - 6.5) / 0.55) ** 2)
        f_T = np.exp(-0.5 * ((Temp[i] - 298.15) / 3.2) ** 2)
        mu = (mu_scale * p["mu_max"] * S[i] / (p["Ks"] + S[i])
              * DO[i] / (p["Ko"] + DO[i]) * f_ph * f_T)
        # Production follows Bajpai-Reuss form: a much lower half-saturation than
        # growth plus substrate inhibition, so the process keeps producing under
        # the substrate-limited fed-batch regime it is deliberately operated in.
        qp = (qp_scale * p["qp_max"]
              * S[i] / (p["Kp"] + S[i] + S[i] ** 2 / p["Ki_p"])
              * DO[i] / (p["Ko"] + DO[i]) * f_ph * f_T)

        F = feed_profile(ti, p)
        dil = F / V[i]

        # contaminant: logistic growth once introduced, competing for substrate
        if contam_t0 is not None and ti >= contam_t0:
            if Cont[i] <= 0:
                Cont[i] = 1e-4
            mu_c = 0.22 * S[i] / (0.05 + S[i]) * DO[i] / (0.01 + DO[i])
            Cont[i + 1] = max(Cont[i] + (mu_c * Cont[i] - dil * Cont[i]) * DT_H, 0.0)
        else:
            Cont[i + 1] = 0.0

        dX = mu * X[i] - p["Kd"] * X[i] - dil * X[i]
        dP = qp * X[i] - p["K_hyd"] * P[i] - dil * P[i]
        dS = (-mu * X[i] / p["Yxs"] - qp * X[i] / p["Yps"] - p["m_s"] * X[i]
              - 0.35 * Cont[i] + dil * (p["Sf"] - S[i]))
        dV = F - 0.9

        X[i + 1] = max(X[i] + dX * DT_H, 0.0)
        P[i + 1] = max(P[i] + dP * DT_H, 0.0)
        S[i + 1] = max(S[i] + dS * DT_H, 0.0)
        V[i + 1] = V[i] + dV * DT_H

        # --- dissolved oxygen ------------------------------------------------
        OUR = (0.42 * mu * X[i] + 0.0016 * X[i] + 0.010 * Cont[i])
        err_do = p["DO_star"] * SETPOINTS["DO_frac"] - DO[i]
        ctrl_int_do += err_do * DT_H
        kla_eff = p["kLa"] * float(np.clip(
            0.45 + 6.0 * err_do / p["DO_star"] + 0.35 * ctrl_int_do / p["DO_star"],
            0.02, 2.5))
        dDO = kla_eff * (p["DO_star"] - DO[i]) - OUR
        DO[i + 1] = float(np.clip(DO[i] + dDO * DT_H
                                  + 0.0012 * g.standard_normal() * np.sqrt(DT_H),
                                  0.0, p["DO_star"]))
        CO2[i + 1] = 92.0 * OUR + 0.02 * g.standard_normal()

    # --- CQAs at release ----------------------------------------------------
    titer = P[-1]
    contam_final = Cont[-1]
    thermal_load = np.trapezoid(np.maximum(Temp - 298.15, 0.0), t)

    degradant = 0.45 + 0.0048 * thermal_load + 0.18 * abs(g.standard_normal())
    endotoxin = 0.6 + 2.1 * contam_final + 0.25 * abs(g.standard_normal())
    bioburden = 0.4 + 9.0 * contam_final + 0.5 * abs(g.standard_normal())
    purity = 99.4 - degradant - 0.35 * contam_final - 0.12 * abs(g.standard_normal())
    particulates = 2600.0 + 420.0 * g.standard_normal()
    water = 0.42 + 0.09 * abs(g.standard_normal())
    ccit = 0.0

    if mode == "cross_contamination_foreign":
        particulates += g.uniform(3200.0, 6500.0)
    if mode == "container_closure":
        ccit = 1.0
        water += g.uniform(0.4, 0.9)

    cqa = dict(titer_g_L=titer, purity_pct=purity, degradant_pct=degradant,
               endotoxin_EU_mL=endotoxin, bioburden_CFU_mL=bioburden,
               particulates_ct=particulates, water_pct=water, ccit_leak=ccit)

    ooS = []
    for k, spec in CQA_SPEC.items():
        if k.startswith("_"):
            continue
        v = cqa[k]
        if spec["lo"] is not None and v < spec["lo"]:
            ooS.append(k)
        if spec["hi"] is not None and v > spec["hi"]:
            ooS.append(k)
    disposition = 1 if ooS else 0  # 1 = reject

    # --- sensor layer: sample, add noise, quantise, drop -------------------
    step = int(SAMPLE_EVERY_H / DT_H)
    idx = np.arange(0, n, step)
    traj = np.vstack([X[idx], S[idx], P[idx], DO[idx], pH[idx],
                      Temp[idx], CO2[idx], V[idx]]).T
    noise_sd = np.array([0.06, 0.25, 0.10, 0.004, 0.012, 0.05, 0.30, 45.0])
    traj = traj + g.standard_normal(traj.shape) * noise_sd
    drift = np.outer(np.linspace(0, 1, traj.shape[0]),
                     g.standard_normal(traj.shape[1]) * noise_sd * 0.4)
    traj = traj + drift
    drop = g.random(traj.shape) < 0.004
    traj[drop] = np.nan

    return dict(traj=traj, t=t[idx], cqa=cqa, disposition=disposition,
                oos=ooS, mode=mode or "none", params=p,
                contam_t0=contam_t0)


VAR_NAMES = ["biomass", "substrate", "product", "DO", "pH", "temp", "offgas_CO2", "volume"]


def batch_record(sim: dict) -> dict:
    """Collapse a batch into the tabular batch-record view a reviewer or a
    release package would actually see: phase-wise summary statistics of each
    process variable, plus the release CQAs."""
    rec: dict = {}
    t = sim["t"]
    A = sim["traj"]
    for j, name in enumerate(VAR_NAMES):
        col = A[:, j]
        for ph, lo, hi in PHASES:
            m = (t >= lo) & (t < hi)
            seg = col[m]
            seg = seg[~np.isnan(seg)]
            if seg.size == 0:
                mu = sd = mn = mx = np.nan
            else:
                mu, sd, mn, mx = seg.mean(), seg.std(), seg.min(), seg.max()
            rec[f"{name}_{ph}_mean"] = mu
            rec[f"{name}_{ph}_sd"] = sd
            rec[f"{name}_{ph}_min"] = mn
            rec[f"{name}_{ph}_max"] = mx
    rec.update(sim["cqa"])
    rec["disposition"] = sim["disposition"]
    rec["failure_mode"] = sim["mode"]
    return rec


def main(n_batches: int = 2000, seed: int = 20260907) -> None:
    global CQA_SPEC
    g = rng(seed)
    CQA_SPEC = calibrate_specs(rng(seed + 99))
    prior = load_mode_prior()
    modes = list(prior); probs = np.array([prior[m] for m in modes])

    records, trajs, tvec = [], [], None
    n_fail_target = 0
    for b in range(n_batches):
        fails = g.random() < NOMINAL_FAILURE_RATE
        mode = str(g.choice(modes, p=probs)) if fails else None
        n_fail_target += int(fails)
        sim = simulate_batch(b, mode, g)
        records.append(batch_record(sim))
        trajs.append(sim["traj"])
        tvec = sim["t"]

    df = pd.DataFrame(records)
    df.insert(0, "batch_id", np.arange(len(df)))
    df.to_csv(INTERIM / "sim_batch_records.csv.gz", index=False)
    np.savez_compressed(INTERIM / "sim_trajectories.npz",
                        traj=np.stack(trajs), t=tvec,
                        disposition=df["disposition"].to_numpy(),
                        mode=df["failure_mode"].to_numpy().astype(str),
                        var_names=np.array(VAR_NAMES))

    observable_modes = [m for m, s in PROCESS_MODES.items() if s["observable"]]
    n_obs = int(df["failure_mode"].isin(observable_modes).sum())
    n_unobs = int((~df["failure_mode"].isin(observable_modes + ["none"])).sum())

    write_json("simulator_spec.json", {
        "n_batches": n_batches,
        "master_seed": seed,
        "params": PARAMS,
        "setpoints": SETPOINTS,
        "duration_h": T_END_H,
        "integration_step_h": DT_H,
        "sample_interval_h": SAMPLE_EVERY_H,
        "phases": [{"name": p, "start_h": a, "end_h": b} for p, a, b in PHASES],
        "cqa_spec": CQA_SPEC,
        "process_modes": PROCESS_MODES,
        "excluded_openfda_modes": EXCLUDED_MODES,
        "mode_prior_renormalised": prior,
        "nominal_failure_rate_assumed": NOMINAL_FAILURE_RATE,
        "nominal_failure_rate_note": (
            "Set to SECOM's observed fail rate so the two corpora are comparable in "
            "class balance. It is NOT estimated from the recall corpus: recalls count "
            "market escapes, not batch rejections."),
        "realised": {
            "n_injected_failures": n_fail_target,
            "n_rejected_by_spec": int(df["disposition"].sum()),
            "reject_rate": float(df["disposition"].mean()),
            "n_trajectory_observable_failures": n_obs,
            "n_trajectory_unobservable_failures": n_unobs,
            "mode_counts": df["failure_mode"].value_counts().to_dict(),
        },
        "n_record_features": int(df.shape[1] - 3),
        "outputs": ["data/interim/sim_batch_records.csv.gz",
                    "data/interim/sim_trajectories.npz"],
    })
    print(df["failure_mode"].value_counts().to_string())
    print("reject rate %.4f over %d batches; %d features per batch record"
          % (df["disposition"].mean(), len(df), df.shape[1] - 3))


if __name__ == "__main__":
    main()
