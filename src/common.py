"""Shared utilities: deterministic seeding, IO paths, JSON writing.

Every script in src/ writes its numbers to results/ as JSON so that each value in
the paper is traceable to the script that produced it.

Independent work. Carried out on personal time and equipment, not connected to
the author's employment. No proprietary, confidential or internal data of any
organization was used. All data is public.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

for _d in (INTERIM, RESULTS, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)

# Seeds used everywhere a stochastic result is reported. Every headline number is
# the mean +/- sd over these seeds; no single-run number is reported as a finding.
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def env_record() -> dict:
    import sklearn
    import scipy
    import pandas
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pandas.__version__,
        "scikit_learn": sklearn.__version__,
        "seeds": SEEDS,
    }


def write_json(name: str, payload: dict) -> Path:
    """Write a results JSON, stamping the environment for reproducibility."""
    out = RESULTS / name
    payload = dict(payload)
    payload["_env"] = env_record()
    payload["_script"] = Path(sys.argv[0]).name
    out.write_text(json.dumps(payload, indent=2, default=_default), encoding="utf-8")
    print(f"wrote {out}")
    return out


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def mean_sd(values) -> dict:
    a = np.asarray(list(values), dtype=float)
    return {
        "mean": float(a.mean()),
        "sd": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "n": int(a.size),
        "values": [float(x) for x in a],
    }
