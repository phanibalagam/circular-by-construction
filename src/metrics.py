"""Fidelity, utility and privacy metrics.

Every metric here operates on standardised matrices and returns a scalar, so
that the same protocol can be applied unchanged to the real corpus (SECOM) and
to the simulated corpus. That symmetry is the point of the study: it is what
lets us ask whether conclusions drawn on simulated data transfer to real data.

Independent work. Carried out on personal time and equipment, not connected to
the author's employment. No proprietary, confidential or internal data of any
organization was used. All data is public.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors


# --------------------------------------------------------------------------
# Fidelity -- marginals
# --------------------------------------------------------------------------

def marginal_ks(real: np.ndarray, synth: np.ndarray) -> dict:
    """Two-sample KS statistic per column. Lower is better (0 = identical)."""
    stats = [ks_2samp(real[:, j], synth[:, j]).statistic
             for j in range(real.shape[1])]
    a = np.asarray(stats)
    return {"ks_mean": float(a.mean()), "ks_median": float(np.median(a)),
            "ks_p90": float(np.percentile(a, 90)), "ks_max": float(a.max())}


def marginal_wasserstein(real: np.ndarray, synth: np.ndarray) -> float:
    w = [wasserstein_distance(real[:, j], synth[:, j])
         for j in range(real.shape[1])]
    return float(np.mean(w))


# --------------------------------------------------------------------------
# Fidelity -- joints
# --------------------------------------------------------------------------

def correlation_delta(real: np.ndarray, synth: np.ndarray) -> dict:
    """Mean absolute difference between the two correlation matrices, and the
    same for Spearman rank correlation (which catches monotone non-linear
    dependence that Pearson misses)."""
    def _corr(A):
        C = np.corrcoef(A, rowvar=False)
        return np.nan_to_num(C, nan=0.0)

    def _rank(A):
        R = np.argsort(np.argsort(A, axis=0), axis=0).astype(float)
        return _corr(R)

    iu = np.triu_indices(real.shape[1], k=1)
    dp = np.abs(_corr(real)[iu] - _corr(synth)[iu])
    ds = np.abs(_rank(real)[iu] - _rank(synth)[iu])
    return {"corr_mae": float(dp.mean()), "corr_max": float(dp.max()),
            "spearman_mae": float(ds.mean())}


def pca_subspace_overlap(real: np.ndarray, synth: np.ndarray, k: int = 10) -> float:
    """Mean squared canonical correlation between the top-k PCA subspaces.
    1.0 = the synthetic data spans the same dominant directions of variation."""
    def _basis(A):
        Ac = A - A.mean(0)
        _, _, vt = np.linalg.svd(Ac, full_matrices=False)
        return vt[:k].T
    U, V = _basis(real), _basis(synth)
    s = np.linalg.svd(U.T @ V, compute_uv=False)
    return float((s ** 2).mean())


def c2st_auc(real: np.ndarray, synth: np.ndarray, seed: int = 0,
             n_splits: int = 3) -> float:
    """Classifier two-sample test. A classifier is trained to tell real from
    synthetic; cross-validated AUC of 0.5 means indistinguishable, 1.0 means
    trivially separable. This is the single most discriminating fidelity metric
    and the one a generative-modelling reviewer will look for."""
    m = min(len(real), len(synth))
    rs = np.random.default_rng(seed)
    R = real[rs.choice(len(real), m, replace=False)]
    S = synth[rs.choice(len(synth), m, replace=False)]
    X = np.vstack([R, S])
    y = np.r_[np.zeros(m), np.ones(m)]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in skf.split(X, y):
        clf = HistGradientBoostingClassifier(max_iter=120, random_state=seed)
        clf.fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
    return float(np.mean(aucs))


# --------------------------------------------------------------------------
# Utility
# --------------------------------------------------------------------------

def _fit_clf(X, y, seed):
    # One learner, fixed in advance and applied identically to every generator
    # and to the real-data baseline, so that a utility difference is attributable
    # to the generator rather than to model selection.
    clf = HistGradientBoostingClassifier(max_iter=200, random_state=seed)
    clf.fit(X, y)
    return clf


def tstr(X_syn, y_syn, X_real_test, y_real_test, seed: int = 0) -> dict:
    """Train on synthetic, test on real."""
    if len(np.unique(y_syn)) < 2 or y_syn.sum() < 5:
        return {"auroc": float("nan"), "auprc": float("nan"), "degenerate": True}
    clf = _fit_clf(X_syn, y_syn, seed)
    p = clf.predict_proba(X_real_test)[:, 1]
    return {"auroc": float(roc_auc_score(y_real_test, p)),
            "auprc": float(average_precision_score(y_real_test, p)),
            "degenerate": False}


def trtr(X_tr, y_tr, X_te, y_te, seed: int = 0) -> dict:
    clf = _fit_clf(X_tr, y_tr, seed)
    p = clf.predict_proba(X_te)[:, 1]
    return {"auroc": float(roc_auc_score(y_te, p)),
            "auprc": float(average_precision_score(y_te, p))}


# --------------------------------------------------------------------------
# Privacy / memorisation
# --------------------------------------------------------------------------

def dcr_nndr(synth: np.ndarray, train: np.ndarray, holdout: np.ndarray) -> dict:
    """Distance to closest record and nearest-neighbour distance ratio.

    The diagnostic quantity is not DCR to the training set on its own -- a
    generator that merely produces typical records will sit close to everything.
    It is the *ratio* of median DCR to train against median DCR to a holdout
    drawn from the same distribution. A ratio near 1 means the generator is no
    closer to its training data than to unseen data of the same kind. A ratio
    well below 1 is evidence of memorisation.
    """
    nn_tr = NearestNeighbors(n_neighbors=2).fit(train)
    nn_ho = NearestNeighbors(n_neighbors=1).fit(holdout)
    d_tr, _ = nn_tr.kneighbors(synth)
    d_ho, _ = nn_ho.kneighbors(synth)
    dcr_train = d_tr[:, 0]
    dcr_hold = d_ho[:, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        nndr = np.where(d_tr[:, 1] > 0, d_tr[:, 0] / d_tr[:, 1], 1.0)
    ratio = float(np.median(dcr_train) / max(np.median(dcr_hold), 1e-12))
    return {
        "dcr_train_median": float(np.median(dcr_train)),
        "dcr_holdout_median": float(np.median(dcr_hold)),
        "dcr_ratio": ratio,
        "nndr_median": float(np.median(nndr)),
        "exact_copy_fraction": float((dcr_train < 1e-9).mean()),
    }


def membership_inference_auc(synth: np.ndarray, train: np.ndarray,
                             holdout: np.ndarray) -> float:
    """Distance-to-synthetic membership attack.

    For every record, compute its distance to the nearest synthetic record. The
    attacker predicts "was in the generator's training set" for records that lie
    closer to the synthetic sample. AUC is reported over train (members) versus
    holdout (non-members); 0.5 means the attack learns nothing.
    """
    n = min(len(train), len(holdout))
    rs = np.random.default_rng(0)
    T = train[rs.choice(len(train), n, replace=False)]
    H = holdout[rs.choice(len(holdout), n, replace=False)]
    nn = NearestNeighbors(n_neighbors=1).fit(synth)
    d_t, _ = nn.kneighbors(T)
    d_h, _ = nn.kneighbors(H)
    score = -np.r_[d_t[:, 0], d_h[:, 0]]  # closer => more likely a member
    y = np.r_[np.ones(n), np.zeros(n)]
    return float(roc_auc_score(y, score))
