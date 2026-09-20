"""Generative models under study ("digital twins" of a batch-record table).

All generators expose the same interface:

    gen = Generator(seed=0)
    gen.fit(X)          # X: (n, d) float array, already standardised by caller
    Xs = gen.sample(m)  # (m, d)

The set spans the standard families a synthetic-tabular-data paper is expected
to compare: an independence baseline that destroys all joint structure, two
classical statistical models, a mixture model, and two neural models. Bootstrap
resampling is included deliberately as a reference point that is perfect on
fidelity and utility and maximally bad on privacy -- it makes the three-axis
trade-off visible rather than assumed.

Independent work. Carried out on personal time and equipment, not connected to
the author's employment. No proprietary, confidential or internal data of any
organization was used. All data is public.
"""
from __future__ import annotations

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.mixture import GaussianMixture

import torch
import torch.nn as nn


# --------------------------------------------------------------------------
class BaseGen:
    name = "base"

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.g = np.random.default_rng(seed)

    def fit(self, X: np.ndarray):  # pragma: no cover - interface
        raise NotImplementedError

    def sample(self, m: int) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


class Bootstrap(BaseGen):
    """Resample training rows with replacement. Reference point, not a method:
    perfect marginals and joints, zero privacy."""
    name = "bootstrap"

    def fit(self, X):
        self.X = np.asarray(X, float)
        return self

    def sample(self, m):
        idx = self.g.integers(0, len(self.X), size=m)
        return self.X[idx].copy()


class IndependentMarginal(BaseGen):
    """Sample each column independently from its empirical marginal. Preserves
    every univariate marginal exactly and destroys every dependency."""
    name = "marginal"

    def fit(self, X):
        self.X = np.asarray(X, float)
        return self

    def sample(self, m):
        n, d = self.X.shape
        out = np.empty((m, d))
        for j in range(d):
            out[:, j] = self.X[self.g.integers(0, n, size=m), j]
        return out


class MultivariateGaussian(BaseGen):
    """Single Gaussian with Ledoit-Wolf shrinkage covariance."""
    name = "gaussian"

    def fit(self, X):
        X = np.asarray(X, float)
        self.mu = X.mean(0)
        self.cov = LedoitWolf().fit(X).covariance_
        self.L = np.linalg.cholesky(self.cov + 1e-8 * np.eye(X.shape[1]))
        return self

    def sample(self, m):
        z = self.g.standard_normal((m, len(self.mu)))
        return self.mu + z @ self.L.T


class GaussianCopula(BaseGen):
    """Rank-transform each margin to normal scores, fit a shrinkage Gaussian on
    the scores, then map back through the empirical inverse CDF. Marginals are
    reproduced non-parametrically; dependence is captured to second order."""
    name = "copula"

    def fit(self, X):
        from scipy.stats import norm
        X = np.asarray(X, float)
        self.X_sorted = np.sort(X, axis=0)
        n, d = X.shape
        ranks = np.argsort(np.argsort(X, axis=0), axis=0) + 1
        u = ranks / (n + 1.0)
        Z = norm.ppf(u)
        self.cov = LedoitWolf().fit(Z).covariance_
        self.L = np.linalg.cholesky(self.cov + 1e-8 * np.eye(d))
        self.n = n
        return self

    def sample(self, m):
        from scipy.stats import norm
        d = self.X_sorted.shape[1]
        z = self.g.standard_normal((m, d)) @ self.L.T
        u = np.clip(norm.cdf(z), 1e-6, 1 - 1e-6)
        out = np.empty((m, d))
        pos = u * (self.n - 1)
        lo = np.floor(pos).astype(int)
        hi = np.minimum(lo + 1, self.n - 1)
        w = pos - lo
        for j in range(d):
            col = self.X_sorted[:, j]
            out[:, j] = col[lo[:, j]] * (1 - w[:, j]) + col[hi[:, j]] * w[:, j]
        return out


class GMMGen(BaseGen):
    """Gaussian mixture; component count chosen by BIC over a small grid."""
    name = "gmm"

    def __init__(self, seed=0, k_grid=(2, 4, 8)):
        super().__init__(seed)
        self.k_grid = k_grid

    def fit(self, X):
        X = np.asarray(X, float)
        best, best_bic = None, np.inf
        for k in self.k_grid:
            gm = GaussianMixture(n_components=k, covariance_type="diag",
                                 reg_covar=1e-4, random_state=self.seed,
                                 max_iter=200).fit(X)
            b = gm.bic(X)
            if b < best_bic:
                best, best_bic = gm, b
        self.gm = best
        self.k = best.n_components
        return self

    def sample(self, m):
        Xs, _ = self.gm.sample(m)
        self.g.shuffle(Xs)
        return Xs


# --------------------------------------------------------------------------
# Neural generators
# --------------------------------------------------------------------------

def _mlp(sizes, out_act=None):
    layers = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), nn.ReLU()]
    layers = layers[:-1]
    if out_act is not None:
        layers.append(out_act)
    return nn.Sequential(*layers)


class VAEGen(BaseGen):
    """MLP variational autoencoder with a Gaussian likelihood."""
    name = "vae"

    def __init__(self, seed=0, latent=32, hidden=256, epochs=250, lr=1e-3, batch=128):
        super().__init__(seed)
        self.latent, self.hidden, self.epochs, self.lr, self.batch = (
            latent, hidden, epochs, lr, batch)

    def fit(self, X):
        torch.manual_seed(self.seed)
        X = torch.tensor(np.asarray(X, np.float32))
        n, d = X.shape
        self.d = d
        self.enc = _mlp([d, self.hidden, self.hidden])
        self.mu = nn.Linear(self.hidden, self.latent)
        self.lv = nn.Linear(self.hidden, self.latent)
        self.dec = _mlp([self.latent, self.hidden, self.hidden, d])
        params = (list(self.enc.parameters()) + list(self.mu.parameters())
                  + list(self.lv.parameters()) + list(self.dec.parameters()))
        opt = torch.optim.Adam(params, lr=self.lr)
        # KL warm-up avoids posterior collapse on wide, correlated tables.
        for ep in range(self.epochs):
            perm = torch.randperm(n)
            beta = min(1.0, (ep + 1) / (0.3 * self.epochs))
            for i in range(0, n, self.batch):
                xb = X[perm[i:i + self.batch]]
                h = self.enc(xb)
                mu, lv = self.mu(h), self.lv(h).clamp(-8, 8)
                z = mu + torch.randn_like(mu) * (0.5 * lv).exp()
                xr = self.dec(z)
                rec = ((xr - xb) ** 2).sum(1).mean()
                kl = (-0.5 * (1 + lv - mu ** 2 - lv.exp()).sum(1)).mean()
                loss = rec + beta * kl
                opt.zero_grad(); loss.backward(); opt.step()
        self.eval_mode()
        return self

    def eval_mode(self):
        for m in (self.enc, self.mu, self.lv, self.dec):
            m.eval()

    @torch.no_grad()
    def sample(self, m):
        torch.manual_seed(self.seed + 10_000)
        z = torch.randn(m, self.latent)
        return self.dec(z).numpy().astype(np.float64)


class WGANGen(BaseGen):
    """WGAN with gradient penalty over MLP critic/generator. Stands in for the
    GAN family (CTGAN and relatives) without their categorical machinery, which
    this all-continuous setting does not need."""
    name = "wgan"

    def __init__(self, seed=0, latent=64, hidden=256, steps=1200, lr=2e-4,
                 batch=128, n_critic=3, gp=10.0):
        super().__init__(seed)
        self.latent, self.hidden, self.steps = latent, hidden, steps
        self.lr, self.batch, self.n_critic, self.gp = lr, batch, n_critic, gp

    def _grad_penalty(self, D, real, fake):
        eps = torch.rand(real.size(0), 1)
        x = (eps * real + (1 - eps) * fake).requires_grad_(True)
        d = D(x)
        g = torch.autograd.grad(d, x, torch.ones_like(d), create_graph=True)[0]
        return ((g.norm(2, dim=1) - 1) ** 2).mean()

    def fit(self, X):
        torch.manual_seed(self.seed)
        X = torch.tensor(np.asarray(X, np.float32))
        n, d = X.shape
        self.G = _mlp([self.latent, self.hidden, self.hidden, d])
        D = _mlp([d, self.hidden, self.hidden, 1])
        oG = torch.optim.Adam(self.G.parameters(), lr=self.lr, betas=(0.5, 0.9))
        oD = torch.optim.Adam(D.parameters(), lr=self.lr, betas=(0.5, 0.9))
        for step in range(self.steps):
            for _ in range(self.n_critic):
                idx = torch.randint(0, n, (self.batch,))
                real = X[idx]
                fake = self.G(torch.randn(self.batch, self.latent)).detach()
                lossD = D(fake).mean() - D(real).mean() \
                    + self.gp * self._grad_penalty(D, real, fake)
                oD.zero_grad(); lossD.backward(); oD.step()
            fake = self.G(torch.randn(self.batch, self.latent))
            lossG = -D(fake).mean()
            oG.zero_grad(); lossG.backward(); oG.step()
        self.G.eval()
        return self

    @torch.no_grad()
    def sample(self, m):
        torch.manual_seed(self.seed + 20_000)
        return self.G(torch.randn(m, self.latent)).numpy().astype(np.float64)


GENERATORS = {
    "bootstrap": Bootstrap,
    "marginal": IndependentMarginal,
    "gaussian": MultivariateGaussian,
    "copula": GaussianCopula,
    "gmm": GMMGen,
    "vae": VAEGen,
    "wgan": WGANGen,
}

# Generators that are methods under evaluation. `bootstrap` is a reference point
# and is excluded from any "best method" ranking.
METHODS = ["marginal", "gaussian", "copula", "gmm", "vae", "wgan"]
