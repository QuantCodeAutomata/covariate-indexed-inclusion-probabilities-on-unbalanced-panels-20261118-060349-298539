"""Synthetic panel generation for the simulation study (Section 6, Appendix C).

All experiments use synthetic data: none of exp_1, exp_2, exp_3, exp_6 or
exp_7 draws on market data, so this module, not data/data_loader.py, is the
data path shared by every experiment script in exp/.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src import config, dictionary


@dataclass
class Period:
    z: np.ndarray        # (n_t,) covariates
    phi: np.ndarray       # (n_t, dict_size) dictionary rows
    incl: np.ndarray      # (n_t,) boolean inclusion mask


def true_curve_matrix(m: int = 4, seed: int = config.TRUE_CURVE_SEED,
                       ridge: float = config.RIDGE_BASE,
                       p_low: float = config.P_RANGE_LOW,
                       p_high: float = config.P_RANGE_HIGH):
    """U^o = s*(W0 W0^T + ridge*I_m), scaled so p^o = phi U^o phi^T ranges
    close to [p_low, p_high] over z in (-1, 1) (Section 6.1). The exact draw
    of W0 is not specified in the paper; a fixed documented seed is used.
    """
    rng = np.random.default_rng(seed)
    W0 = rng.standard_normal((m, m))
    M = W0 @ W0.T + ridge * np.eye(m)
    phi, _ = dictionary.cosine_dictionary(m)
    z_grid = np.linspace(-0.999, 0.999, 4001)
    q = np.einsum('ij,jk,ik->i', phi(z_grid), M, phi(z_grid))
    s = p_high / q.max()
    U_o = s * M
    q_scaled = s * q
    return U_o, float(q_scaled.min()), float(q_scaled.max())


def true_curve_fn(U_o: np.ndarray, m: int):
    phi, _ = dictionary.cosine_dictionary(m)

    def p_o(z):
        return np.einsum('...j,jk,...k->...', phi(z), U_o, phi(z))

    return p_o, phi


def generate_covariate_path(T: int, dependent: bool, seed: int, m: int = 4):
    """z_{t,i} = tanh(a_t + eps_{t,i}), eps ~ N(0, 0.7^2); a_t = 0 (baseline)
    or AR(1) (dependent design), Section 6.1. n_t ~ Uniform{15,...,45}."""
    rng = np.random.default_rng(seed)
    a = np.zeros(T)
    if dependent:
        for t in range(1, T):
            a[t] = config.AR_COEF * a[t - 1] + rng.normal(0, config.AR_INNOVATION_SD)
    n_t = rng.integers(config.N_ITEMS_LOW, config.N_ITEMS_HIGH + 1, size=T)
    zs = []
    for t in range(T):
        eps = rng.normal(0, config.EPS_SD, size=n_t[t])
        zs.append(np.tanh(a[t] + eps))
    return zs, rng


def sample_dgp_i(p_vals: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Independent inclusions given the margins (DGP-I)."""
    u = rng.random(p_vals.shape[0])
    return u < p_vals


def _dpp_sample_from_kernel(L: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Exact sampling from an L-ensemble determinantal point process (Hough,
    Krishnapur, Peres and Virag, 2006; Algorithm 1 of Kulesza and Taskar,
    2012, "Determinantal Point Processes for Machine Learning").

    Custom implementation: no installed library ships a general finite DPP
    sampler (the suggested 'dppy' package is not part of the available
    environment), so the standard spectral algorithm is implemented directly.
    """
    n = L.shape[0]
    eigvals, eigvecs = np.linalg.eigh(L)
    eigvals = np.clip(eigvals, 0.0, None)
    probs = eigvals / (1.0 + eigvals)
    chosen = rng.random(n) < probs
    V = eigvecs[:, chosen]
    k = V.shape[1]
    included = np.zeros(n, dtype=bool)
    for _ in range(k):
        weights = np.sum(V ** 2, axis=1)
        total = weights.sum()
        if total <= 1e-14:
            break
        probs_i = weights / total
        i = rng.choice(n, p=probs_i)
        included[i] = True
        # remove the component of e_i from V and re-orthonormalize
        if V.shape[1] == 1:
            V = np.zeros((n, 0))
            break
        e_i = np.zeros(n)
        e_i[i] = 1.0
        proj = V.T @ e_i
        j = np.argmax(np.abs(proj))
        v_j = V[:, j]
        V = V - np.outer(v_j, V[j, :] / v_j[i]) if abs(v_j[i]) > 1e-12 else V
        V = np.delete(V, j, axis=1)
        if V.shape[1] > 0:
            Q, _ = np.linalg.qr(V)
            V = Q[:, : V.shape[1]]
    return included


def sample_dgp_d(z: np.ndarray, p_o_fn, rng: np.random.Generator,
                  bandwidth: float = config.DPP_RBF_BANDWIDTH) -> np.ndarray:
    """Repulsive determinantal point process (DGP-D), L-ensemble kernel
    L_t = Diag(sqrt(varpi)) K^rbf Diag(sqrt(varpi)), Section 6.1."""
    p_vals = np.clip(p_o_fn(z), 1e-6, 1 - 1e-6)
    varpi = p_vals / (1 - p_vals)
    diff = z[:, None] - z[None, :]
    K = np.exp(-(diff ** 2) / (2 * bandwidth ** 2))
    sq = np.sqrt(varpi)
    L = (sq[:, None] * K) * sq[None, :]
    return _dpp_sample_from_kernel(L, rng)


def build_panel(T: int, dependent: bool, dgp: str, m: int, U_o: np.ndarray,
                 seed: int) -> tuple[list[Period], np.random.Generator]:
    """Build a panel of T periods under DGP-I ('independent') or DGP-D ('dpp')."""
    zs, rng = generate_covariate_path(T, dependent, seed, m)
    p_o_fn, phi_fn = true_curve_fn(U_o, m)
    periods = []
    for z in zs:
        phi_z = phi_fn(z)
        if dgp == "independent":
            p_vals = p_o_fn(z)
            incl = sample_dgp_i(p_vals, rng)
        elif dgp == "dpp":
            incl = sample_dgp_d(z, p_o_fn, rng)
        else:
            raise ValueError("dgp must be 'independent' or 'dpp'")
        periods.append(Period(z=z, phi=phi_z, incl=incl))
    return periods, rng


def population_targets(U_o: np.ndarray, m: int, n_periods: int, seed: int,
                        n_bins: int = 200, dependent: bool = False):
    """Population quantities on a large panel with exact margins in place of
    the sampled inclusions (Section 6.1): the true margin p_{t,i} = p^o(z_ti),
    its composition average bar_p(z) in covariate bins, and the dictionary
    part / composition part of Proposition 3.1(iv)."""
    zs, _ = generate_covariate_path(n_periods, dependent, seed, m)
    p_o_fn, _ = true_curve_fn(U_o, m)
    all_z = np.concatenate(zs)
    all_p = p_o_fn(all_z)
    bins = np.linspace(-1.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(all_z, bins) - 1, 0, n_bins - 1)
    bar_p_per_bin = np.zeros(n_bins)
    counts = np.zeros(n_bins)
    for b in range(n_bins):
        mask = bin_idx == b
        counts[b] = mask.sum()
        bar_p_per_bin[b] = all_p[mask].mean() if mask.sum() > 0 else np.nan
    bar_p_of_z = bar_p_per_bin[bin_idx]
    return all_z, all_p, bar_p_of_z, bin_idx, bar_p_per_bin, counts
