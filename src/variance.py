"""Variance estimators and confidence bands for the fitted curve
(Proposition 4.4, Theorem 4.3(v), Remark B.1, Proposition C.1).

Custom implementation (paper Eq. 44, Theorem 4.3(v), Remark B.1, Proposition
C.1): these are bespoke long-run-variance and cone-geometry constructions tied
to the paper's estimator; statsmodels' sandwich_covariance implements a
Newey-West/Bartlett kernel for i.i.d.-indexed regression residuals, which is
reused for the Bartlett part, but the martingale variance, the equal-weighted
cosine (EWC) estimator of Lazarus et al. (2018), and the face/projection bands
of Appendix B-C have no library equivalent and are implemented here.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from src import linalg


def martingale_variance(period_scores: np.ndarray) -> np.ndarray:
    """G(0): empirical variance of the centered per-period scores (Eq. 44)."""
    T = period_scores.shape[0]
    centered = period_scores - period_scores.mean(axis=0, keepdims=True)
    return (centered.T @ centered) / T


def bartlett_long_run_variance(period_scores: np.ndarray, lag: int) -> np.ndarray:
    """Newey-West/Bartlett-kernel long-run covariance Q_LR = G(0) + sum_k w_k (G(k)+G(k)^T)."""
    T = period_scores.shape[0]
    centered = period_scores - period_scores.mean(axis=0, keepdims=True)
    Q = (centered.T @ centered) / T
    for k in range(1, min(lag, T - 1) + 1):
        w = 1.0 - k / (lag + 1)
        Gk = (centered[k:].T @ centered[:-k]) / T
        Q += w * (Gk + Gk.T)
    return Q


def newey_west_lag(T: int) -> int:
    return int(np.floor(4 * (T / 100) ** (2 / 9)))


def ewc_variance(period_scores: np.ndarray):
    """Equal-weighted cosine (EWC) long-run variance estimator (Lazarus, Lewis,
    Stock and Watson, 2018): project the centered score series onto nu cosine
    tapers and average the outer products of the projections.

    Returns (Q_EWC, nu) so that a t_nu critical value is used for the band.
    """
    T, M = period_scores.shape
    centered = period_scores - period_scores.mean(axis=0, keepdims=True)
    nu = max(int(np.floor(0.4 * T ** (2 / 3))), 1)
    t_idx = np.arange(1, T + 1)
    Q = np.zeros((M, M))
    for j in range(1, nu + 1):
        basis = np.sqrt(2.0) * np.cos(j * np.pi * (t_idx - 0.5) / T)
        Lambda_j = (basis @ centered) / np.sqrt(T)
        Q += np.outer(Lambda_j, Lambda_j)
    Q /= nu
    return Q, nu


def curve_band(
    C_z: np.ndarray,
    H_inv: np.ndarray,
    Q: np.ndarray,
    T: int,
    level: float = 0.95,
    dof: float | None = None,
) -> float:
    """Half-width of the asymptotic interval sqrt(<C(z), H^-1 Q H^-1 [C(z)]> / T),
    Theorem 4.3(v). If dof is given (EWC), use the t_dof critical value."""
    V = H_inv @ Q @ H_inv
    sigma2 = float(C_z @ V @ C_z)
    sigma2 = max(sigma2, 0.0)
    if dof is None:
        crit = stats.norm.ppf(0.5 + level / 2)
    else:
        crit = stats.t.ppf(0.5 + level / 2, dof)
    return crit * np.sqrt(sigma2 / T)


def pinv(H: np.ndarray, rcond: float = 1e-10) -> np.ndarray:
    return np.linalg.pinv(H, rcond=rcond, hermitian=True)


# ---------------------------------------------------------------------------
# Bands at a boundary target of the psd cone (Remark B.1, Proposition C.1)
# ---------------------------------------------------------------------------

def fit_rank(U: np.ndarray, rel_tol: float = 1e-4) -> tuple[int, np.ndarray, np.ndarray]:
    """Rank of U at a relative eigenvalue tolerance, and its eigenbasis sorted
    descending (Table 7/8 caption: "a relative eigenvalue tolerance of 1e-4")."""
    w, Q = np.linalg.eigh(U)
    order = np.argsort(w)[::-1]
    w, Q = w[order], Q[:, order]
    tol = rel_tol * max(w[0], 1e-12)
    r = int(np.sum(w > tol))
    return r, w, Q


def face_operator(
    hessian_vec: np.ndarray,
    Q_eig: np.ndarray,
    r: int,
    multiplier: np.ndarray,
    U: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project the vec-space Hessian onto the tangent space T_star of the
    rank-r manifold at U, and add the curvature term S_star of Remark B.1:
    S_star[A] = Lambda A U^+ + U^+ A Lambda. Returns (P, S_vec) such that the
    face operator is P @ (H + S_vec) @ P, in the SAME (standard) vec-space
    basis as hessian_vec (so it composes directly with C(z)).
    """
    m = Q_eig.shape[0]
    U_pinv = pinv(U)
    M = linalg.vech_dim(m)
    S_vec = np.zeros((M, M))
    for k in range(M):
        e = np.zeros(M)
        e[k] = 1.0
        A = linalg.vec_to_sym(e, m)
        SA = multiplier @ A @ U_pinv + U_pinv @ A @ multiplier
        S_vec[:, k] = linalg.sym_to_vec(SA)

    # Projection onto T_star = {A : Q_perp^T A Q_perp = 0}: zero the
    # kernel-kernel block in the eigenbasis coordinate system, then rotate back.
    R = linalg.basis_change_matrix(Q_eig)  # standard -> eigenbasis coordinates
    idx_i, idx_j = np.triu_indices(m)
    keep = (idx_i < r) | (idx_j < r)  # eigenbasis ordering: range directions first
    P_eig = np.diag(keep.astype(float))
    P = R.T @ P_eig @ R
    return P, S_vec


def face_band(
    hessian_vec: np.ndarray,
    Q_matrix: np.ndarray,
    Q_eig: np.ndarray,
    r: int,
    multiplier: np.ndarray,
    U: np.ndarray,
    C_z: np.ndarray,
    T: int,
    level: float = 0.95,
    with_curvature: bool = True,
) -> float:
    """Face band of Remark B.1 (or its no-curvature variant)."""
    m = Q_eig.shape[0]
    P, S_vec = face_operator(hessian_vec, Q_eig, r, multiplier, U)
    H_face = P @ hessian_vec @ P
    if with_curvature:
        H_face = P @ (hessian_vec + S_vec) @ P
    Q_face = P @ Q_matrix @ P
    H_face_pinv = pinv(H_face)
    return curve_band(C_z, H_face_pinv, Q_face, T, level=level)


def projection_band_quantile(
    hessian_vec: np.ndarray,
    Q_matrix: np.ndarray,
    U: np.ndarray,
    C_z: np.ndarray,
    T: int,
    n_draws: int = 400,
    c_T: float | None = None,
    level: float = 0.95,
    rng: np.random.Generator | None = None,
) -> float:
    """Plug-in projection band of Proposition C.1 at a zero multiplier.

    Simplified implementation: the kernel-kernel block of each simulated
    direction is projected onto the PSD cone by eigenvalue clipping under the
    identity metric (rather than the exact H-weighted metric), which is a
    documented numerical simplification; the range and range-kernel blocks are
    left unconstrained, matching the paper's cone (39) restricted to the
    estimated kernel subspace.
    """
    rng = rng or np.random.default_rng(0)
    m = U.shape[0]
    M = linalg.vech_dim(m)
    H_pinv = pinv(hessian_vec)
    V = H_pinv @ Q_matrix @ H_pinv  # asymptotic covariance of sqrt(T)(U_hat - U_star)

    w_eig, Q_eig = np.linalg.eigh(U)
    order = np.argsort(w_eig)[::-1]
    w_eig, Q_eig = w_eig[order], Q_eig[:, order]
    if c_T is None:
        c_T = np.sqrt(np.log(max(T, 3)))
    q_m = Q_eig[:, -1]
    e_mm = linalg.sym_to_vec(np.outer(q_m, q_m))
    s_hat = float(np.sqrt(max(e_mm @ V @ e_mm, 1e-14)))
    threshold = c_T * s_hat / np.sqrt(T)
    r_hat = int(np.sum(w_eig > threshold))

    R = linalg.basis_change_matrix(Q_eig)
    V_eig = R @ V @ R.T
    L = np.linalg.cholesky(V_eig + 1e-12 * np.eye(M))
    idx_i, idx_j = np.triu_indices(m)
    kernel_mask = (idx_i >= r_hat) & (idx_j >= r_hat)
    kdim = m - r_hat

    draws = rng.standard_normal((n_draws, M)) @ L.T  # eta in eigenbasis coordinates
    projected = draws.copy()
    if kdim > 0:
        kernel_idx = np.where(kernel_mask)[0]
        for d in range(n_draws):
            sub_vec = draws[d, kernel_idx]
            sub_mat = linalg.vec_to_sym(sub_vec, kdim)
            w_sub, Q_sub = np.linalg.eigh(sub_mat)
            w_sub = np.clip(w_sub, 0.0, None)
            proj_mat = (Q_sub * w_sub) @ Q_sub.T
            projected[d, kernel_idx] = linalg.sym_to_vec(proj_mat)

    C_z_eig = R @ C_z
    values = projected @ C_z_eig
    half = 0.5 * (1 - level)
    lo, hi = np.quantile(values, [half, 1 - half])
    center = 0.0
    return float(max(abs(lo - center), abs(hi - center))), lo, hi
