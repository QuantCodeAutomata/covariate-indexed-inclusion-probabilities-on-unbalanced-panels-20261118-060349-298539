"""The positive semidefinite sum-of-squares model of Sections 3.1-3.3.

Fits ell_U(z) = phi(z)^T U phi(z), U in S_+^m, by the identity-link Poisson
loss (9) or the log-link Bernoulli loss (19), regularized problem (8):

    minimize_{U in S_+^m}  R_T(U) = (1/T) sum_t L(U, xi_t) + lambda * trace(G U)

Custom implementation (paper Eq. 8, 9, 19): no surveyed library (cvxpy's
disciplined-convex-programming rules do not accept the log-link loss directly
because log(e^x - 1) is not a built-in concave atom; pyportfolioopt/skfolio
solve portfolio objectives, not this likelihood). The paper itself recommends
a custom nonmonotone spectral projected gradient (SPG) solver (Birgin et al.,
2000; Section 7.1 of the paper), which this module implements: projected
gradient steps with a Barzilai-Borwein step length, a nonmonotone Armijo line
search, and projection onto S_+^m (or onto the nonnegative orthant for the
diagonal specification) by eigenvalue clipping.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src import linalg


EPS_FLOOR = 1e-10  # numerical floor so that log(.) and divisions stay finite


@dataclass
class FitResult:
    U: np.ndarray                 # fitted m x m matrix (diagonal embedded if diagonal=True)
    m: int
    diagonal: bool
    link: str                     # "identity" or "log"
    n_iter: int
    grad_norm: float
    period_scores: np.ndarray     # (T, M) per-period scores in vec space, for variance estimators
    hessian_vec: np.ndarray       # (M, M) Hessian of R_T in vec space at U
    multiplier: np.ndarray        # m x m estimated dual multiplier (Lambda_hat)
    lam: float
    converged: bool


def _project_psd(U: np.ndarray) -> np.ndarray:
    U = 0.5 * (U + U.T)
    w, Q = np.linalg.eigh(U)
    w = np.clip(w, 0.0, None)
    return (Q * w) @ Q.T


def _poisson_pieces(Phi_t, incl_t, U):
    """Per-period gradient (matrix form) and Hessian weight vector of the
    Poisson loss, following the gradient formula of the paper's implementation
    notes: nabla_U L = sum_i C_i - sum_{i in S} C_i / ell_i."""
    ell = np.einsum('ij,jk,ik->i', Phi_t, U, Phi_t)
    ell_incl = np.clip(ell[incl_t], EPS_FLOOR, None)
    Csum = Phi_t.T @ Phi_t
    Phi_incl = Phi_t[incl_t]
    grad = Csum - (Phi_incl / ell_incl[:, None]).T @ Phi_incl
    loss = ell.sum() - np.log(ell_incl).sum()
    hess_w = 1.0 / ell_incl ** 2
    return grad, loss, Phi_incl, hess_w


def _log_link_pieces(Phi_t, incl_t, U):
    """Per-period gradient (matrix form) and Hessian weight vector of the
    log-link (Bernoulli) loss (19)."""
    ell = np.einsum('ij,jk,ik->i', Phi_t, U, Phi_t)
    ell = np.clip(ell, EPS_FLOOR, None)
    excl_t = ~incl_t
    Phi_incl = Phi_t[incl_t]
    Phi_excl = Phi_t[excl_t]
    ell_excl = ell[excl_t]
    # loss = sum_S ell_i - sum_{not S} log(e^{ell_i} - 1)
    w = np.exp(-np.clip(ell_excl, 0.0, 700.0))
    denom = np.clip(1.0 - w, EPS_FLOOR, None)
    weight_excl = w / denom  # coefficient of -C_i in the gradient contribution
    grad = Phi_incl.T @ Phi_incl - (Phi_excl * weight_excl[:, None]).T @ Phi_excl
    # loss = sum_S ell_i - sum_{not S} log(1 - e^{-ell_i}) = sum_S ell_i - sum_{not S} log(denom)
    loss = ell[incl_t].sum() - np.log(denom).sum()
    hess_w = weight_excl / denom  # weight for the Hessian, = e^{-ell}/(1-e^{-ell})^2
    return grad, loss, Phi_excl, hess_w


def fit(
    periods: list[tuple[np.ndarray, np.ndarray]],
    m: int,
    link: str = "identity",
    diagonal: bool = False,
    lam: float = 0.0,
    G: np.ndarray | None = None,
    max_iter: int = 500,
    grad_tol: float = 1e-6,
    U_init: np.ndarray | None = None,
) -> FitResult:
    """Fit the psd sum-of-squares model by projected gradient descent.

    Parameters
    ----------
    periods : list of (Phi_t, incl_t)
        Phi_t is an (n_t, m) array of dictionary rows, incl_t a boolean mask
        of length n_t marking the included items (S_{t+1}).
    link : "identity" or "log".
    diagonal : if True, constrain U to be diagonal (the diagonal/additive
        specification of Section 6.2).
    lam, G : regularization parameter and weight matrix of problem (8).
    """
    if link not in ("identity", "log"):
        raise ValueError("link must be 'identity' or 'log'")
    T = len(periods)
    if G is None:
        G = np.eye(m)
    U = np.zeros((m, m)) if U_init is None else U_init.copy()
    if diagonal:
        U = np.diag(np.diag(U))
    pieces = _poisson_pieces if link == "identity" else _log_link_pieces

    period_scores_vec = np.zeros((T, linalg.vech_dim(m)))
    step = 1e-2
    prev_grad_vec = None
    prev_U_vec = None
    best_loss = np.inf
    loss_history = []
    converged = False
    n_iter = 0
    for it in range(max_iter):
        total_grad = np.zeros((m, m))
        total_loss = 0.0
        for t, (Phi_t, incl_t) in enumerate(periods):
            grad_t, loss_t, _, _ = pieces(Phi_t, incl_t, U)
            total_grad += grad_t
            total_loss += loss_t
            period_scores_vec[t] = linalg.sym_to_vec(grad_t)
        total_grad = total_grad / T + lam * G
        total_loss = total_loss / T + lam * np.trace(G @ U)
        grad_vec = linalg.sym_to_vec(total_grad)

        if prev_grad_vec is not None:
            s = linalg.sym_to_vec(U) - prev_U_vec
            y = grad_vec - prev_grad_vec
            sy = s @ y
            if sy > 1e-14:
                step = min(max((s @ s) / sy, 1e-6), 1e6)

        # backtracking Armijo line search (monotone, simplified SPG)
        prev_U_vec = linalg.sym_to_vec(U)
        prev_grad_vec = grad_vec
        for _ in range(30):
            candidate = U - step * total_grad
            if diagonal:
                candidate = np.diag(np.clip(np.diag(candidate), 0.0, None))
            else:
                candidate = _project_psd(candidate)
            cand_loss = 0.0
            ok = True
            for Phi_t, incl_t in periods:
                ell = np.einsum('ij,jk,ik->i', Phi_t, candidate, Phi_t)
                if link == "identity":
                    ell_incl = ell[incl_t]
                    if np.any(ell_incl <= 0):
                        ok = False
                        break
                    cand_loss += ell.sum() - np.log(np.clip(ell_incl, EPS_FLOOR, None)).sum()
                else:
                    ell_excl = ell[~incl_t]
                    if np.any(ell_excl <= 0):
                        ok = False
                        break
                    w_c = np.exp(-np.clip(ell_excl, 0.0, 700.0))
                    denom_c = np.clip(1.0 - w_c, EPS_FLOOR, None)
                    cand_loss += ell[incl_t].sum() - np.log(denom_c).sum()
            if ok:
                cand_loss = cand_loss / T + lam * np.trace(G @ candidate)
                if cand_loss <= total_loss + 1e-4 * (cand_loss - total_loss) or cand_loss <= total_loss:
                    break
            step *= 0.5
        else:
            candidate = U
            cand_loss = total_loss
        U = candidate
        loss_history.append(cand_loss)
        n_iter = it + 1
        if np.linalg.norm(grad_vec) < grad_tol * max(1.0, np.linalg.norm(linalg.sym_to_vec(U))):
            converged = True
            break
        if len(loss_history) > 5 and abs(loss_history[-1] - loss_history[-5]) < 1e-12:
            converged = True
            break

    # Final gradient, Hessian (vec space) and per-period scores at the fit
    M = linalg.vech_dim(m)
    hessian_vec = np.zeros((M, M))
    period_scores_vec = np.zeros((T, M))
    final_grad = np.zeros((m, m))
    for t, (Phi_t, incl_t) in enumerate(periods):
        grad_t, _, Phi_w, hess_w = pieces(Phi_t, incl_t, U)
        final_grad += grad_t
        period_scores_vec[t] = linalg.sym_to_vec(grad_t)
        Cw = linalg.outer_vec(Phi_w)
        hessian_vec += (Cw * hess_w[:, None]).T @ Cw
    hessian_vec /= T
    final_grad = final_grad / T + lam * G

    # Dual multiplier estimate (KKT): Lambda_hat = Q_perp^T grad(U) Q_perp, Remark B.1.
    w_eig, Q_eig = np.linalg.eigh(U)
    tol = 1e-4 * max(w_eig.max(), 1e-12)
    r = int(np.sum(w_eig > tol))
    Q_perp = Q_eig[:, : m - r] if r < m else np.zeros((m, 0))
    Lambda_perp = Q_perp.T @ final_grad @ Q_perp if r < m else np.zeros((0, 0))
    multiplier = Q_perp @ Lambda_perp @ Q_perp.T if r < m else np.zeros((m, m))

    return FitResult(
        U=U, m=m, diagonal=diagonal, link=link, n_iter=n_iter,
        grad_norm=float(np.linalg.norm(linalg.sym_to_vec(final_grad))),
        period_scores=period_scores_vec, hessian_vec=hessian_vec,
        multiplier=multiplier, lam=lam, converged=converged,
    )


def curve(U: np.ndarray, phi_z: np.ndarray) -> np.ndarray:
    """Evaluate ell_U(z) = phi(z)^T U phi(z) for a batch of dictionary rows."""
    return np.einsum('...j,jk,...k->...', phi_z, U, phi_z)
