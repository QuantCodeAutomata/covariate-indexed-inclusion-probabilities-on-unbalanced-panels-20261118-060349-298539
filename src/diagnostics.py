"""The link discrepancy diagnostic D(z) = ell_star^Po(z) - exp(-ell_star^log(z))
of Section 5 (Proposition 5.1, Corollary 5.2).

Custom implementation (paper Eq. 45-46, Corollary 5.2): a bespoke stacked-score
sandwich variance tied to the paper's two fits; no surveyed library computes it.
"""

from __future__ import annotations

import numpy as np

from src import linalg, psd_model, variance


def discrepancy(fit_po: psd_model.FitResult, fit_log: psd_model.FitResult, phi_z: np.ndarray) -> np.ndarray:
    ell_po = psd_model.curve(fit_po.U, phi_z)
    ell_log = psd_model.curve(fit_log.U, phi_z)
    return ell_po - np.exp(-ell_log)


def stacked_full_space_se(
    fit_po: psd_model.FitResult,
    fit_log: psd_model.FitResult,
    Q_po: np.ndarray,
    Q_log: np.ndarray,
    phi_z: np.ndarray,
    T: int,
) -> np.ndarray:
    """Full-space stacked-score standard error of Corollary 5.2(i) at a batch
    of evaluation points z, via the block-diagonal Hessian H_j and d(z) = (C(z),
    p_star^log(z) C(z)), Q_{j,LR} the long-run covariance of the stacked score.
    """
    m_po, m_log = fit_po.m, fit_log.m
    H_po_pinv = variance.pinv(fit_po.hessian_vec)
    H_log_pinv = variance.pinv(fit_log.hessian_vec)

    scores_po = fit_po.period_scores
    scores_log = fit_log.period_scores
    stacked_scores = np.concatenate([scores_po, scores_log], axis=1)
    Q_stacked = variance.bartlett_long_run_variance(stacked_scores, variance.newey_west_lag(T))

    p_log_z = np.exp(-psd_model.curve(fit_log.U, phi_z))
    C_z = linalg.outer_vec(phi_z)

    ses = np.zeros(len(phi_z))
    M_po = linalg.vech_dim(m_po)
    for idx in range(len(phi_z)):
        d_po = H_po_pinv @ C_z[idx]
        d_log = p_log_z[idx] * (H_log_pinv @ C_z[idx])
        d_stacked = np.concatenate([d_po, d_log])
        var = float(d_stacked @ Q_stacked @ d_stacked)
        ses[idx] = np.sqrt(max(var, 0.0) / T)
    return ses
