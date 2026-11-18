"""Fast sanity tests for exp_2 (Section 6.3, Table 2): coverage of confidence
intervals for the target curve. Uses tiny T and few replications so the whole
file runs in well under 15 seconds; exact paper coverage numbers are not
expected to reproduce at this scale (see exp/exp2.py docstring)."""

from __future__ import annotations

import numpy as np

from data import generate_synthetic as gs
from src import config, dictionary, linalg, psd_model, variance


def _fit_once(T: int, dgp: str, m: int, seed: int, diagonal: bool = False):
    U_o, _, _ = gs.true_curve_matrix(m=4)
    periods, _ = gs.build_panel(T, dependent=True, dgp=dgp, m=4, U_o=U_o, seed=seed)
    if m == 4:
        fit_periods = [(p.phi, p.incl) for p in periods]
    else:
        phi8, _ = dictionary.cosine_dictionary(8)
        fit_periods = [(phi8(p.z), p.incl) for p in periods]
    fit_result = psd_model.fit(periods=fit_periods, m=m, link="identity", diagonal=diagonal, lam=0.0)
    return U_o, fit_result


def test_martingale_coverage_reasonable_under_dgp_i():
    T = 60
    n_reps = 10
    z = 0.0
    covered = []
    half_widths = []
    for rep in range(n_reps):
        U_o, fit_result = _fit_once(T, "independent", m=4, seed=1000 + rep)
        p_o_fn, _ = gs.true_curve_fn(U_o, 4)
        phi4, _ = dictionary.cosine_dictionary(4)

        phi_z = phi4(np.array([z]))[0]
        C_z = linalg.outer_vec(phi_z)
        H_inv = variance.pinv(fit_result.hessian_vec)
        ell_hat = float(psd_model.curve(fit_result.U, phi_z[None, :])[0])
        p_true = float(p_o_fn(np.array([z]))[0])

        Q = variance.martingale_variance(fit_result.period_scores)
        half = variance.curve_band(C_z, H_inv, Q, T, level=config.CONFIDENCE_LEVEL)

        half_widths.append(half)
        covered.append(abs(ell_hat - p_true) <= half)

    half_widths = np.array(half_widths)
    assert np.all(np.isfinite(half_widths))
    assert np.all(half_widths > 0.0)

    coverage = np.mean(covered)
    assert 0.5 <= coverage <= 1.0


def test_ewc_uses_t_distribution_dof():
    rng = np.random.default_rng(0)
    T, M = 80, 5
    period_scores = rng.standard_normal((T, M))
    Q, nu = variance.ewc_variance(period_scores)
    assert nu >= 1

    H_inv = np.eye(M)
    C_z = rng.standard_normal(M)
    half_t = variance.curve_band(C_z, H_inv, Q, T, level=config.CONFIDENCE_LEVEL, dof=nu)
    half_normal = variance.curve_band(C_z, H_inv, Q, T, level=config.CONFIDENCE_LEVEL, dof=None)
    assert half_t > half_normal


def test_bartlett_band_differs_from_martingale_under_serial_correlation():
    rng = np.random.default_rng(1)
    T, M = 100, 3
    innovations = rng.standard_normal((T, M))
    scores = np.zeros((T, M))
    scores[0] = innovations[0]
    for t in range(1, T):
        scores[t] = 0.8 * scores[t - 1] + innovations[t]

    G0 = variance.martingale_variance(scores)
    lag = variance.newey_west_lag(T)
    Q_bart = variance.bartlett_long_run_variance(scores, lag)

    assert np.all(np.diag(Q_bart) > np.diag(G0))


def test_fit_at_m8_runs_and_covers_true_curve_roughly():
    T = 80
    U_o, fit_result = _fit_once(T, "independent", m=8, seed=2024)
    p_o_fn, _ = gs.true_curve_fn(U_o, 4)
    phi8, _ = dictionary.cosine_dictionary(8)

    z = 0.0
    phi_z = phi8(np.array([z]))[0]
    ell_hat = float(psd_model.curve(fit_result.U, phi_z[None, :])[0])
    p_true = float(p_o_fn(np.array([z]))[0])

    assert abs(ell_hat - p_true) < 0.15
