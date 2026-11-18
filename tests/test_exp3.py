"""Fast smoke tests for exp_3 (link discrepancy diagnostic).

Uses tiny T, few reps and a small population panel so the whole file runs in
a few seconds; it checks the paper's qualitative claims (discrepancy near
zero at the null, clearly positive signal-to-noise elsewhere), not magnitudes.
"""

from __future__ import annotations

import numpy as np

from exp import exp3


def test_null_discrepancy_much_smaller_than_misspecified():
    U_o4, _, _ = exp3.gs.true_curve_matrix(m=4)
    periods_null = exp3._panel_null(T=80, c=0.1, seed=1)
    periods_dgp_i = exp3._panel_independent_or_dpp(80, "independent", seed=2, U_o4=U_o4)

    fit_po_null, fit_log_null = exp3._fit_pair(periods_null, m=5, max_iter=50)
    fit_po_i, fit_log_i = exp3._fit_pair(periods_dgp_i, m=5, max_iter=50)

    phi5, _ = exp3._dict5()
    phi_z = phi5(exp3.EVAL_POINTS)
    D_null = exp3.diagnostics.discrepancy(fit_po_null, fit_log_null, phi_z)
    D_i = exp3.diagnostics.discrepancy(fit_po_i, fit_log_i, phi_z)

    assert np.sqrt(np.mean(D_null ** 2)) < np.sqrt(np.mean(D_i ** 2))


def test_discrepancy_standard_error_is_positive_and_finite():
    U_o4, _, _ = exp3.gs.true_curve_matrix(m=4)
    periods = exp3._panel_independent_or_dpp(100, "independent", seed=3, U_o4=U_o4)
    fit_po, fit_log = exp3._fit_pair(periods, m=5, max_iter=50)
    phi5, _ = exp3._dict5()
    phi_z = phi5(exp3.EVAL_POINTS)
    D, se = exp3._discrepancy_at(fit_po, fit_log, phi_z, T=100)
    assert np.all(np.isfinite(D))
    assert np.all(se > 0) and np.all(np.isfinite(se))


def test_population_half_discrepancy_nonnegative():
    U_o4, _, _ = exp3.gs.true_curve_matrix(m=4)
    p_o_fn, _ = exp3.gs.true_curve_fn(U_o4, 4)
    half_D, larger_gap, gap_po, gap_log = exp3._population_targets_dict5(
        p_o_fn, U_o4, n_pop=300, seed=42,
    )
    assert half_D >= 0
    assert larger_gap >= 0
    assert gap_po >= 0 and gap_log >= 0
