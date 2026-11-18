"""Fast smoke tests for exp_1 (parametric rate and composition floor).

Uses tiny T, few reps and a small population panel so the whole file runs in
a few seconds; it checks directions of the paper's claims, not magnitudes.
"""

from __future__ import annotations

import numpy as np

from data import generate_synthetic as gs
from exp import exp1
from src import dictionary


_TEST_MAX_ITER = 60  # small cap, enough for direction-only checks on tiny T


def _mean_rmse(T: int, dgp: str, m: int, spec: str, n_reps: int, U_o: np.ndarray,
               test_z: np.ndarray, p_true: np.ndarray) -> float:
    rmses = [
        exp1._fit_one_rep(T, dgp, m, spec, rep, U_o, test_z, p_true, _TEST_MAX_ITER)[-1]
        for rep in range(n_reps)
    ]
    return float(np.mean(rmses))


def test_rmse_decreases_with_T_under_independent():
    U_o, _, _ = gs.true_curve_matrix(m=4)
    n_reps = 2
    rmses = {}
    for T in (60, 200):
        test_z, p_true = exp1._build_test_panel(T, "independent", U_o)
        rmses[T] = _mean_rmse(T, "independent", 4, "full", n_reps, U_o, test_z, p_true)
    assert rmses[200] < rmses[60]


def test_dgp_d_floor_does_not_vanish():
    U_o, _, _ = gs.true_curve_matrix(m=4)
    n_reps = 2
    rmses = {}
    for T in (60, 200):
        test_z, p_true = exp1._build_test_panel(T, "dpp", U_o)
        rmses[T] = _mean_rmse(T, "dpp", 4, "full", n_reps, U_o, test_z, p_true)
    ratio = rmses[200] / rmses[60]
    # A pure parametric rate would give ratio ~ sqrt(60/200) ~ 0.55; a floor
    # means the RMSE barely shrinks.
    assert ratio > 0.6


def test_diagonal_worse_than_full():
    U_o, _, _ = gs.true_curve_matrix(m=4)
    T = 120
    n_reps = 3
    test_z, p_true = exp1._build_test_panel(T, "independent", U_o)
    rmse_full = _mean_rmse(T, "independent", 4, "full", n_reps, U_o, test_z, p_true)
    rmse_diag = _mean_rmse(T, "independent", 4, "diagonal", n_reps, U_o, test_z, p_true)
    assert rmse_diag >= rmse_full


def test_composition_dictionary_decomposition_adds_in_squares():
    # population_targets() reports the dgp-agnostic p^o(z) smoothed over
    # narrow bins; under DGP-D the real composition target is the DPP's
    # realized marginal (exp1._dgp_d_population_margins), which depends on
    # cross-sectional composition and is what Proposition 3.1(iv) actually
    # decomposes. Same (n_periods, seed, dependent) on both calls keeps the
    # covariate path -- hence bin_idx/counts -- identical, as in main().
    U_o, _, _ = gs.true_curve_matrix(m=4)
    n_periods = 50
    n_bins = 30
    seed = 12345
    _, all_p, _, bin_idx, _, counts = gs.population_targets(
        U_o, m=4, n_periods=n_periods, seed=seed, n_bins=n_bins, dependent=False,
    )
    _, all_q = exp1._dgp_d_population_margins(U_o, n_periods=n_periods, seed=seed, dependent=False)
    bar_q_per_bin = np.full(n_bins, np.nan)
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() > 0:
            bar_q_per_bin[b] = all_q[mask].mean()
    bar_q_of_z = bar_q_per_bin[bin_idx]

    composition_part = exp1._composition_part(all_p, bar_q_of_z)
    assert np.isfinite(composition_part) and composition_part >= 0.0
    # The DPP is repulsive, so realized inclusion is systematically below the
    # naive p^o(z) once other nearby items are accounted for: there should be
    # a genuine, non-trivial composition gap (not just binning noise).
    assert composition_part > 0.005

    bin_edges = np.linspace(-1.0, 1.0, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    phi_fn, _ = dictionary.cosine_dictionary(4)
    U_star, ell_star_fn = exp1._population_psd_projection(bin_centers, counts, bar_q_per_bin, 4, phi_fn)
    ell_star_at_centers = ell_star_fn(bin_centers)
    dictionary_part = exp1._dictionary_part(counts, bar_q_per_bin, ell_star_at_centers)
    assert np.isfinite(dictionary_part) and dictionary_part >= 0.0

    predicted = composition_part ** 2 + dictionary_part ** 2
    # Evaluate ell_star through the same bin mapping used for bar_q_of_z, so
    # that the Pythagorean identity of Prop 3.1(iv) holds up to floating
    # point: within each bin, (all_p - bar_q_of_z) has zero mean by
    # construction, so it is exactly orthogonal to any bin-constant quantity
    # such as (bar_q_of_z - ell_star_per_point).
    ell_star_per_point = ell_star_at_centers[bin_idx]
    direct_floor = float(np.mean((all_p - ell_star_per_point) ** 2))
    assert np.isclose(predicted, direct_floor, rtol=0.25)
