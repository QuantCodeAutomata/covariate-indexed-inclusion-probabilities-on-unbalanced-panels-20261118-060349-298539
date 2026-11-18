import numpy as np

from data import generate_synthetic as gs


def test_true_curve_matrix_within_target_range():
    U_o, p_min, p_max = gs.true_curve_matrix(m=4, seed=0)
    assert U_o.shape == (4, 4)
    assert np.allclose(U_o, U_o.T)
    assert np.linalg.eigvalsh(U_o).min() > -1e-9
    assert 0.0 <= p_min < p_max <= 0.4


def test_covariate_path_dependent_vs_baseline_differ():
    zs_base, _ = gs.generate_covariate_path(50, dependent=False, seed=1)
    zs_dep, _ = gs.generate_covariate_path(50, dependent=True, seed=1)
    assert len(zs_base) == 50
    assert len(zs_dep) == 50
    concat_base = np.concatenate(zs_base)
    concat_dep = np.concatenate(zs_dep)
    assert not np.allclose(np.sort(concat_base)[:5], np.sort(concat_dep)[:5])
    assert np.all(np.abs(concat_base) < 1.0)
    assert np.all(np.abs(concat_dep) < 1.0)


def test_sample_dgp_i_matches_bernoulli_rate_on_average():
    rng = np.random.default_rng(0)
    p = np.full(20000, 0.3)
    incl = gs.sample_dgp_i(p, rng)
    assert abs(incl.mean() - 0.3) < 0.01


def test_sample_dgp_d_is_repulsive_relative_to_independent():
    """A DPP should spread inclusions out: the included count's variance under
    DGP-D should not exceed that of an independent Bernoulli draw at the same
    mean margin, when covariates are tightly clustered (repulsion matters)."""
    U_o, _, _ = gs.true_curve_matrix(m=4, seed=0)
    p_o_fn, _ = gs.true_curve_fn(U_o, 4)
    rng = np.random.default_rng(1)
    z = np.tanh(rng.normal(0, 0.1, size=30))  # clustered covariates
    counts_dpp = []
    counts_ind = []
    for s in range(300):
        rng_s = np.random.default_rng(1000 + s)
        incl_dpp = gs.sample_dgp_d(z, p_o_fn, rng_s)
        incl_ind = gs.sample_dgp_i(p_o_fn(z), rng_s)
        counts_dpp.append(incl_dpp.sum())
        counts_ind.append(incl_ind.sum())
    assert np.var(counts_dpp) < np.var(counts_ind) + 1e-6


def test_build_panel_shapes():
    U_o, _, _ = gs.true_curve_matrix(m=4, seed=0)
    periods, _ = gs.build_panel(10, dependent=False, dgp="independent", m=4, U_o=U_o, seed=0)
    assert len(periods) == 10
    for p in periods:
        assert p.phi.shape[0] == p.z.shape[0] == p.incl.shape[0]
        assert p.phi.shape[1] == 4


def test_population_targets_bins_cover_support():
    U_o, _, _ = gs.true_curve_matrix(m=4, seed=0)
    all_z, all_p, bar_p_of_z, bin_idx, bar_p_per_bin, counts = gs.population_targets(
        U_o, m=4, n_periods=50, seed=0, n_bins=20
    )
    assert all_z.shape == all_p.shape == bar_p_of_z.shape
    assert counts.sum() == len(all_z)
    assert np.nansum(counts > 0) <= 20
