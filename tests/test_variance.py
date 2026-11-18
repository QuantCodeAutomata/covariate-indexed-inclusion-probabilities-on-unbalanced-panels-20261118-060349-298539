import numpy as np

from src import linalg, variance


def test_martingale_variance_is_psd_and_symmetric():
    rng = np.random.default_rng(0)
    scores = rng.standard_normal((200, 5))
    G0 = variance.martingale_variance(scores)
    assert np.allclose(G0, G0.T)
    assert np.linalg.eigvalsh(G0).min() > -1e-8


def test_bartlett_reduces_to_martingale_at_lag_zero():
    rng = np.random.default_rng(1)
    scores = rng.standard_normal((100, 3))
    Q0 = variance.bartlett_long_run_variance(scores, lag=0)
    G0 = variance.martingale_variance(scores)
    assert np.allclose(Q0, G0)


def test_newey_west_lag_grows_with_T():
    assert variance.newey_west_lag(100) <= variance.newey_west_lag(1000)


def test_ewc_variance_is_psd():
    rng = np.random.default_rng(2)
    scores = rng.standard_normal((500, 4))
    Q, nu = variance.ewc_variance(scores)
    assert nu > 0
    assert np.linalg.eigvalsh(Q).min() > -1e-6


def test_curve_band_shrinks_with_T():
    rng = np.random.default_rng(3)
    m = 4
    M = linalg.vech_dim(m)
    H_inv = np.eye(M)
    Q = np.eye(M)
    C_z = rng.standard_normal(M)
    band_small = variance.curve_band(C_z, H_inv, Q, T=100)
    band_large = variance.curve_band(C_z, H_inv, Q, T=10000)
    assert band_large < band_small
    assert np.isclose(band_small / band_large, np.sqrt(10000 / 100), atol=1e-6)


def test_fit_rank_detects_full_and_low_rank():
    m = 4
    full = np.eye(m)
    r_full, w, Q = variance.fit_rank(full)
    assert r_full == 4

    low = np.diag([1.0, 1.0, 1.0, 0.0])
    r_low, _, _ = variance.fit_rank(low)
    assert r_low == 3
