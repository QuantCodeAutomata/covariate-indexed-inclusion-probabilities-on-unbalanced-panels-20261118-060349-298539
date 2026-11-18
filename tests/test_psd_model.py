import numpy as np

from data import generate_synthetic as gs
from src import dictionary, psd_model


def _toy_periods(T=60, m=4, seed=0):
    U_o, _, _ = gs.true_curve_matrix(m=m, seed=123)
    periods, _ = gs.build_panel(T, dependent=False, dgp="independent", m=m, U_o=U_o, seed=seed)
    return periods, U_o


def test_identity_link_fit_is_psd_and_reduces_grad_norm():
    periods, _ = _toy_periods()
    fit = psd_model.fit([(p.phi, p.incl) for p in periods], m=4, link="identity", max_iter=200)
    w = np.linalg.eigvalsh(fit.U)
    assert w.min() > -1e-8
    assert fit.grad_norm < 1.0


def test_log_link_fit_curve_in_unit_interval():
    periods, _ = _toy_periods(seed=1)
    fit = psd_model.fit([(p.phi, p.incl) for p in periods], m=4, link="log", max_iter=200)
    phi, _ = dictionary.cosine_dictionary(4)
    z = np.linspace(-0.9, 0.9, 11)
    ell = psd_model.curve(fit.U, phi(z))
    p = np.exp(-ell)
    assert np.all(p >= 0.0) and np.all(p <= 1.0 + 1e-9)


def test_diagonal_specification_is_diagonal():
    periods, _ = _toy_periods(seed=2)
    fit = psd_model.fit([(p.phi, p.incl) for p in periods], m=4, link="identity", diagonal=True, max_iter=200)
    off_diag = fit.U - np.diag(np.diag(fit.U))
    assert np.allclose(off_diag, 0.0)


def test_fit_recovers_curve_better_than_initial_guess_with_more_data():
    """More data should not increase RMSE against the true curve (law of large numbers)."""
    U_o, _, _ = gs.true_curve_matrix(m=4, seed=123)
    phi, _ = dictionary.cosine_dictionary(4)
    z_test = np.linspace(-0.9, 0.9, 50)
    true_p = psd_model.curve(U_o, phi(z_test))

    def rmse_at(T, seed):
        periods, _ = gs.build_panel(T, dependent=False, dgp="independent", m=4, U_o=U_o, seed=seed)
        fit = psd_model.fit([(p.phi, p.incl) for p in periods], m=4, link="identity", max_iter=300)
        pred = psd_model.curve(fit.U, phi(z_test))
        return np.sqrt(np.mean((pred - true_p) ** 2))

    rmse_small = rmse_at(80, seed=10)
    rmse_large = rmse_at(2000, seed=11)
    assert rmse_large < rmse_small
