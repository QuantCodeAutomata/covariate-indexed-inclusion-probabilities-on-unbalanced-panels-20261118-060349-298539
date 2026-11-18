"""Experiment 3: link discrepancy diagnostic D(z) = ell_star^Po(z) - exp(-ell_star^log(z))
(paper Section 6.4, Proposition 5.1(i), Corollary 5.2).

Fits the identity-link (Poisson) and log-link (Bernoulli) sum-of-squares models
on the dictionary {1, phi_1, ..., phi_4} under four designs: log-correct (the
identity class is misspecified), DGP-I (the log class is misspecified), DGP-D
(both classes are misspecified) and the null (both classes are correct,
constant margins). The discrepancy is estimated at a few evaluation points and
compared with its stacked-score standard error (Corollary 5.2) and with the
population half-discrepancy / larger dictionary gap (Proposition 5.1(i)).
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from data import generate_synthetic as gs
from src import config, dictionary, diagnostics, linalg, psd_model, variance

EVAL_POINTS = np.array(config.EXP3_EVAL_POINTS)


def _log_correct_curve(seed: int, ridge: float, c0: float, p_low: float, p_high: float):
    """ell^o = c0 + s * phi U_4 phi^T, p^o = exp(-ell^o), with the 4-cosine
    dictionary (Section 6.4). s is chosen so p^o ranges close to [p_low, p_high]."""
    rng = np.random.default_rng(seed)
    W1 = rng.standard_normal((4, 4))
    M = W1 @ W1.T + ridge * np.eye(4)
    phi, _ = dictionary.cosine_dictionary(4)
    z_grid = np.linspace(-0.999, 0.999, 2001)
    q = np.einsum('ij,jk,ik->i', phi(z_grid), M, phi(z_grid))
    target_ell_range = np.log(1.0 / np.array([p_high, p_low])) - c0
    s = float(np.clip(target_ell_range[1] / q.max(), 1e-6, None))
    U4 = s * M

    def ell_o(z):
        return c0 + np.einsum('...j,jk,...k->...', phi(z), U4, phi(z))

    def p_o(z):
        return np.exp(-ell_o(z))

    return U4, ell_o, p_o


def _dict5():
    """{1, phi_1, ..., phi_4}, dict_size = 5."""
    return dictionary.cosine_dictionary(4, include_constant=True)


def _panel_log_correct(T: int, seed: int, c0: float, ridge: float, p_low: float, p_high: float):
    phi5, _ = _dict5()
    _, _, p_o = _log_correct_curve(config.EXP3_LOG_CORRECT_SEED, ridge, c0, p_low, p_high)
    rng = np.random.default_rng(seed)
    n_t = rng.integers(config.N_ITEMS_LOW, config.N_ITEMS_HIGH + 1, size=T)
    periods = []
    for nt in n_t:
        eps = rng.normal(0, config.EPS_SD, size=nt)
        z = np.tanh(eps)
        p = p_o(z)
        incl = rng.random(nt) < p
        periods.append((phi5(z), incl))
    return periods, p_o


def _panel_independent_or_dpp(T: int, dgp: str, seed: int, U_o4: np.ndarray):
    """DGP-I / DGP-D panels on dictionary {1, phi_1..4}; the true curve only
    uses phi_1..4 (constant coefficient 0), so U_o5 embeds U_o4 in the lower
    4x4 block with a zero row/column for the constant."""
    phi5, _ = _dict5()
    phi4, _ = dictionary.cosine_dictionary(4)
    zs, rng = gs.generate_covariate_path(T, dependent=False, seed=seed, m=4)
    periods = []
    for z in zs:
        p = np.einsum('ij,jk,ik->i', phi4(z), U_o4, phi4(z))
        if dgp == "independent":
            incl = gs.sample_dgp_i(p, rng)
        else:
            def p_o4(zz):
                return np.einsum('...j,jk,...k->...', phi4(zz), U_o4, phi4(zz))
            incl = gs.sample_dgp_d(z, p_o4, rng)
        periods.append((phi5(z), incl))
    return periods


def _panel_null(T: int, c: float, seed: int):
    """Null design: independent inclusions with constant margin p = c,
    represented by U = diag(c, 0, 0, 0, 0) on {1, phi_1..4} (phi_0=1 so
    ell = c is a valid sum of squares, c = 1^2 * c... here fit U directly;
    the DGP itself just draws Bernoulli(c))."""
    phi5, _ = _dict5()
    rng = np.random.default_rng(seed)
    n_t = rng.integers(config.N_ITEMS_LOW, config.N_ITEMS_HIGH + 1, size=T)
    periods = []
    for nt in n_t:
        eps = rng.normal(0, config.EPS_SD, size=nt)
        z = np.tanh(eps)
        incl = rng.random(nt) < c
        periods.append((phi5(z), incl))
    return periods


def _fit_pair(periods, m, max_iter=150):
    fit_po = psd_model.fit(periods, m=m, link="identity", lam=0.0, max_iter=max_iter)
    fit_log = psd_model.fit(periods, m=m, link="log", lam=0.0, max_iter=max_iter)
    return fit_po, fit_log


def _discrepancy_at(fit_po, fit_log, phi_z, T):
    D = diagnostics.discrepancy(fit_po, fit_log, phi_z)
    Q_po = variance.bartlett_long_run_variance(fit_po.period_scores, variance.newey_west_lag(T))
    Q_log = variance.bartlett_long_run_variance(fit_log.period_scores, variance.newey_west_lag(T))
    se = diagnostics.stacked_full_space_se(fit_po, fit_log, Q_po, Q_log, phi_z, T)
    return D, se


def _population_targets_dict5(p_o_fn_4, U_o4, n_pop: int, seed: int, m: int = 5):
    """Population half-discrepancy, larger dictionary gap, on a fixed large
    sample with exact margins (Proposition 5.1(i)); uses the identity-link and
    log-link POPULATION projections via cvxpy SDP on the dictionary {1,phi1..4}.
    """
    import cvxpy as cp

    phi5, _ = _dict5()
    phi4, _ = dictionary.cosine_dictionary(4)
    rng = np.random.default_rng(seed)
    z = np.tanh(rng.normal(0, config.EPS_SD, size=n_pop))
    p_true = p_o_fn_4(z)
    Phi5 = phi5(z)

    U_po = cp.Variable((m, m), symmetric=True)
    ell_po = cp.sum(cp.multiply(Phi5 @ U_po, Phi5), axis=1)
    obj_po = cp.sum(ell_po) / n_pop - cp.sum(cp.multiply(p_true, cp.log(ell_po))) / n_pop
    cp.Problem(cp.Minimize(obj_po), [U_po >> 0]).solve(solver=cp.SCS)
    ell_star_po = Phi5.value if False else np.einsum('ij,jk,ik->i', Phi5, U_po.value, Phi5)

    U_log = cp.Variable((m, m), symmetric=True)
    ell_log = cp.sum(cp.multiply(Phi5 @ U_log, Phi5), axis=1)
    # population Bernoulli risk: E[(1-p) ell - p log(e^ell - 1)]; use the
    # identity (1-p)*ell - p*log(expm1(ell)) with a smooth cvxpy surrogate via
    # softplus is not DCP; instead minimize the equivalent cross-entropy on
    # q = 1 - exp(-ell) directly is non-convex in U, so project numerically
    # with scipy instead of cvxpy for the log-link population target.
    from scipy.optimize import minimize

    def log_loss(u_vec):
        Uu = linalg.vec_to_sym(u_vec, m)
        ell = np.einsum('ij,jk,ik->i', Phi5, Uu, Phi5)
        ell = np.clip(ell, 1e-8, 700.0)
        return float(np.mean((1 - p_true) * ell - p_true * np.log(np.expm1(ell))))

    u0 = linalg.sym_to_vec(np.eye(m) * 0.05)
    res = minimize(log_loss, u0, method="Nelder-Mead", options={"maxiter": 4000, "xatol": 1e-5, "fatol": 1e-8})
    U_log_star = linalg.vec_to_sym(res.x, m)
    w, Q = np.linalg.eigh(U_log_star)
    U_log_star = (Q * np.clip(w, 0, None)) @ Q.T
    ell_star_log = np.einsum('ij,jk,ik->i', Phi5, U_log_star, Phi5)
    p_star_log = np.exp(-ell_star_log)

    D_pop = ell_star_po - p_star_log
    half_D = 0.5 * np.sqrt(np.mean(D_pop ** 2))
    gap_po = np.sqrt(np.mean((ell_star_po - p_true) ** 2))
    gap_log = np.sqrt(np.mean((p_star_log - p_true) ** 2))
    larger_gap = max(gap_po, gap_log)
    return half_D, larger_gap, gap_po, gap_log


def main(full: bool = False) -> None:
    t0 = time.time()
    n_reps = config.EXP3_N_REPS_PAPER if full else config.EXP3_N_REPS_RUN
    n_reps_dpp = config.EXP3_N_REPS_DPP_PAPER if full else config.EXP3_N_REPS_DPP_RUN
    n_reps_null = config.EXP3_N_REPS_NULL_PAPER if full else config.EXP3_N_REPS_NULL_RUN
    T = config.EXP3_T if full else 150
    n_pop = config.EXP3_POPULATION_PANEL_SIZE_PAPER if full else 800
    max_iter = 400 if full else 60
    m = 5

    rows = []
    U_o4, _, _ = gs.true_curve_matrix(m=4)
    _, _, p_o_log_correct = _log_correct_curve(
        config.EXP3_LOG_CORRECT_SEED, config.EXP3_LOG_CORRECT_RIDGE,
        config.EXP3_LOG_CORRECT_C0, config.EXP3_LOG_CORRECT_P_LOW, config.EXP3_LOG_CORRECT_P_HIGH,
    )
    phi5, _ = _dict5()
    phi4, _ = dictionary.cosine_dictionary(4)

    designs = [
        ("log_correct", n_reps),
        ("dgp_i", n_reps),
        ("dgp_d", n_reps_dpp),
        ("null_p05", n_reps_null),
        ("null_p20", n_reps_null),
    ]

    for design, reps in designs:
        D_reps = []
        se_reps = []
        for rep in range(reps):
            seed = hash((design, rep)) % (2 ** 31)
            if design == "log_correct":
                T_use = T
                periods, _ = _panel_log_correct(
                    T_use, seed, config.EXP3_LOG_CORRECT_C0, config.EXP3_LOG_CORRECT_RIDGE,
                    config.EXP3_LOG_CORRECT_P_LOW, config.EXP3_LOG_CORRECT_P_HIGH,
                )
            elif design == "dgp_i":
                T_use = T
                periods = _panel_independent_or_dpp(T_use, "independent", seed, U_o4)
            elif design == "dgp_d":
                T_use = config.EXP3_DPP_TEST_PANEL_PERIODS
                periods = _panel_independent_or_dpp(T_use, "dpp", seed, U_o4)
            elif design == "null_p05":
                T_use = T
                periods = _panel_null(T_use, config.EXP3_NULL_PROBS[0], seed)
            else:
                T_use = T
                periods = _panel_null(T_use, config.EXP3_NULL_PROBS[1], seed)

            fit_po, fit_log = _fit_pair(periods, m, max_iter=max_iter)
            phi_z = phi5(EVAL_POINTS)
            D, se = _discrepancy_at(fit_po, fit_log, phi_z, T_use)
            D_reps.append(D)
            se_reps.append(se)
        D_reps = np.array(D_reps)
        se_reps = np.array(se_reps)
        mean_D = D_reps.mean(axis=0)
        mc_sd = D_reps.std(axis=0, ddof=1) if reps > 1 else np.full(len(EVAL_POINTS), np.nan)
        mean_se = se_reps.mean(axis=0)
        D_norm = np.sqrt(np.mean(mean_D ** 2))
        se_norm = np.sqrt(np.mean(mean_se ** 2))
        ratio = D_norm / se_norm if se_norm > 0 else np.nan
        for idx, z in enumerate(EVAL_POINTS):
            rows.append({
                "design": design, "z": z, "mean_D": mean_D[idx], "mc_sd": mc_sd[idx],
                "mean_se": mean_se[idx], "D_norm": D_norm, "se_norm": se_norm,
                "ratio": ratio, "n_reps": reps, "T": T_use,
            })

    df = pd.DataFrame(rows)
    df.to_csv("results/exp3_discrepancy.csv", index=False)

    # Population half-discrepancy / larger gap decomposition (Proposition 5.1(i))
    pop_rows = []
    p_o_fn4_dgp, _ = gs.true_curve_fn(U_o4, 4)
    for design_name, p_o_fn in [
        ("log_correct", p_o_log_correct),
        ("dgp_i", p_o_fn4_dgp),
    ]:
        half_D, larger_gap, gap_po, gap_log = _population_targets_dict5(p_o_fn, U_o4, n_pop, seed=999)
        pop_rows.append({
            "design": design_name, "half_D_population": half_D, "larger_gap_population": larger_gap,
            "gap_po_population": gap_po, "gap_log_population": gap_log, "n_pop": n_pop,
        })
    pd.DataFrame(pop_rows).to_csv("results/exp3_population_targets.csv", index=False)

    elapsed = time.time() - t0
    print(f"exp3: wall-clock {elapsed:.1f}s, T={T}, n_reps(main)={n_reps}, "
          f"n_reps(dpp)={n_reps_dpp}, n_reps(null)={n_reps_null}, n_pop={n_pop}")
    print(df.groupby("design")[["D_norm", "se_norm", "ratio"]].first())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="run the paper-scale configuration")
    args = parser.parse_args()
    main(full=args.full)
