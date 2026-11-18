"""Simulation Study: Coverage of Confidence Intervals for the Target Curve
(paper Section 6.3, Table 2).

Fits ell_U(z) = phi(z)^T U phi(z) by the identity-link Poisson loss on
synthetic panels under two designs for the inclusion mechanism given the
margins: DGP-I (independent Bernoulli inclusions, so the per-period score is
a martingale difference) and DGP-D (a repulsive determinantal point process,
whose cross-sectional composition induces serial dependence in the score).
Three long-run-variance estimators for the pointwise asymptotic variance of
the fitted curve are compared by their empirical coverage of nominal 95%
intervals over many replications: the martingale estimator G(0) (Theorem
4.3(v)), the Bartlett/Newey-West long-run variance, and the equal-weighted
cosine (EWC) estimator of Lazarus, Lewis, Stock and Watson (2018), which uses
a Student-t critical value with its own degrees of freedom.

Run with `python -m exp.exp2` (reduced replication count) or
`python -m exp.exp2 --full` (paper replication count, Table 2 caption).
"""

from __future__ import annotations

import argparse
import os
import time
from multiprocessing import get_context

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from data import generate_synthetic as gs
from src import config, dictionary, linalg, psd_model, variance

# (label, diagonal, m) for the three fit specifications compared in Table 2.
DICT_SPECS = [("full_m4", False, 4), ("diagonal_m4", True, 4), ("full_m8", False, 8)]
DGPS = ["independent", "dpp"]
ESTIMATORS = ["martingale", "bartlett", "bartlett_robust", "ewc"]
SEED_BASE = 20240700


def _job_list(n_reps: int) -> list[tuple]:
    """Deterministic (T, dgp, spec, rep, seed) jobs in a fixed loop order so
    that seeds never collide and results are reproducible across runs."""
    jobs = []
    counter = 0
    for T in config.EXP2_T_GRID:
        for dgp in DGPS:
            for spec_label, diagonal, m in DICT_SPECS:
                for rep in range(n_reps):
                    seed = SEED_BASE + counter
                    counter += 1
                    jobs.append((T, dgp, spec_label, diagonal, m, rep, seed))
    return jobs


def _run_one(job: tuple, U_o: np.ndarray) -> list[dict]:
    """Build one replicated panel, fit it, and return per-(z, estimator)
    coverage records. Periods are discarded as soon as the fit dictionary
    rows / inclusion masks have been extracted, to keep memory flat across
    replications."""
    T, dgp, spec_label, diagonal, m, rep, seed = job

    periods, _ = gs.build_panel(T, dependent=True, dgp=dgp, m=4, U_o=U_o, seed=seed)
    if m == 4:
        fit_periods = [(p.phi, p.incl) for p in periods]
    else:
        phi8, _ = dictionary.cosine_dictionary(8)
        fit_periods = [(phi8(p.z), p.incl) for p in periods]
    del periods

    # m=4 fits converge (grad_tol) in well under 150 Newton iterations; the
    # full_m8 spec is ill-conditioned enough that it rarely reaches grad_tol
    # even at the fit() default of 500 iterations (cost grows with T and with
    # M = m(m+1)/2), so the iteration budget is capped here to keep the full
    # (T up to 4000) x (dict spec) x (dgp) x (replication) grid tractable;
    # this only trims the tail of near-converged Newton steps for full_m8; it
    # does not change the m=4 fits, which already converge well inside the cap.
    fit_result = psd_model.fit(
        periods=fit_periods, m=m, link="identity", diagonal=diagonal, lam=0.0,
        max_iter=150,
    )
    del fit_periods

    p_o_fn, _ = gs.true_curve_fn(U_o, 4)
    phi_fit, _ = dictionary.cosine_dictionary(m)

    H_inv = variance.pinv(fit_result.hessian_vec)
    lag = variance.newey_west_lag(T)
    lag_robust = int(np.ceil(config.EXP2_NW_LAG_ROBUSTNESS_EXP * np.sqrt(T)))

    Q_mart = variance.martingale_variance(fit_result.period_scores)
    Q_bart = variance.bartlett_long_run_variance(fit_result.period_scores, lag)
    Q_bart_robust = variance.bartlett_long_run_variance(fit_result.period_scores, lag_robust)
    Q_ewc, nu = variance.ewc_variance(fit_result.period_scores)

    records = []
    for z in config.EXP2_EVAL_POINTS:
        phi_z = phi_fit(np.array([z]))[0]
        C_z = linalg.outer_vec(phi_z)
        ell_hat = float(psd_model.curve(fit_result.U, phi_z[None, :])[0])
        p_true = float(p_o_fn(np.array([z]))[0])
        err = abs(ell_hat - p_true)

        halves = {
            "martingale": variance.curve_band(C_z, H_inv, Q_mart, T, level=config.CONFIDENCE_LEVEL),
            "bartlett": variance.curve_band(C_z, H_inv, Q_bart, T, level=config.CONFIDENCE_LEVEL),
            "bartlett_robust": variance.curve_band(
                C_z, H_inv, Q_bart_robust, T, level=config.CONFIDENCE_LEVEL
            ),
            "ewc": variance.curve_band(
                C_z, H_inv, Q_ewc, T, level=config.CONFIDENCE_LEVEL, dof=nu
            ),
        }
        for estimator in ESTIMATORS:
            half = halves[estimator]
            records.append(
                {
                    "T": T,
                    "dgp": dgp,
                    "dict_spec": spec_label,
                    "z": z,
                    "estimator": estimator,
                    "rep": rep,
                    "covered": bool(err <= half),
                    "half_width": float(half),
                    "ell_hat": ell_hat,
                }
            )
    return records


def _run_jobs(jobs: list[tuple], U_o: np.ndarray, n_workers: int) -> pd.DataFrame:
    all_records: list[dict] = []
    t0 = time.time()
    if n_workers <= 1:
        results_iter = (_run_one(job, U_o) for job in jobs)
    else:
        ctx = get_context("fork")
        pool = ctx.Pool(processes=n_workers)
        results_iter = pool.imap_unordered(_WorkerFn(U_o), jobs, chunksize=1)

    done = 0
    for records in results_iter:
        all_records.extend(records)
        done += 1
        if done % max(1, len(jobs) // 20) == 0 or done == len(jobs):
            elapsed = time.time() - t0
            print(
                f"  [{done}/{len(jobs)} jobs done, {elapsed:6.1f}s elapsed, "
                f"eta {elapsed / done * (len(jobs) - done):6.1f}s]",
                flush=True,
            )
    if n_workers > 1:
        pool.close()
        pool.join()
    return pd.DataFrame.from_records(all_records)


class _WorkerFn:
    """Picklable callable binding U_o for multiprocessing.Pool.imap_unordered."""

    def __init__(self, U_o: np.ndarray):
        self.U_o = U_o

    def __call__(self, job: tuple) -> list[dict]:
        return _run_one(job, self.U_o)


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["T", "dgp", "dict_spec", "z", "estimator"]
    agg = df.groupby(group_cols).agg(
        coverage=("covered", "mean"),
        n_reps=("covered", "count"),
        mean_half_width=("half_width", "mean"),
        mc_std_ell_hat=("ell_hat", "std"),
    )
    agg = agg.reset_index().sort_values(group_cols).reset_index(drop=True)
    agg["mc_std_ell_hat"] = agg["mc_std_ell_hat"].fillna(0.0)
    return agg


def _plot_coverage(agg: pd.DataFrame, out_path: str) -> None:
    """One subplot per evaluation point, x=T (log scale), y=coverage, one line
    per estimator, solid for DGP-I and dashed for DGP-D. Restricted to the
    headline 'full_m4' specification (Table 2's primary comparison) to keep
    the figure readable; the CSV carries every (T, dgp, dict_spec) cell."""
    spec_label = "full_m4"
    sub = agg[agg["dict_spec"] == spec_label]
    z_points = config.EXP2_EVAL_POINTS
    colors = {"martingale": "C0", "bartlett": "C1", "bartlett_robust": "C2", "ewc": "C3"}
    dgp_style = {"independent": "-", "dpp": "--"}
    dgp_label = {"independent": "DGP-I", "dpp": "DGP-D"}

    fig, axes = plt.subplots(1, len(z_points), figsize=(5 * len(z_points), 4.2), sharey=True)
    if len(z_points) == 1:
        axes = [axes]
    for ax, z in zip(axes, z_points):
        for dgp in DGPS:
            for estimator in ESTIMATORS:
                cell = sub[(sub["z"] == z) & (sub["dgp"] == dgp) & (sub["estimator"] == estimator)]
                cell = cell.sort_values("T")
                if cell.empty:
                    continue
                ax.plot(
                    cell["T"], cell["coverage"],
                    color=colors[estimator], linestyle=dgp_style[dgp], marker="o",
                    label=f"{estimator} ({dgp_label[dgp]})",
                )
        ax.axhline(config.CONFIDENCE_LEVEL, color="black", linewidth=1.0, linestyle=":")
        ax.set_xscale("log")
        ax.set_xticks(config.EXP2_T_GRID)
        ax.set_xticklabels([str(t) for t in config.EXP2_T_GRID])
        ax.set_xlabel("T")
        ax.set_title(f"z = {z}")
        ax.set_ylim(0.0, 1.02)
    axes[0].set_ylabel("empirical coverage")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle(
        "Exp 2: coverage of nominal 95% confidence intervals for the target curve "
        f"(dictionary spec = {spec_label})"
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _print_se_mcsd_diagnostic(agg: pd.DataFrame) -> None:
    """Implied-se / Monte-Carlo-sd ratio: ratio close to 1 means the reported
    half-width's implied standard error matches the actual sampling spread of
    ell_hat across replications; ratio << 1 signals under-estimated variance
    (likely under-coverage) and ratio >> 1 over-estimated variance."""
    crit = stats.norm.ppf(0.5 + config.CONFIDENCE_LEVEL / 2)
    out = agg.copy()
    out["implied_se"] = out["mean_half_width"] / crit
    with np.errstate(divide="ignore", invalid="ignore"):
        out["se_over_mcsd"] = out["implied_se"] / out["mc_std_ell_hat"].replace(0.0, np.nan)
    print("\nImplied-se / Monte-Carlo-sd diagnostic (ratio near 1 is well-calibrated):")
    cols = ["T", "dgp", "dict_spec", "z", "estimator", "se_over_mcsd"]
    print(out[cols].to_string(index=False))


def main(full: bool = False) -> None:
    n_reps = config.EXP2_N_REPS_PAPER if full else config.EXP2_N_REPS_RUN
    n_workers = min(4, os.cpu_count() or 1)

    print("=" * 70)
    print("exp_2: Coverage of Confidence Intervals for the Target Curve")
    print(f"T grid           : {config.EXP2_T_GRID}")
    print(f"Replications     : {n_reps} ({'paper' if full else 'reduced run'})")
    print(f"Eval points      : {config.EXP2_EVAL_POINTS}")
    print(f"Dict specs       : {[s[0] for s in DICT_SPECS]}")
    print(f"DGPs             : {DGPS}")
    print(f"Seed base        : {SEED_BASE}")
    print(f"Worker processes : {n_workers}")
    print("=" * 70)

    U_o, p_min, p_max = gs.true_curve_matrix(m=4)
    print(f"True curve p^o range over the fit grid: [{p_min:.4f}, {p_max:.4f}]")

    jobs = _job_list(n_reps)
    print(f"Total fits to run: {len(jobs)}")

    t_start = time.time()
    df = _run_jobs(jobs, U_o, n_workers)
    elapsed = time.time() - t_start
    print(f"\nAll fits done in {elapsed:.1f}s ({elapsed / 60:.2f} min).")

    agg = _aggregate(df)

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(config.RESULTS_DIR, "exp2_coverage.csv")
    plot_path = os.path.join(config.RESULTS_DIR, "exp2_coverage_plot.png")
    agg.to_csv(csv_path, index=False)
    _plot_coverage(agg, plot_path)

    print(f"\nWrote {csv_path}")
    print(f"Wrote {plot_path}")

    _print_se_mcsd_diagnostic(agg)

    print("\nCoverage at z=0.5, dict_spec=full_m4 (DGP-I vs DGP-D undercoverage/recovery check):")
    sub = agg[(agg["z"] == 0.5) & (agg["dict_spec"] == "full_m4")]
    print(sub[["T", "dgp", "estimator", "coverage"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 2: coverage of confidence intervals for the target curve")
    parser.add_argument(
        "--full", action="store_true",
        help="use the paper's replication count (EXP2_N_REPS_PAPER) instead of the reduced run count",
    )
    args = parser.parse_args()
    main(full=args.full)
