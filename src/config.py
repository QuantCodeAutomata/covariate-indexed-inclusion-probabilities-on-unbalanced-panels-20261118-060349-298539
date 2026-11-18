"""Parameters for every experiment, taken from the paper's Section 6 (simulation
study) and Appendix C.3 (regime of the application), as summarized in the
evidence used to build this repository. Every variable carries a one-line
source note; "not specified in the paper" marks a value the paper leaves open,
for which a documented, reproducible choice was made.

No classes or dictionaries: every parameter is a module-level UPPER_CASE
constant, grouped by the experiment that uses it.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Shared dictionary (Section 6.1): phi_k(z) = cos(omega_k z + varsigma_k)
# ---------------------------------------------------------------------------
OMEGA_M4 = np.array([1.5, 4.0, 8.5, 14.0])          # m=4 frequencies, Section 6.1
VARSIGMA_M4 = np.array([0.3, 1.1, 2.0, 0.7])        # m=4 phases, Section 6.1
OMEGA_M8_EXTRA = np.array([2.7, 6.2, 11.0, 17.5])   # extra frequencies for m=8, Section 6.2
VARSIGMA_M8_EXTRA = np.array([1.9, 0.5, 1.4, 2.6])  # extra phases for m=8, Section 6.2

# Covariate law (Section 6.1): z_{t,i} = tanh(a_t + eps_{t,i}), eps ~ N(0, 0.7^2)
EPS_SD = 0.7                      # innovation std of the covariate, Section 6.1
AR_COEF = 0.9                     # AR(1) coefficient of a_t in the dependent design, Section 6.1
AR_INNOVATION_SD = 0.4            # AR(1) innovation std of a_t, Section 6.1
N_ITEMS_LOW = 15                  # lower bound of n_t ~ Uniform{15,...,45}, Section 6.1
N_ITEMS_HIGH = 45                 # upper bound of n_t ~ Uniform{15,...,45}, Section 6.1

# True curve U^o = s*(W0 W0^T + 0.6 I_4) (Section 6.1); the exact draw of W0 is
# not specified in the paper, so a fixed documented seed is used throughout.
TRUE_CURVE_SEED = 20240601        # seed for the fixed standard normal draw W0, not specified in the paper
RIDGE_BASE = 0.6                  # ridge added to W0 W0^T in U^o, Section 6.1
P_RANGE_LOW = 0.021               # target lower bound of p^o over z, Section 6.1
P_RANGE_HIGH = 0.35               # target upper bound of p^o over z, Section 6.1

# Determinantal point process kernel (Section 6.1): RBF bandwidth
DPP_RBF_BANDWIDTH = 0.15          # bandwidth of K^rbf in the DPP kernel, Section 6.1

# ---------------------------------------------------------------------------
# exp_1: parametric rate and composition floor (Section 6.2, Table 1)
# ---------------------------------------------------------------------------
EXP1_T_GRID_PAPER = [250, 1000, 4000]      # sample sizes, Table 1
EXP1_T_GRID_RUN = [250, 1000, 4000]        # kept identical: these define the rate, not scalable
EXP1_N_REPS_PAPER = 200                    # replications, Table 1 caption
EXP1_N_REPS_RUN = 20                       # reduced scalable repetition count to fit the time limit
EXP1_TEST_PANEL_SIZE = 200                 # fixed test panel for RMSE, Section 6.1
EXP1_POPULATION_PANEL_SIZE_PAPER = 20000   # population target panel, Section 6.1
EXP1_POPULATION_PANEL_SIZE_RUN = 3000      # reduced scalable panel size to fit the time limit
EXP1_N_COVARIATE_BINS = 200                # bins for composition part, Section 6.1

# ---------------------------------------------------------------------------
# exp_2: coverage of confidence intervals for the target curve (Section 6.3, Table 2)
# ---------------------------------------------------------------------------
EXP2_T_GRID = [250, 1000, 4000]            # sample sizes, Table 2
EXP2_N_REPS_PAPER = 500                    # replications, Table 2 caption
EXP2_N_REPS_RUN = 25                       # reduced scalable repetition count to fit the time limit
EXP2_EVAL_POINTS = [-0.5, 0.0, 0.5]        # evaluation points, Table 2
EXP2_NW_LAG_ROBUSTNESS_EXP = 1.3           # Bartlett robustness lag ceil(1.3*sqrt(T)), Table 2 caption

# ---------------------------------------------------------------------------
# exp_3: link discrepancy diagnostic (Section 6.4, Table 3)
# ---------------------------------------------------------------------------
EXP3_T = 1000                              # sample size, Table 3 caption
EXP3_N_REPS_PAPER = 200                    # replications (independent designs), Table 3 caption
EXP3_N_REPS_RUN = 20                       # reduced scalable repetition count to fit the time limit
EXP3_N_REPS_DPP_PAPER = 100                # replications under DGP-D, Table 3 caption
EXP3_N_REPS_DPP_RUN = 10                   # reduced scalable repetition count to fit the time limit
EXP3_N_REPS_NULL_PAPER = 500               # replications at the null, Table 3 caption
EXP3_N_REPS_NULL_RUN = 30                  # reduced scalable repetition count to fit the time limit
EXP3_EVAL_POINTS = [-0.75, -0.25, 0.25, 0.75]  # evaluation points, Section 6.4
EXP3_DPP_TEST_PANEL_PERIODS = 60           # test panel length under DGP-D, Section 6.4
EXP3_NULL_PROBS = [0.05, 0.20]             # constant margins at the null, Section 6.4
EXP3_LOG_CORRECT_RIDGE = 0.6               # ridge for U_4 in the log-correct design, inferred from Section 6.1's U^o form
EXP3_LOG_CORRECT_SEED = 20240602           # seed for W1, not specified in the paper
EXP3_LOG_CORRECT_C0 = 0.0                  # constant offset c_0 in the log-correct design, not specified in the paper
EXP3_LOG_CORRECT_P_LOW = 0.02              # target lower bound of p^o, Section 6.4
EXP3_LOG_CORRECT_P_HIGH = 0.30             # target upper bound of p^o, Section 6.4
EXP3_POPULATION_PANEL_SIZE_PAPER = 20000   # population target panel, Section 6.4
EXP3_POPULATION_PANEL_SIZE_RUN = 3000      # reduced scalable panel size to fit the time limit

# ---------------------------------------------------------------------------
# exp_6: regime of the application with known truth (Appendix C.3, Tables 9-11)
# ---------------------------------------------------------------------------
EXP6_D_COVARIATES = 10                      # covariate dimension, Appendix C.3
EXP6_FACTOR_LOADING = 0.3                   # common-factor weight in z_j, Appendix C.3
EXP6_IDIOSYNCRATIC_LOADING = 0.7            # idiosyncratic weight in z_j, Appendix C.3
EXP6_N_ITEMS_LOW_PAPER = 3450                # lower bound of items per period, Appendix C.3
EXP6_N_ITEMS_HIGH_PAPER = 5750                # upper bound of items per period, Appendix C.3
EXP6_N_ITEMS_LOW_RUN = 400                   # reduced universe size to fit the time limit
EXP6_N_ITEMS_HIGH_RUN = 600                   # reduced universe size to fit the time limit
EXP6_EVENT_RATE = 0.0025                     # mean failure rate, Appendix C.3
EXP6_AR1_LOG_RATE_COEF = 0.95                # AR(1) coefficient of log c_t under drift, Appendix C.3
EXP6_AR1_LOG_RATE_INNOV_SD = 0.1             # AR(1) innovation std of log c_t, Appendix C.3
EXP6_TRAIN_WINDOW = 97                       # training window length (months), Appendix C.3
EXP6_N_TEST_PERIODS_PAPER = 100              # test periods, Appendix C.3
EXP6_N_TEST_PERIODS_RUN = 10                 # reduced scalable count to fit the time limit
EXP6_N_REPS_PAPER = 50                       # replications, Appendix C.3
EXP6_N_REPS_RUN = 5                          # reduced scalable repetition count to fit the time limit
EXP6_M_GRID_LINEAR = 11                      # linear-kernel dictionary size (d+1), Appendix C.3
EXP6_M_GRID_RICH = 40                        # linear+Gaussian dictionary size, Appendix C.3
EXP6_N_BINS = 10                             # bins of true margin / fitted probability, Appendix C.3
EXP6_Q_A_COEFS_A = np.array([1.0, -1.0] + [0.3] * (EXP6_D_COVARIATES - 2))  # not specified in the paper
EXP6_Q_A_COEFS_B = np.array([-1.0, 1.0] + [-0.2] * (EXP6_D_COVARIATES - 2))  # not specified in the paper
EXP6_DESIGN_B_C0 = 2.0                       # constant c_0 in design B, not specified in the paper
EXP6_SEED = 20240603                         # seed for the covariate draw, not specified in the paper

# ---------------------------------------------------------------------------
# exp_7: confidence bands at boundary targets of the psd cone (Appendix C.1, Tables 7-8)
# ---------------------------------------------------------------------------
EXP7_T_GRID = [250, 1000, 4000]              # sample sizes, Table 8
EXP7_N_REPS_PAPER = 500                      # replications, Table 8 caption
EXP7_N_REPS_RUN = 25                         # reduced scalable repetition count to fit the time limit
EXP7_EVAL_POINTS = [-0.5, 0.0, 0.5]          # evaluation points, Table 8
EXP7_RANK_TOLERANCE = 1e-4                   # relative eigenvalue tolerance for the fit rank, Table 7 caption
EXP7_THRESHOLD_CT = "sqrt(log T)"            # c_T = sqrt(log T) in Proposition C.1
EXP7_WIDE_RIDGE = 4.0                        # ridge of the "wide" zero-multiplier design, Table 7 caption
EXP7_GAMMA_ZERO_DROP = 0.026                 # weight on phi_4^2 removed to force Gamma=0, Appendix C.1
EXP7_N_PROJECTION_DRAWS = 400                # Monte Carlo draws to simulate the projection band quantile

# ---------------------------------------------------------------------------
# Common optimizer settings for the custom projected-gradient (SPG-style) solver
# ---------------------------------------------------------------------------
SOLVER_MAX_ITER = 500                        # iteration cap, not specified in the paper
SOLVER_GRAD_TOL = 1e-6                       # gradient tolerance, Section 7.1 (ex post check)
SOLVER_REL_DECREASE_TOL = 1e-11              # relative available-decrease tolerance, Section 7.1
LAMBDA_DEFAULT = 0.0                         # unregularized fits unless stated, Section 6.1

CONFIDENCE_LEVEL = 0.95                      # nominal coverage level, Table 2 caption

# ---------------------------------------------------------------------------
# exp_4 / exp_5: US stock failure application (Section 7, Tables 4-6, Fig. 2-3)
# ---------------------------------------------------------------------------
# The paper's panel is CRSP common stock on NYSE/AMEX/NASDAQ, Jan 1963-Nov
# 2024, with 121 of the 153 characteristics of Jensen, Kelly and
# Pedersen (2023) and CRSP delisting codes 500/520-599 as the failure event
# (Section 7.1). CRSP, Compustat and the Jensen-Kelly-Pedersen characteristic
# panel are not served by the QuantX Dataloader or by Massive, so this
# replication substitutes a reduced, OHLCV-derived characteristic panel and a
# severe-decline/inactivity failure proxy built from the Dataloader; see
# results/RESULTS.md, "Deviations from the paper".
EXP4_START_DATE_PAPER = "1963-01-01"         # CRSP sample start, Section 7.1
EXP4_END_DATE_PAPER = "2024-11-30"           # CRSP sample end, Section 7.1
EXP4_START_DATE_RUN = "2005-01-01"           # executed start: earliest date with reliable Dataloader daily coverage
EXP4_END_DATE_RUN = "2023-12-31"             # executed end: leaves a one-year lookahead buffer for the failure proxy
EXP4_UNIVERSE_SIZE_PAPER = None              # "all CRSP common stocks", no fixed count, Section 7.1
EXP4_UNIVERSE_SIZE_RUN = 150                 # tickers, cut by dollar volume (liquidity) at the sample start
EXP4_TRAIN_WINDOW_PAPER = 97                 # months, Section 7.1
EXP4_TRAIN_WINDOW_RUN = 36                   # months, reduced to fit the time limit with EXP4_START/END_DATE_RUN
EXP4_N_TEST_MONTHS_PAPER = 646               # test months, March 1971-Dec 2024, Section 7.1
EXP4_N_TEST_MONTHS_RUN = 60                  # test months, reduced to fit the time limit
EXP4_FAILURE_DRAWDOWN = -0.70                # monthly return below which a name is flagged as a failure proxy, not specified in the paper (CRSP delisting codes unavailable)
EXP4_N_SUMMARIES = 5                         # rank of ||z||, missing share, >20% loss share, its 12m mean, cross-sectional IQR of log returns, Section 7.1
EXP4_GRID_M = [5, 10, 20, 40]                # grid-dictionary sizes, Section 7.1
EXP4_RICH_LAMBDA_RULE = "T**-0.5"            # richer-dictionary penalty lambda = T^{-1/2}, Section 7.1
EXP4_NW_LAGS = 6                             # Newey-West lags for Diebold-Mariano / Giacomini-White t-stats, Section 7.1
EXP4_CALIBRATION_N_BINS = 10                 # reliability bins, Section 7.1
EXP4_LOG_SCORE_CLIP = 1e-6                   # probability clip for the log score, Section 7.1
EXP4_CHS_PREDICTORS = [
    "NIMTAAVG", "TLMTA", "EXRETAVG", "SIGMA", "RSIZE", "CASHMTA", "MB", "PRICE",
]                                             # Campbell, Hilscher and Szilagyi (2008) failure-logit predictors, Section 7.1

# ---------------------------------------------------------------------------
# Common plotting / output settings
# ---------------------------------------------------------------------------
RESULTS_DIR = "results"                      # all tables, plots and logs are written directly here
RANDOM_SEED_DEFAULT = 0                       # fallback seed, not specified in the paper
