# AGENTS.md

Guide for coding agents and contributors working on this repository.

## What this repository implements

A replication of selected experiments from *Covariate-Indexed Inclusion
Probabilities on Unbalanced Panels*: a positive-semidefinite (psd)
sum-of-squares model for the probability that an item is included in an
unbalanced cross-sectional panel, fit by an identity-link (Poisson
quasi-likelihood) or log-link (Bernoulli likelihood) loss over the cone of
psd matrices (Section 3), its parametric-rate and composition-floor
behaviour under independent versus determinantal-point-process inclusion
(Section 6.1), pointwise confidence bands for the fitted curve under
several long-run-variance estimators (Section 6.3), and a link-discrepancy
diagnostic between the identity-link and log-link model classes
(Section 6.4). The US stock failure application (Sections 7.1-7.5), its
link-discrepancy diagnostic (Section 7.4), the simulation mimicking that
application (Appendix C.3), and the boundary-target confidence bands of
Appendix C.1 are not implemented; see `results/RESULTS.md` → "Not
reproduced" for the reasons.

## Layout

```
src/     psd sum-of-squares model (psd_model.py), variance/confidence-band
         estimators (variance.py), link-discrepancy diagnostic
         (diagnostics.py), dictionaries (dictionary.py), linear algebra
         helpers (linalg.py), all experiment parameters (config.py)
exp/     exp1.py, exp2.py, exp3.py — one script per reproduced experiment,
         run as `python -m exp.<name>`
data/    generate_synthetic.py (simulation panels for exp1-exp3)
tests/   pytest unit tests for src/ and exp/
results/ exp1_rmse.csv, exp2_coverage.csv, exp3_discrepancy.csv and
         RESULTS.md with the comparison against the paper
```

## Commands

Setup:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tests (no network access needed; all data is synthetic for exp1-exp3):

```bash
python -m pytest -m "not market_data"
```

Experiments, at the reduced configuration actually used for the results in
`results/RESULTS.md`:

```bash
python -m exp.exp1
python -m exp.exp2
python -m exp.exp3
```

The paper's full Monte Carlo replication counts and population-panel sizes
(Section 6):

```bash
python -m exp.exp1 --full
python -m exp.exp2 --full
python -m exp.exp3 --full
```

`reproduce.sh` runs the test suite and all three experiment scripts with
these same commands.

## Conventions

- Every experiment parameter lives in `src/config.py` as a module-level
  `UPPER_CASE` variable with a comment naming its meaning, unit and source
  (a paper section, or "not specified in the paper"). No parameter is
  hard-coded inside `src/` or `exp/`.
- All data preparation lives in `data/`: `data/generate_synthetic.py`
  generates the simulation panels used by exp1-exp3. `data/data_loader.py`
  loads OHLCV bars through the QuantX Dataloader REST API; it is kept for
  the market-data integration test in `tests/conftest.py` (skipped when
  `DATALOADER_API_KEY`/`MASSIVE_API_KEY` are absent) but no implemented
  experiment calls it, because none of exp1-exp3 uses market data (see
  "Not reproduced" in `results/RESULTS.md` for the application experiments
  that would have needed it).
- Library code is flat modules directly under `src/`; experiment scripts
  only in `exp/`; all tests only in `tests/`. Experiments run as
  `python -m exp.<name>` from the repository root; tests as
  `python -m pytest`. `sys.path` is never modified in code.
- Every run of an experiment script overwrites its CSV/plot outputs in
  `results/`. The prose, tables and verdicts in `results/RESULTS.md` are
  not regenerated automatically — after a rerun, check the new files and
  update the numbers and verdicts by hand.
- No API keys are used by this repository; if a future experiment needs
  market data, keys belong only in an uncommitted `.env` (see
  `.env.example` for the expected variable names and where to obtain them).
