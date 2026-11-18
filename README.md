# Covariate-Indexed Inclusion Probabilities on Unbalanced Panels

Replication of selected experiments from *Covariate-Indexed Inclusion
Probabilities on Unbalanced Panels*.

The paper studies a positive-semidefinite (psd) sum-of-squares model for the
probability that an item is included in an unbalanced cross-sectional panel,
fit by an identity-link (Poisson quasi-likelihood) or log-link (Bernoulli
likelihood) loss over the cone of psd matrices. It develops the parametric
rate of the fitted curve, a composition floor under cross-sectionally
dependent inclusion (a determinantal point process), pointwise confidence
bands for the curve (including at rank-deficient, boundary targets of the
cone), a link-discrepancy diagnostic between the two model classes, and an
application to monthly US stock failure prediction.

## Repository layout

```
src/     psd sum-of-squares model, variance/confidence-band estimators,
         link-discrepancy diagnostic, dictionaries, linear algebra, config
exp/     one script per reproduced experiment, run as python -m exp.<name>
data/    data/generate_synthetic.py (simulation panels) and
         data/data_loader.py (QuantX Dataloader OHLCV bars for exp_4/exp_5)
tests/   pytest unit and (market-data) integration tests
results/ tables, plots and RESULTS.md with the paper comparison
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in DATALOADER_API_KEY and MASSIVE_API_KEY for exp_4/exp_5
```

## Tests

```bash
python -m pytest -m "not market_data"   # unit tests, no network access
python -m pytest                        # also runs the market-data integration test (needs API keys)
```

## Experiments

Each experiment runs in a reduced configuration by default (so the full suite
finishes in the time available) and in the paper's full configuration with
`--full`. Every run overwrites its output files in `results/`.

```bash
python -m exp.exp1   # Simulation: parametric rate and composition floor (Section 6.1-6.2)
python -m exp.exp2   # Simulation: coverage of confidence intervals for the curve (Section 6.3)
python -m exp.exp3   # Simulation: link discrepancy diagnostic (Section 6.4)
python -m exp.exp4   # Application: US stock failure prediction (Section 7.1-7.3, 7.5)
python -m exp.exp5   # Application: link discrepancy diagnostic on the failure panel (Section 7.4)
python -m exp.exp6   # Simulation: regime of the application with known truth (Appendix C.3)
python -m exp.exp7   # Simulation: confidence bands at boundary targets of the psd cone (Appendix C.1)
python -m exp.exp1 --full   # paper-scale configuration (see results/RESULTS.md for runtimes)
```

`reproduce.sh` runs the full test suite and every experiment at its default
(time-limited) configuration.

See `results/RESULTS.md` for the numbers next to the paper, every deviation
from the paper's data and methodology, and the experiments not reproduced.
See `AGENTS.md` for a short contributor-facing guide to the repository.
