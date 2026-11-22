# Group-Conditional Adaptive Conformal Inference (GACI)

Reproduction and manuscript project: marginal adaptive conformal coverage
can conceal severe subgroup undercoverage; GACI restores per-group
coverage at essentially unchanged mean width (width is
reallocated toward the groups that need it), exactly when groups differ in
conditional difficulty.

- `experiments/` — e1..e10 experiment scripts + ci_analysis.py + figures.py
- `results/` — metrics JSONs (incl. ci_metrics.json), per-step series NPZ, figures
- `notebooks/exploration.ipynb` — interactive exploration of saved artifacts (no retraining)
- `src/` — ACI/GACI implementation
- `WORKSHOP.md` — workshop-version manuscript (the paper)
- `THEORY.md` — claim boundaries and honest limitations

## Reproduce

```bash
conda activate credit-risk_VENV   # or: conda run -n credit-risk_VENV python ...
python -m pytest tests/ -q              # 16 unit tests for src/aci.py
python experiments/e4_group_aci.py     # headline experiment
python experiments/e9_unseen_groups.py # unseen-group fallback (E9)
python experiments/figures.py          # regenerate figures
```

Datasets (not committed): Lending Club Kaggle slice (200k rows, see
data-prep notes in experiments) and UCI household_power_consumption.txt
(place in `data/`). Environment pins: `requirements.txt`.

Note on determinism: E1 uses an explicit seed; E2-E9 perform no random
operations (closed-form ridge, deterministic pandas reads), so reruns
reproduce `results/` bit-for-bit apart from floating-point noise. E8's
LightGBM sets `seed=42` but multi-threaded histogram building can vary
slightly between runs.
