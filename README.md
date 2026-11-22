# Group-Conditional Adaptive Conformal Inference (GACI)

Reproduction and manuscript project: marginal adaptive conformal coverage
can conceal severe subgroup undercoverage; GACI restores per-group
coverage at ~2x interval width, exactly when groups differ in
conditional difficulty.

- `experiments/` — e1..e9 experiment scripts + figures.py
- `results/` — metrics JSONs, per-step series NPZ, figures
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
(place in `data/`).
