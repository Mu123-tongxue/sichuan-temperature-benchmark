# Reproducibility contract

## Environment

The manuscript workflow was run with **Python 3.10.11**. The direct project dependencies captured from that environment are: NumPy 1.26.4, pandas 2.2.3, Matplotlib 3.9.4, SciPy 1.13.1, scikit-learn 1.6.1, XGBoost 2.1.4, PyTorch 2.8.0 (CPU runtime reports `2.8.0+cpu`), and SHAP 0.49.1. The same pins are listed in `requirements.txt`.


## Clean-environment validation

A fresh Windows virtual environment was created with **Python 3.10.11** and only the direct dependencies pinned in `requirements.txt`. Installation completed successfully. The clean environment returned:

```text
No broken requirements found.
```

for `python -m pip check`, and the import smoke test

```bash
python -c "import numpy,pandas,scipy,sklearn,xgboost,torch,shap; print('Environment OK')"
```

returned:

```text
Environment OK
```

This clean-environment check is distinct from the original manuscript development environment, which also contained unrelated packages not required by this repository.

## Dataset partition used in the manuscript

- Quality-controlled samples: **3415**
- Training subset: **2732**
- Common evaluation subset: **683**
- Partition: random **sample-level** 80/20 split
- **Random seed 42**
- Identical partition indices are used across compared methods.

The 683-sample evaluation subset is also evaluated after each PINN epoch for **checkpoint selection**, and the PINN state with minimum evaluation-subset MSE is retained. Therefore, this subset is **not an independent test set** and is not described as an independent external validation set in this repository.

## Manuscript-critical defaults

- Thermal-conductivity multiplier bounds: 0.60–1.70
- Radiogenic heat-production multiplier bounds: 0.60–1.70
- Surrogate search: `N_INIT=60`, `N_ITER=25`, `N_CAND=2000`, `TOP_K=6`
- PINN: hidden layers 256–256–128, `tanh`, 1200 epochs, batch size up to 256, learning rate `1e-3`, weight decay `1e-5`
- PINN loss weights: data 1.0, conductive prior 0.5, monotonicity 0.2, surface boundary 0.3
- Hybrid XGBoost: 1000 trees, learning rate 0.03, max depth 5, minimum child weight 3, row subsampling 0.85, feature subsampling 0.85, L1 0, L2 1.0, random seed 42

## Result files used for benchmark reconstruction

The full workflow writes `method_comparison_summary.csv` and `method_ranking_by_Evaluation_RMSE.csv`, together with per-method prediction tables and figures. These files contain the training and evaluation metrics used for cross-method comparison. SHAP outputs are specific to the XGBoost residual-correction stage.

## Public synthetic example versus exact manuscript reproduction

The committed example inputs are synthetic and are intended to verify parsing, cleaning, spatial matching, conductive prediction, and reduced-runtime model execution. They do not contain the full Sichuan Basin study inputs and therefore cannot reproduce the manuscript's reported RMSE/MAE/R² values or regional temperature maps. Exact numerical reproduction requires the study inputs used in the manuscript, subject to their release permissions.
