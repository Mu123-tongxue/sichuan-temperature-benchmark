# Physics-informed and hybrid residual-learning models for deep formation temperature prediction in the Sichuan Basin

This repository contains the Python implementation accompanying the manuscript **“Physics-informed and hybrid residual-learning models for deep formation temperature prediction in the Sichuan Basin: benchmarking against physics-based and machine-learning methods.”** It provides the implemented benchmarking workflow for the initial layered conductive model, surrogate-assisted thermophysical calibration, a physics-informed neural network (PINN), hybrid PINN-XGBoost residual correction, and SHAP interpretation.

## Scientific scope

The public code preserves the manuscript method and hyperparameters. The manuscript dataset contains 3415 quality-controlled temperature samples and uses a fixed random **sample-level** 80/20 partition: 2732 training samples and 683 samples in the common evaluation subset, with random seed 42. The same sample indices are used across compared methods. During PINN fitting, the evaluation-subset MSE is also used for checkpoint selection; accordingly, the evaluation subset is **not an independent test set**.

## Repository structure

```text
src/                    Authoritative implementation
examples/               Synthetic, publication-safe example inputs
docs/                   Input, model, reproducibility, and traceability documentation
tests/                  Scientific-invariant and smoke tests
example_outputs/         Synthetic demonstration outputs (added in the release workflow)
```

## Python environment

The manuscript workflow was run in **Python 3.10.11**. Direct project dependencies captured from that environment are pinned in `requirements.txt`:

- NumPy 1.26.4
- pandas 2.2.3
- Matplotlib 3.9.4
- SciPy 1.13.1
- scikit-learn 1.6.1
- XGBoost 2.1.4
- PyTorch 2.8.0 (the study CPU build reports `2.8.0+cpu` at runtime)
- SHAP 0.49.1

Install the direct dependencies with:

```bash
python -m pip install -r requirements.txt
```

Tkinter is required for the graphical interface and is commonly supplied with the Python installation or operating-system package manager rather than via pip.

A clean Windows **Python 3.10.11** virtual environment containing only these direct dependencies was also validated successfully with `python -m pip check` and a core-package import smoke test. See [Reproducibility](docs/REPRODUCIBILITY.md).

## Launch

From the repository root:

```bash
python src/geothermal_temperature_benchmark.py
```

The GUI requests an output directory, a heat-flow grid, a measured-temperature CSV, and at least two stratigraphic-interface files with initial thermal conductivity and radiogenic heat-production values.

## Synthetic example data

The files under `examples/` are deterministic **synthetic** inputs created only to verify the public workflow and input schemas. They contain no manuscript well locations, restricted temperature measurements, or proprietary stratigraphic surfaces. The synthetic example **does not reproduce the manuscript** numerical benchmark values or Sichuan Basin maps. Exact reproduction of the reported regional values requires the corresponding study inputs, subject to their data-availability restrictions.

A lightweight demonstration summary can be regenerated with:

```bash
python examples/generate_example_outputs.py
```

The demo reduces runtime settings only inside that process; it does not edit the manuscript defaults in `src/`. The committed `example_outputs/method_comparison_example.csv` is therefore a workflow demonstration, not a paper-results table.

## Main outputs

The full workflow writes, among other files:

- `method_comparison_summary.csv`
- `method_ranking_by_Evaluation_RMSE.csv`
- training/evaluation observed-versus-predicted tables and figures
- calibrated thermophysical-parameter tables for surrogate-assisted schemes
- PINN and hybrid diagnostics
- SHAP values and feature-importance figures for the residual corrector
- 1–8 km temperature surfaces and formation-interface temperature outputs

## Documentation

- [Input data](docs/INPUT_DATA.md)
- [Model description](docs/MODEL_DESCRIPTION.md)
- [Reproducibility](docs/REPRODUCIBILITY.md)
- [Manuscript-to-code map](docs/MANUSCRIPT_CODE_MAP.md)

## Data availability

The repository is designed so that publication-safe synthetic data can be distributed independently of restricted study data. Do not add confidential well coordinates, proprietary formation grids, credentials, or unapproved temperature observations to the public repository.

## Citation

Citation metadata are provided in `CITATION.cff` (software version `1.0.0`).

## License

The code is distributed under the MIT License; see `LICENSE`.
