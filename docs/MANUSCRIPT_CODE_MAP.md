# Manuscript-to-code traceability

Authoritative implementation: `src/geothermal_temperature_benchmark.py`.

| Manuscript method | Source implementation | Role |
|---|---|---|
| §3.2 Initial one-dimensional layered conductive model | `physics_vec()` | layered steady-state conductive temperature calculation |
| §3.3 Surrogate-assisted thermophysical calibration | `build_objective()`, `smbo_optimize()` | calibration objective and sequential surrogate search |
| §3.4 Physics-informed neural-network prediction | `build_pinn_features()`, `train_pinn()`, `pinn_predict()` | feature construction, constrained training, prediction |
| §3.5 Hybrid PINN-XGBoost residual correction | `build_hybrid_features()` plus the XGBoost residual block in `engine_start()` | residual target construction and additive correction |
| §3.6 Data partitioning and unified evaluation | `split_sample_indices()`, `calc_metrics()` | deterministic sample-level split and common metrics |
| §3.8 SHAP interpretation | `run_shap_analysis()` | residual-corrector explanation |

## Preprocessing and spatial matching

- `read_measure_csv()` validates and parses temperature observations.
- `clean_measurements()` merges duplicate depths, determines surface temperature, and removes implemented downhole anomalies.
- `read_xyz_dat()` parses heat-flow and interface triplets.
- `spatial_join_z()` performs nearest-neighbor stratigraphic-interface matching.

## Benchmark outputs and Results support

Within `engine_start()`:

- the initial conductive, RF-assisted, XGBoost-assisted, MLP-assisted, PINN, and hybrid blocks populate the common method summary that supports the overall benchmark comparison in the Results;
- `method_comparison_summary.csv` and `create_method_comparison_plots()` support the manuscript's aggregate method-comparison metrics;
- per-method observed-versus-predicted CSV/figures support the observed-versus-predicted diagnostics;
- PINN-XGBoost SHAP calls support the residual-correction interpretation;
- the final 1–8 km loop produces `temperature_map_<depth>m` outputs;
- the formation-interface loop produces formation top/bottom temperature summaries and maps.
