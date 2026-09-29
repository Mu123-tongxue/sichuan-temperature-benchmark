# Synthetic demonstration output

`method_comparison_example.csv` is generated from the committed synthetic example inputs using `examples/generate_example_outputs.py`.

The generator calls the real `GeoThermalDoctor.engine_start()` workflow but uses reduced runtime settings inside the demo process so the example can finish quickly. These settings are **not** the manuscript settings, and the resulting values are **not** the Sichuan Basin results reported in the manuscript.

The authoritative source defaults remain 60 initial surrogate candidates, 25 iterations, 2000 candidates per iteration, TOP_K = 6, and 1200 PINN epochs.
