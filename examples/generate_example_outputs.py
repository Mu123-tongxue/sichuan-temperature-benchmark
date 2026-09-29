"""Generate lightweight synthetic demonstration outputs using the real workflow.

This script modifies runtime globals only inside this process. It never edits the
authoritative source defaults and never uses manuscript study data.
"""
from __future__ import annotations

import os
os.environ.setdefault("MPLBACKEND", "Agg")

from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import geothermal_temperature_benchmark as m


EXAMPLES = ROOT / "examples"
OUTPUT = ROOT / "example_outputs" / "method_comparison_example.csv"


class Value:
    def __init__(self, value):
        self.value = str(value)

    def get(self):
        return self.value


def _layer(name: str, path: Path, conductivity: float, heat_production: float):
    return {
        "name": Value(name),
        "path": Value(path),
        "lam": Value(conductivity),
        "a": Value(heat_production),
    }


def run_demo() -> Path:
    # Reduced runtime settings are process-local and intentionally do not match
    # the manuscript runtime. The scientific source defaults remain unchanged.
    m.N_INIT = 20
    m.N_ITER = 1
    m.N_CAND = 32
    m.TOP_K = 2
    m.PINN_EPOCHS = 6
    m._HAS_SHAP = False

    # Avoid GUI dialogs in headless demonstration execution.
    m.messagebox.showinfo = lambda *args, **kwargs: None
    m.messagebox.showwarning = lambda *args, **kwargs: None
    errors = []
    m.messagebox.showerror = lambda title, message: errors.append((title, message))

    with tempfile.TemporaryDirectory(prefix="sichuan_temp_demo_") as tmp:
        tmp_path = Path(tmp)
        app = m.GeoThermalDoctor.__new__(m.GeoThermalDoctor)
        app.target_dir = Value(tmp_path)
        app.q_path = Value(EXAMPLES / "heat_flow_example.dat")
        app.m_path = Value(EXAMPLES / "measured_temperature_example.csv")
        app.ts = Value("18.0")
        app.test_ratio = Value("0.2")
        app.layers = [
            _layer("Layer_01", EXAMPLES / "stratigraphic_interfaces/layer_01_example.dat", 2.5, 1.0),
            _layer("Layer_02", EXAMPLES / "stratigraphic_interfaces/layer_02_example.dat", 3.0, 0.5),
        ]
        app.update_pb = lambda *args, **kwargs: None

        app.engine_start()
        summary = tmp_path / "method_comparison_summary.csv"
        if errors:
            raise RuntimeError(errors[-1][1])
        if not summary.exists():
            raise RuntimeError("Demo workflow did not create method_comparison_summary.csv")

        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(summary, OUTPUT)

    return OUTPUT


if __name__ == "__main__":
    result = run_demo()
    print(result, flush=True)
    # Some scientific wheels in the local Python 3.13 assembly environment
    # can spend excessive time in interpreter finalization after all files are
    # written. The demo is a one-shot process, so exit immediately after flush.
    os._exit(0)
