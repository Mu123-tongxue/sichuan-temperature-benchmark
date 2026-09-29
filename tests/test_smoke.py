from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from scipy.spatial import KDTree

from src import geothermal_temperature_benchmark as m

EXAMPLES = Path('examples')


class _Value:
    def __init__(self, value):
        self.value = str(value)

    def get(self):
        return self.value



def _app(tmp_path):
    app = m.GeoThermalDoctor.__new__(m.GeoThermalDoctor)
    app.target_dir = _Value(tmp_path)
    return app


def test_example_inputs_parse_and_produce_finite_conductive_output(tmp_path):
    app = _app(tmp_path)
    heat = app.read_xyz_dat(EXAMPLES / 'heat_flow_example.dat', names=('X', 'Y', 'Q'))
    layer1 = app.read_xyz_dat(EXAMPLES / 'stratigraphic_interfaces/layer_01_example.dat')
    layer2 = app.read_xyz_dat(EXAMPLES / 'stratigraphic_interfaces/layer_02_example.dat')
    measurements = app.read_measure_csv(EXAMPLES / 'measured_temperature_example.csv')
    measurements, _ = app.clean_measurements(measurements, default_ts=18.0)

    grid_xy = heat[['X', 'Y']].to_numpy(float)
    tree = KDTree(grid_xy)
    _, idx = tree.query(measurements[['X', 'Y']].to_numpy(float))
    d_matrix = np.column_stack([
        app.spatial_join_z(grid_xy, layer1, 'Layer 1'),
        app.spatial_join_z(grid_xy, layer2, 'Layer 2'),
    ])
    pred = app.physics_vec(
        q_arr=heat['Q'].to_numpy(float)[idx],
        d_matrix=d_matrix[idx],
        l_list=np.array([2.5, 3.0]),
        a_list=np.array([1.0, 0.5]),
        target_z=measurements['Depth'].to_numpy(float),
        ts=measurements['Ts'].to_numpy(float),
    )
    assert len(pred) == len(measurements)
    assert np.isfinite(pred).all()


def test_missing_required_measurement_column_raises_clear_error(tmp_path):
    app = _app(tmp_path)
    path = tmp_path / 'missing_temp.csv'
    pd.DataFrame({'X': [0], 'Y': [0], 'Depth': [100]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match='missing required column: temp'):
        app.read_measure_csv(path)


def test_missing_wellid_is_synthesized_and_surface_temperature_falls_back(tmp_path):
    app = _app(tmp_path)
    path = tmp_path / 'no_wellid.csv'
    pd.DataFrame({
        'X': [10.0, 10.0], 'Y': [20.0, 20.0],
        'Depth': [500.0, 1000.0], 'temp': [30.0, 45.0],
    }).to_csv(path, index=False)
    raw = app.read_measure_csv(path)
    cleaned, ts_map = app.clean_measurements(raw, default_ts=18.0)
    assert cleaned['WellID'].nunique() == 1
    well_id = cleaned['WellID'].iloc[0]
    assert well_id.startswith('WELL_')
    assert ts_map[well_id] == pytest.approx(18.0)
    assert np.allclose(cleaned['Ts'], 18.0)


def test_zero_depth_measurement_overrides_default_surface_temperature(tmp_path):
    app = _app(tmp_path)
    path = tmp_path / 'surface.csv'
    pd.DataFrame({
        'X': [10.0, 10.0], 'Y': [20.0, 20.0],
        'Depth': [0.0, 1000.0], 'temp': [16.5, 44.0], 'WellID': ['S1', 'S1'],
    }).to_csv(path, index=False)
    raw = app.read_measure_csv(path)
    cleaned, ts_map = app.clean_measurements(raw, default_ts=18.0)
    assert ts_map['S1'] == pytest.approx(16.5)
    assert np.allclose(cleaned['Ts'], 16.5)


def test_non_numeric_xyz_rows_are_excluded(tmp_path):
    app = _app(tmp_path)
    path = tmp_path / 'xyz.dat'
    path.write_text('0 0 60\n1 BAD 61\n2 2 62\n', encoding='utf-8')
    cleaned = app.read_xyz_dat(path, names=('X', 'Y', 'Q'), bad_export_name='bad_rows.csv')
    assert len(cleaned) == 2
    assert np.isfinite(cleaned[['X', 'Y', 'Q']].to_numpy(float)).all()
    assert (tmp_path / 'bad_rows.csv').exists()


def test_reduced_surrogate_search_executes(tmp_path):
    app = _app(tmp_path)
    q = np.array([55.0, 60.0, 62.0, 58.0])
    d = np.array([[1500.0, 3000.0]] * 4)
    z = np.array([500.0, 1000.0, 1800.0, 2500.0])
    ts = np.array([18.0] * 4)
    base_lams = np.array([2.5, 3.0])
    base_as = np.array([1.0, 0.5])
    t = app.physics_vec(q, d, base_lams, base_as, z, ts) + np.array([0.2, -0.3, 0.1, 0.0])
    obj = app.build_objective(base_lams, base_as, q, d, z, t, ts)
    surrogate = RandomForestRegressor(n_estimators=10, random_state=42)
    theta, loss, history = app.smbo_optimize(
        'smoke_RF', surrogate, obj,
        bounds=[m.LAM_BOUNDS] * 2 + [m.A_BOUNDS] * 2,
        n_init=4, n_iter=1, n_cand=8, top_k=2, out_dir=tmp_path,
    )
    assert theta.shape == (4,)
    assert np.isfinite(loss)
    assert not history.empty


def test_optional_scientific_packages_are_truly_optional():
    code = r'''
import builtins
import scipy.stats
import sklearn.model_selection
real_import = builtins.__import__
blocked = {'xgboost', 'torch', 'shap'}
def blocking_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name.split('.')[0] in blocked:
        raise ModuleNotFoundError(name)
    return real_import(name, globals, locals, fromlist, level)
builtins.__import__ = blocking_import
from src import geothermal_temperature_benchmark as m
assert m._HAS_XGB is False
assert m._HAS_TORCH is False
assert m._HAS_SHAP is False
print('optional imports skipped')
'''
    result = subprocess.run([sys.executable, '-c', code], cwd=Path.cwd(), text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert 'optional imports skipped' in result.stdout


def test_example_generator_does_not_change_manuscript_defaults_after_process_exit():
    generator = Path('examples/generate_example_outputs.py')
    assert generator.exists()
    source = generator.read_text(encoding='utf-8')
    for assignment in ['m.N_INIT = 20', 'm.N_ITER = 1', 'm.N_CAND = 32', 'm.TOP_K = 2', 'm.PINN_EPOCHS = 6']:
        assert assignment in source

    before = (m.N_INIT, m.N_ITER, m.N_CAND, m.TOP_K, m.PINN_EPOCHS)
    code = "from src import geothermal_temperature_benchmark as x; x.N_INIT=1; x.N_ITER=1; x.N_CAND=1; x.TOP_K=1; x.PINN_EPOCHS=1"
    result = subprocess.run([sys.executable, '-c', code], cwd=Path.cwd(), timeout=20)
    assert result.returncode == 0
    after = (m.N_INIT, m.N_ITER, m.N_CAND, m.TOP_K, m.PINN_EPOCHS)
    assert after == before == (60, 25, 2000, 6, 1200)


def test_example_method_comparison_output_has_expected_columns():
    path = Path('example_outputs/method_comparison_example.csv')
    assert path.exists()
    df = pd.read_csv(path)
    expected = {
        'Method', 'Kind',
        'Train_RMSE', 'Train_MAE', 'Train_R2', 'Train_Within5Ratio',
        'Evaluation_RMSE', 'Evaluation_MAE', 'Evaluation_R2', 'Evaluation_Within5Ratio',
    }
    assert expected.issubset(df.columns)
