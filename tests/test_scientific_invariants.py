import numpy as np
import pytest

from src import geothermal_temperature_benchmark as m


def test_manuscript_constants():
    assert m.RANDOM_SEED == 42
    assert m.LAM_BOUNDS == (0.60, 1.70)
    assert m.A_BOUNDS == (0.60, 1.70)
    assert m.N_INIT == 60
    assert m.N_ITER == 25
    assert m.N_CAND == 2000
    assert m.TOP_K == 6
    assert m.PINN_EPOCHS == 1200
    assert m.PINN_BATCH_SIZE == 256
    assert m.PINN_LR == 1e-3
    assert m.PINN_DATA_W == 1.0
    assert m.PINN_PHYS_W == 0.5
    assert m.PINN_MONO_W == 0.2
    assert m.PINN_BOUNDARY_W == 0.3
    assert m.PINN_WD == 1e-5


def test_manuscript_sample_split_is_deterministic():
    train_a, evaluation_a = m.split_sample_indices(3415, 0.20, 42)
    train_b, evaluation_b = m.split_sample_indices(3415, 0.20, 42)
    assert len(train_a) == 2732
    assert len(evaluation_a) == 683
    assert np.array_equal(train_a, train_b)
    assert np.array_equal(evaluation_a, evaluation_b)
    assert set(train_a).isdisjoint(set(evaluation_a))


def test_single_layer_conductive_temperature():
    app = m.GeoThermalDoctor.__new__(m.GeoThermalDoctor)
    pred = app.physics_vec(
        q_arr=np.array([60.0]),
        d_matrix=np.array([[2000.0]]),
        l_list=np.array([2.5]),
        a_list=np.array([1.0]),
        target_z=np.array([1000.0]),
        ts=np.array([18.0]),
    )
    assert pred[0] == pytest.approx(41.8)


def test_metrics_include_within_five_and_exceedance():
    app = m.GeoThermalDoctor.__new__(m.GeoThermalDoctor)
    metrics = app.calc_metrics(np.array([10.0, 20.0]), np.array([12.0, 28.0]))
    assert metrics["Within5Ratio"] == pytest.approx(0.5)
    assert metrics["MeanExceedOver5"] == pytest.approx(1.5)
