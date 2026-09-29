from pathlib import Path
import re

SOURCE = Path('src/geothermal_temperature_benchmark.py')


def test_source_has_no_cjk_reviewer_text():
    source = SOURCE.read_text(encoding='utf-8')
    assert not re.search(r'[\u4e00-\u9fff]', source)


def test_legacy_test_subset_label_is_not_exposed():
    source = SOURCE.read_text(encoding='utf-8')
    assert 'Test_RMSE' not in source
    assert 'Test_' not in source
    assert 'Test RMSE' not in source
    assert 'Test MAE' not in source
    assert 'Test R2' not in source
    assert 'Test Within5Ratio' not in source
    assert 'TEST' not in source


def test_qualitative_method_commentary_is_removed():
    source = SOURCE.read_text(encoding='utf-8')
    assert 'def build_method_comment' not in source
