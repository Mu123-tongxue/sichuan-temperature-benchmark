from pathlib import Path


def test_required_reviewer_documents_exist():
    required = [
        Path('README.md'),
        Path('docs/INPUT_DATA.md'),
        Path('docs/MODEL_DESCRIPTION.md'),
        Path('docs/REPRODUCIBILITY.md'),
        Path('docs/MANUSCRIPT_CODE_MAP.md'),
    ]
    assert all(path.exists() for path in required)


def test_reproducibility_contract_is_explicit():
    text = Path('docs/REPRODUCIBILITY.md').read_text(encoding='utf-8')
    assert '3415' in text
    assert '2732' in text
    assert '683' in text
    assert 'sample-level' in text
    assert 'random seed 42' in text.lower()
    assert 'checkpoint' in text.lower()
    assert 'not an independent test set' in text.lower()
    assert 'Python 3.10' in text
    assert 'No broken requirements found.' in text
    assert 'Environment OK' in text


def test_readme_distinguishes_workflow_from_exact_numerical_reproduction():
    text = Path('README.md').read_text(encoding='utf-8').lower()
    assert 'synthetic' in text
    assert 'does not reproduce the manuscript' in text
    assert 'evaluation subset' in text
    assert 'not an independent test set' in text


def test_manuscript_code_map_names_core_functions():
    text = Path('docs/MANUSCRIPT_CODE_MAP.md').read_text(encoding='utf-8')
    for name in [
        'physics_vec()', 'build_objective()', 'smbo_optimize()',
        'build_pinn_features()', 'train_pinn()', 'build_hybrid_features()',
        'split_sample_indices()', 'calc_metrics()', 'run_shap_analysis()',
    ]:
        assert name in text


def test_input_data_doc_defines_units_and_required_columns():
    text = Path('docs/INPUT_DATA.md').read_text(encoding='utf-8')
    for token in ['X', 'Y', 'Depth', 'temp', 'WellID', 'mW/m²', '°C', 'KDTree', '3.0 °C']:
        assert token in text


def test_submission_metadata_files_exist_and_are_pinned():
    requirements = Path('requirements.txt')
    license_file = Path('LICENSE')
    citation = Path('CITATION.cff')
    gitignore = Path('.gitignore')
    assert requirements.exists()
    assert license_file.exists()
    assert citation.exists()
    assert gitignore.exists()

    req = requirements.read_text(encoding='utf-8')
    for pin in [
        'numpy==1.26.4',
        'pandas==2.2.3',
        'matplotlib==3.9.4',
        'scipy==1.13.1',
        'scikit-learn==1.6.1',
        'xgboost==2.1.4',
        'torch==2.8.0',
        'shap==0.49.1',
    ]:
        assert pin in req


def test_license_citation_and_ignore_metadata_are_submission_ready():
    license_text = Path('LICENSE').read_text(encoding='utf-8')
    assert 'MIT License' in license_text
    assert 'Copyright (c) 2026 Feisheng Mou and co-authors' in license_text

    citation = Path('CITATION.cff').read_text(encoding='utf-8')
    assert 'cff-version: 1.2.0' in citation
    assert 'version: 1.0.0' in citation
    assert 'Physics-informed and hybrid residual-learning models for deep formation temperature prediction in the Sichuan Basin: benchmarking against physics-based and machine-learning methods' in citation
    for family in ['Mou', 'Zuo', 'Yang', 'Chen', 'Zhang', 'Jing', 'Wang', 'Li', 'Cui']:
        assert family in citation

    ignore = Path('.gitignore').read_text(encoding='utf-8')
    for pattern in ['__pycache__/', '.pytest_cache/', '.venv/', '.env', '*.pt', '*.pth', '*.pkl', 'outputs/', 'results/', 'data/raw/', 'data/private/', '.idea/']:
        assert pattern in ignore
    assert 'examples/' not in ignore
    assert 'example_outputs/' not in ignore
