from unittest.mock import MagicMock

import numpy as np
import muon as mu 
import pandas as pd
import pytest

from fractel.fractel import element_test, run_fractel_analysis, compute_effect_sizes

@pytest.fixture
def test_data():
    np.random.seed(113)
    dhs_df = pd.DataFrame({
        'grna': [f'g{i}' for i in range(1, 21)],
        'dhs': [f'dhs{1 if i<11 else 2 }' for i in range(1, 21)],
        'pvalue': np.random.random(20),
        "effect_size": [0.5 - i for i in np.random.random(20)]
    })
    sim_data = {
        10: np.random.random(10)
    }
    return dhs_df, sim_data

def test_element_test_with_different_bnd_values(test_data):
    dhs_df, sim_data = test_data
    for gg in dhs_df.groupby('dhs'):
        for bnd in [0.1, 0.25, 0.5, 0.75, 1]:
            result = element_test(gg[1], sim_data, pval_col='pvalue', bnd=bnd)
            assert isinstance(result, float)
            assert 0 <= result <= 1

def test_element_test_with_custom_pval_col():
    dhs_df = pd.DataFrame({
        'grna': [f'g{i}' for i in range(1, 11)],
        'custom_pval': np.random.random(10)
    })
    sim_data = {10: np.random.random(10)}
    result = element_test(dhs_df, sim_data, pval_col='custom_pval', bnd=0.5)
    assert isinstance(result, float)
    assert 0 <= result <= 1

def test_element_test_with_empty_dataframe():
    dhs_df = pd.DataFrame(columns=['grna', 'pvalue'])
    sim_data = {10: np.random.random(10)}
    with pytest.raises(ValueError):
        element_test(dhs_df, sim_data, pval_col='pvalue', bnd=0.5)

def test_element_test_with_missing_pval_column():
    dhs_df = pd.DataFrame({
        'grna': [f'g{i}' for i in range(1, 11)],
        'other_col': np.random.random(10)
    })
    sim_data = {10: np.random.random(10)}
    with pytest.raises(KeyError):
        element_test(dhs_df, sim_data, pval_col='pvalue', bnd=0.5)

def test_element_test_with_large_bnd_value(test_data):
    dhs_df, sim_data = test_data
    for gg in dhs_df.groupby('dhs'):
        result = element_test(gg[1], sim_data, pval_col='pvalue', bnd=6)
        assert isinstance(result, float)
        assert 0 <= result <= 1

def test_element_test_with_small_bnd_min():
    dhs_df = pd.DataFrame({
        'grna': [f'g{i}' for i in range(1, 11)],
        'pvalue': np.random.random(10)
    })
    sim_data = {10: np.random.random(10)}
    result = element_test(dhs_df, sim_data, pval_col='pvalue', bnd=0.5, bnd_min=1)
    assert isinstance(result, float)
    assert 0 <= result <= 1

def test_element_test_basic(test_data):
    dhs_df, sim_data = test_data
    for gg in dhs_df.groupby('dhs'):
        result = element_test(gg[1], sim_data, pval_col='pvalue', bnd=0.5)
        assert isinstance(result, float)
        assert 0 <= result <= 1

def test_element_test_return_guide(test_data):
    dhs_df, sim_data = test_data
    for gg in dhs_df.groupby('dhs'):
        result = element_test(gg[1], sim_data, pval_col='pvalue', bnd=0.5, return_guide=True)
        assert result in dhs_df['grna'].values

def test_element_test_return_index_min_grna(test_data):
    dhs_df, sim_data = test_data
    for gg in dhs_df.groupby('dhs'):
        result = element_test(gg[1], sim_data, pval_col='pvalue', bnd=0.5, return_index_min_grna=True)
        assert isinstance(result, np.int64)
        assert 0 <= result < len(dhs_df)

def test_element_test_invalid_num_guides(test_data):
    dhs_df, sim_data = test_data
    with pytest.raises(ValueError):
        element_test(dhs_df.iloc[:2], sim_data, pval_col='pvalue', bnd=0.5)

@pytest.fixture
def mock_args():
    args = MagicMock()
    args.data_frame = "test_data.tsv"
    args.aggregating_cols = ["dhs"]
    args.usecols = ["grna", "dhs", "pvalue"]
    args.discard_values_in_aggr_cols = None
    args.sim_data = ["tests/test_data/simulated_data.npz"]#["sim_data.npz"]
    args.pval_col = "pvalue"
    args.bnd = 0.5
    args.bnd_min = 3
    args.output_col_basename = "FRACTEL"
    args.fdr_thres = 0.05
    args.row_id_col = "grna"
    return args

@pytest.fixture
def mock_data():
    df = pd.DataFrame({
        "grna": [f"g{i}" for i in range(1, 21)],
        "dhs": [f"dhs{1 if i < 11 else 2}" for i in range(1, 21)],
        "pvalue": np.random.random(20)
    })

    sim_data = {
        10: np.random.random(10)
    }
    return df, sim_data

def test_run_fractel_analysis(mock_args, mock_data, monkeypatch):
    df, sim_data = mock_data

    # Mock file reading
    monkeypatch.setattr(pd, "read_csv", lambda *args, **kwargs: df)

    result_df = run_fractel_analysis(mock_args)

    assert isinstance(result_df, pd.DataFrame)
    assert f"{mock_args.output_col_basename}_pval" in result_df.columns
    assert f"{mock_args.output_col_basename}_pval_fdr_corr" in result_df.columns
    assert not result_df.empty

def test_run_fractel_analysis_with_discard_values(mock_args, mock_data, monkeypatch):
    df, sim_data = mock_data
    mock_args.discard_values_in_aggr_cols = ["dhs1"]

    # Mock file reading
    monkeypatch.setattr(pd, "read_csv", lambda *args, **kwargs: df)
    # monkeypatch.setattr(np, "load", lambda *args, **kwargs: {"simulations": sim_data})

    result_df = run_fractel_analysis(mock_args)

    assert isinstance(result_df, pd.DataFrame)
    assert f"{mock_args.output_col_basename}_pval" in result_df.columns
    assert f"{mock_args.output_col_basename}_pval_fdr_corr" in result_df.columns
    assert not result_df.empty
    assert "dhs1" not in result_df.index

def test_run_fractel_analysis_with_nan_values(mock_args, monkeypatch):
    df = pd.DataFrame({
        "grna": [f"g{i}" for i in range(1, 21)],
        "dhs": [f"dhs{1 if i < 11 else 2}" for i in range(1, 21)],
        "pvalue": [np.nan if i % 5 == 0 else np.random.random() for i in range(1, 21)]
    })
    sim_data = {
        10: np.random.random(10)
    }

    # Mock file reading
    monkeypatch.setattr(pd, "read_csv", lambda *args, **kwargs: df)
    # monkeypatch.setattr(np, "load", lambda *args, **kwargs: {"simulations": sim_data})

    result_df = run_fractel_analysis(mock_args)

    assert isinstance(result_df, pd.DataFrame)
    assert f"{mock_args.output_col_basename}_pval"  in result_df.columns
    assert f"{mock_args.output_col_basename}_pval_fdr_corr" in result_df.columns

def test_compute_effect_sizes_basic():
    df = pd.DataFrame({
        "grna": [f"g{i}" for i in range(1, 11)],
        "dhs": ["dhs1"] * 5 + ["dhs2"] * 5,
        "pvalue": np.random.random(10),
        "Estimate": 0.5 - np.random.random(10)
    })
    result = compute_effect_sizes(df, aggregating_cols="dhs", pval_col="pvalue")
    assert isinstance(result, pd.DataFrame)
    assert "FRACTEL_effect_size" in result.columns
    assert set(result.index) == {"dhs1", "dhs2"}
    assert result["FRACTEL_effect_size"].notnull().all()

def test_compute_effect_sizes_with_nan_values():
    df = pd.DataFrame({
        "grna": [f"g{i}" for i in range(1, 11)],
        "dhs": ["dhs1"] * 5 + ["dhs2"] * 5,
        "pvalue": np.random.random(10),
        "Estimate": [np.nan if i == 3 else 5-i for i in range(10)]
    })
    result = compute_effect_sizes(df, aggregating_cols=["dhs"], pval_col="pvalue")
    assert isinstance(result, pd.DataFrame)
    assert "FRACTEL_effect_size" in result.columns
    assert set(result.index) == {"dhs1", "dhs2"}

def test_compute_effect_sizes_empty_dataframe():
    df = pd.DataFrame(columns=["grna", "dhs", "pvalue", "Estimate"])
    with pytest.raises(ValueError):
        compute_effect_sizes(df, aggregating_cols=["dhs"], pval_col="pvalue")

def test_compute_effect_sizes_missing_value_col():
    df = pd.DataFrame({
        "grna": [f"g{i}" for i in range(1, 11)],
        "dhs": ["dhs1"] * 5 + ["dhs2"] * 5,
        "pvalue": np.random.random(10)
    })
    with pytest.raises(KeyError):
        compute_effect_sizes(df, aggregating_cols=["dhs"], pval_col="value")

def test_compute_effect_sizes_single_group():
    df = pd.DataFrame({
        "grna": [f"g{i}" for i in range(1, 6)],
        "dhs": ["dhs1"] * 5,
        "pvalue": np.random.random(5),
        "Estimate": 0.5 - np.random.random(5)
    })
    result = compute_effect_sizes(df, aggregating_cols=["dhs"], pval_col="pvalue")
    assert isinstance(result, pd.DataFrame)

@pytest.fixture
def mock_mudata_args():
    args = MagicMock()
    args.data_frame = None
    args.mudata = "tests/test_data/test_mudata.h5mu"
    args.mudata_mod = "guide"
    args.mu_uns_key_results = "test_results"
    args.mu_guide_mod = "guide"
    args.mu_guide_id_col = "guide_id"
    args.mu_element_id_cols = ['intended_target_name', 'intended_target_chr', 'intended_target_start', 
                                       'intended_target_end']
    args.aggregating_cols = ["element_id", "gene_id"]
    args.bnd = 1
    args.discard_values_in_aggr_cols = None
    args.sim_data = ["tests/test_data/simulated_data.npz"]
    args.pval_col = "sceptre_p_value"
    args.effect_size_col = "sceptre_log2_fc"
    args.bnd_min = 3
    args.output_col_basename = "FRACTEL"
    args.fdr_thres = 0.05
    args.row_id_col = "grna_id"
    return args

def test_run_fractel_analysis_with_mudata(mock_mudata_args, monkeypatch):
    """Test run_fractel_analysis with mudata input."""
            
    result_df = run_fractel_analysis(mock_mudata_args)
    
    assert isinstance(result_df, pd.DataFrame)
    assert f"{mock_mudata_args.output_col_basename}_pval" in result_df.columns
    assert f"{mock_mudata_args.output_col_basename}_pval_fdr_corr" in result_df.columns
    assert not result_df.empty
