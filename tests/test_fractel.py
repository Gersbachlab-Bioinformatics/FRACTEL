import pytest
import numpy as np
import pandas as pd
from fractel.fractel import element_test

@pytest.fixture
def test_data():
    dhs_df = pd.DataFrame({
        'grna': ['g1', 'g2', 'g3', 'g4'],
        'pvalue': [0.01, 0.05, 0.02, 0.03]
    })
    sim_data = {
        4: np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    }
    return dhs_df, sim_data

def test_element_test_basic(test_data):
    dhs_df, sim_data = test_data
    result = element_test(dhs_df, sim_data, pval_col='pvalue', bnd=0.5)
    assert isinstance(result, float)
    assert 0 <= result <= 1

def test_element_test_return_guide(test_data):
    dhs_df, sim_data = test_data
    result = element_test(dhs_df, sim_data, pval_col='pvalue', bnd=0.5, return_guide=True)
    assert result in dhs_df['grna'].values

def test_element_test_return_index_min_grna(test_data):
    dhs_df, sim_data = test_data
    result = element_test(dhs_df, sim_data, pval_col='pvalue', bnd=0.5, return_index_min_grna=True)
    assert isinstance(result, np.int64)
    assert 0 <= result < len(dhs_df)

def test_element_test_invalid_num_guides(test_data):
    dhs_df, sim_data = test_data
    with pytest.raises(ValueError):
        element_test(dhs_df.iloc[:2], sim_data, pval_col='pvalue', bnd=0.5)
