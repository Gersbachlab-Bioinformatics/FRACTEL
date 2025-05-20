from unittest.mock import MagicMock
import numpy as np
import pandas as pd
import pytest

from fractel.fractel import interpolate_pvalues, run_interpolate


@pytest.fixture
def interpolate_test_data():
    df = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "pvalue": [0.01, 0.2, 0.5, 0.8, 0.99]
    })
    reference_df = pd.DataFrame({
        "id": [10, 11, 12, 13, 14],
        "pvalue": [0.05, 0.15, 0.25, 0.75, 0.95],
        "group": ["A", "A", "B", "B", "A"]
    })
    return df, reference_df

def test_interpolate_pvalues_basic(interpolate_test_data):
    df, reference_df = interpolate_test_data
    result = interpolate_pvalues(df.copy(), reference_df, pval_col="pvalue", interpolated_col="interp")
    assert "interp" in result.columns
    assert np.all((result["interp"] >= 0) & (result["interp"] <= 1))
    assert len(result) == len(df)

def test_interpolate_pvalues_with_different_col_names():
    df = pd.DataFrame({"x": [0.1, 0.2, 0.3]})
    ref = pd.DataFrame({"x": [0.05, 0.15, 0.25, 0.35]})
    result = interpolate_pvalues(df.copy(), ref, pval_col="x", interpolated_col="y")
    assert "y" in result.columns
    assert np.all((result["y"] >= 0) & (result["y"] <= 1))

def test_interpolate_pvalues_empty_df():
    df = pd.DataFrame(columns=["pvalue"])
    ref = pd.DataFrame({"pvalue": [0.1, 0.2, 0.3]})
    result = interpolate_pvalues(df.copy(), ref)
    assert "interpolated_pvalue" in result.columns
    assert result.empty

def test_interpolate_pvalues_empty_reference():
    df = pd.DataFrame({"pvalue": [0.1, 0.2, 0.3]})
    ref = pd.DataFrame(columns=["pvalue"])
    with pytest.raises(ValueError):
        interpolate_pvalues(df.copy(), ref)

@pytest.fixture
def interpolate_args(tmp_path, interpolate_test_data):
    df, reference_df = interpolate_test_data
    df_path = tmp_path / "df.tsv"
    ref_path = tmp_path / "ref.tsv"
    df.to_csv(df_path, sep="\t", index=False)
    reference_df.to_csv(ref_path, sep="\t", index=False)
    args = MagicMock()
    args.data_frame = str(df_path)
    args.reference_data_frame = str(ref_path)
    args.pval_col = "pvalue"
    args.interpolated_col = "interp"
    args.output_basename = str(tmp_path / "output")
    args.reference_df_select_col = None
    args.reference_df_select_value = None
    return args, df, reference_df

def test_run_interpolate_basic(interpolate_args):
    args, df, reference_df = interpolate_args
    result = run_interpolate(args)
    assert "interp" in result.columns
    assert len(result) == len(df)
    assert np.all((result["interp"] >= 0) & (result["interp"] <= 1))

def test_run_interpolate_with_reference_selection(tmp_path, interpolate_test_data):
    df, reference_df = interpolate_test_data
    df_path = tmp_path / "df.tsv"
    ref_path = tmp_path / "ref.tsv"
    df.to_csv(df_path, sep="\t", index=False)
    reference_df.to_csv(ref_path, sep="\t", index=False)
    args = MagicMock()
    args.data_frame = str(df_path)
    args.reference_data_frame = str(ref_path)
    args.pval_col = "pvalue"
    args.interpolated_col = "interp"
    args.output_basename = str(tmp_path / "output")
    args.reference_df_select_col = "group"
    args.reference_df_select_value = "A"
    result = run_interpolate(args)
    assert "interp" in result.columns
    assert len(result) == len(df)
    assert np.all((result["interp"] >= 0) & (result["interp"] <= 1))

def test_run_interpolate_with_reference_selection_no_rows(tmp_path, interpolate_test_data):
    df, reference_df = interpolate_test_data
    df_path = tmp_path / "df.tsv"
    ref_path = tmp_path / "ref.tsv"
    df.to_csv(df_path, sep="\t", index=False)
    reference_df.to_csv(ref_path, sep="\t", index=False)
    args = MagicMock()
    args.data_frame = str(df_path)
    args.reference_data_frame = str(ref_path)
    args.pval_col = "pvalue"
    args.interpolated_col = "interp"
    args.output_basename = str(tmp_path / "output")
    args.reference_df_select_col = "group"
    args.reference_df_select_value = "Z"  # not present
    with pytest.raises(AssertionError):
        run_interpolate(args)

