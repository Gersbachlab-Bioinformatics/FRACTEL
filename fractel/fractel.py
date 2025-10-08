#!/usr/bin/env python
# coding: utf-8
"""
FRACTEL: Framework for Rank Aggregation of CRISPR Tests within ELements
"""

import argparse
import gc
import logging
from importlib.metadata import version
from functools import reduce

import mudata as mu
import numpy as np
import pandas as pd
from scipy.stats import beta
from scipy.interpolate import interp1d
from scipy.stats import ecdf, kstest
from statsmodels.stats.multitest import fdrcorrection
import seaborn as sns
import matplotlib.pyplot as plt

# Global constants
PSEUDOCOUNT = 1e-10
logging.basicConfig(level=logging.INFO)
rng = np.random.default_rng(seed=42)
mu.set_options(pull_on_update=False)

def interpolate_pvalues(df, reference_df, pval_col='pvalue', 
                        interpolated_col='interpolated_pvalue'):
    """
    Interpolate the p-values in a dataframe based on the distribution of p-values in a reference dataframe.

    Parameters:
    - df: DataFrame containing the p-values to interpolate.
    - reference_df: DataFrame containing the reference distribution of p-values.
    - pval_col: Column name in `df` containing the p-values to interpolate.
    - interpolated_col: Column name to store the interpolated p-values in `df`.

    Returns:
    - DataFrame with an additional column containing the interpolated p-values.
    """

    # Convert p-values to numeric and sort
    p = reference_df[pval_col].astype(float)

    # Check if reference_pvals is empty
    if p.size == 0:
        raise ValueError("Reference p-values array is empty. Please provide a non-empty reference dataframe.")

    p_s = np.sort(p)

    # Calculate ECDF
    _ecdf = ecdf(p_s)
    cdf_p = _ecdf.cdf.evaluate(p_s)

    # Midpoint adjustment of CDF approximation
    p2 = np.concatenate(([0], cdf_p))
    p3 = np.diff(p2)
    p2[1:] = p2[1:] - p3/2
    p4 = np.concatenate((p2, [1]))
    p_s_temp = np.concatenate(([0], p_s, [1]))

    # Create interpolation function
    f_p = interp1d(p_s_temp, p4, bounds_error=False, fill_value=(0,1))

    # Transform p-values
    p_targeting = df[pval_col].astype(float)
    df[interpolated_col] = f_p(p_targeting)

    return df

def element_test(dhs_df, sim_data, pval_col='pvalue', bnd=None, bnd_min=3,
                 row_id_col='grna', return_guide=False, return_index_min_grna=False):
    """
    Perform the FRACTEL test on a given dataframe of grouped data.
    """
    num_guides = dhs_df.shape[0]

    if num_guides not in sim_data:
        raise ValueError(f"Number of guides {num_guides} not found in simulated data.\
                          Please run the simulation with this number of guides.")
    if bnd < 1:
        bnd = max(int(round(bnd*num_guides)),min(num_guides, bnd_min))
    else:
        bnd = min(bnd, num_guides)
    pvals = dhs_df.sort_values(by=pval_col).head(bnd).loc[:,pval_col]
    aux = np.arange(1, num_guides + 1)[:bnd]
    beta_evals = beta.cdf(pvals, aux, (num_guides - aux + 1))
    pmin = np.min(beta_evals)

    if return_guide or return_index_min_grna:
        grnas = dhs_df[row_id_col]
        if return_index_min_grna:
            return np.argmin(beta_evals)
        grna = grnas.values[np.argmin(beta_evals)]
        return grna
    return (sim_data[num_guides] < pmin).sum() / sim_data[num_guides].size

def merge_dicts(x, y):
    """
    Merge two dictionaries, ignoring if a key is already found.
    """
    for key in y:
        if key not in x:
            x[key] = y[key]
    return x    

def run_simulation(args):
    """
    Simulate data for FRACTEL analysis.
    """
    if len(args.num_guides) == 0:
        raise ValueError("Please provide a list of integers for the number of guides to simulate.")
    
    sim_dict = {}
    for m in [int(m) for m in args.num_guides]:
        if args.bnd < 1:
            bnd = max(int(round(args.bnd*m)),min(m,args.bnd_min))
        else:
            bnd = min(args.bnd, m)
        pvals = np.sort(np.random.random((args.num_simulations, m)))
        aux = np.arange(1, m + 1)
        beta_evals = beta.cdf(pvals, aux, m - aux + 1)
        sim_dict[m] = np.min(beta_evals[:,:bnd], axis=1)
    return sim_dict

def save_simulation_data(sim_dict, output_basename):
    """
    Save the simulated dictionary data to a file.
    """
    np.savez(f'{output_basename}', simulations=sim_dict)
    logging.info("Simulated dictionary data saved to %s.npz under the key 'simulations'.", 
                 output_basename)

def load_and_filter_df(args):
    """
    Load and filter the data frame based on the provided arguments.
    """
    # Load the data frame
    df = pd.read_csv(
        args.data_frame,
        sep='\t',
        usecols=args.usecols
    )

    # If a keyword for background values is specified, create a separate dataframe for background
    df_background = None
    if args.keyword_for_background_values and len(args.aggregating_cols) > 0:
        df_background = df[df[args.aggregating_cols[0]] == args.keyword_for_background_values]
        df = df[df[args.aggregating_cols[0]] != args.keyword_for_background_values]

    # Discard rows with NaN values in the aggregating columns
    df = df[~df[args.aggregating_cols[0]].isna()]

    # If specified, discard rows with certain values in the aggregating columns
    if args.discard_values_in_aggr_cols:
        discard_mask = reduce(
            lambda x, y: x | y,
            [df[col].isin(args.discard_values_in_aggr_cols) for col in args.aggregating_cols]
        )
        df = df[~discard_mask]
    return df, df_background

def load_and_filter_mudata(args):
    """
    Load and filter the MuData object based on the provided arguments.
    """
    # Load the MuData object
    mu_data = mu.read_h5mu(args.mudata)
    
    # Create a DataFrame from the MuData object results
    df = pd.DataFrame(mu_data.uns[args.mu_uns_key_results])

    # Create a copy of the guide metadata from the MuData object
    metadata_df = mu_data[args.mu_guide_mod].var.copy()

    # Close mudata, as is no longer needed
    mu_data = None
    gc.collect()

    # Create a new column in the data frame with the element IDs
    metadata_df[args.aggregating_cols[0]] = metadata_df[args.mu_element_id_cols]\
        .astype(str).agg('_'.join, axis=1)

    # Merge the data frame with the guide metadata in the MuData object
    df = df.merge(
        metadata_df.loc[:, [args.mu_guide_id_col, args.aggregating_cols[0]]],
        on=args.mu_guide_id_col,
        how='left'
    )    

    # If a keyword for background values is specified, create a separate dataframe for background
    df_background = None
    if args.keyword_for_background_values and len(args.aggregating_cols) > 0:
        df_background = df[df[args.aggregating_cols[0]] == args.keyword_for_background_values]
        df = df[df[args.aggregating_cols[0]] != args.keyword_for_background_values]

    # If specified, discard rows with certain values in the aggregating columns
    if args.discard_values_in_aggr_cols:
        discard_mask = reduce(
            lambda x, y: x | y,
            [df[col].isin(args.discard_values_in_aggr_cols) for col in args.aggregating_cols]
        )
        df = df[~discard_mask]

    # Return the filtered data frame
    return df, df_background


def load_and_filter_data(args):
    """
    Load and filter the data frame or MuData object based on the provided arguments.
    """
    if args.data_frame is not None:
        return load_and_filter_df(args)
    return load_and_filter_mudata(args)

def check_sizes_in_sim_data(group_sizes, sim_data):
    """
    Check that all group sizes are present in the simulated data dictionary.

    Parameters:
    - group_sizes: iterable of group sizes to check
    - sim_data: dictionary with simulated data, keys are group sizes

    Raises:
    - ValueError if any group size is missing in sim_data
    """
    missing_sizes = [size for size in group_sizes if size not in sim_data]
    if missing_sizes:
        raise ValueError(
            f"The following group sizes are missing in the simulated data: {missing_sizes}. "
            "Please run the simulation for these group sizes."
        )

def run_fractel_analysis(args):
    """
    Run the FRACTEL analysis based on the provided arguments. Returns a DataFrame with the results.
    """

    df, df_background = load_and_filter_data(args)
    if args.keyword_for_background_value and df_background.size >0:
        uniform_test = check_pvalues_uniform(df_background[args.pval_col])
        if uniform_test['is_uniform']:
            logging.info("Background p-values appear to be uniformly distributed (p-value: %.4e; statistic: %.4f). Proceeding with FRACTEL test.",
                         uniform_test['p_value'], uniform_test['statistic'])
        else:
            save_qq_plot(
                df_background,
                pvalue_col=args.pval_col,
                output_basename=f'{args.output_basename}_background_pvalues'
            )
            if args.ignore_miscalibration:
                logging.warning("Background p-values do not appear to be uniformly distributed (p-value: %.4e; statistic: %.4f). \
                                You have chosen to ignore this miscalibration, but please consider running the `calibrate` command to adjust your p-values (recommended).",
                                uniform_test['p_value'], uniform_test['statistic']
                )
            else:
                logging.error("Background p-values do not appear to be uniformly distributed (p-value: %.4e; statistic: %.4f). " \
                "Please consider running the `calibrate` command to adjust your p-values (recommended) if you haven't done so already. " \
                "Alternatively you can use `--ignore-miscalibration` to skip this error.",
                uniform_test['p_value'], uniform_test['statistic']
                )
                raise ValueError("Background p-values do not appear to be uniformly distributed. Please calibrate your p-values or use --ignore-miscalibration to proceed.")

    # Load the simulated data
    sim_data = list(map(lambda x: np.load(x, allow_pickle = True)['simulations'].item(), args.sim_data))
    if len(sim_data) > 1:
        reduce(merge_dicts, sim_data)
    sim_data = sim_data[0]
    
    # Pre-flight check that all group sizes are present in the simulated data
    group_sizes = sorted([int(c) for c in df.groupby(args.aggregating_cols).count().iloc[:, -1].unique()])
    check_sizes_in_sim_data(group_sizes, sim_data)

    # Group the data frame by the specified columns and apply the FRACTEL test
    df_tmp = (
        df.groupby(args.aggregating_cols)
        [df.columns]
        .apply(lambda x: element_test(
            x,
            sim_data,
            pval_col=args.pval_col,
            bnd=args.bnd,
            bnd_min=args.bnd_min,
            row_id_col=args.row_id_col,
            return_guide=False
        ))
        .to_frame()
    )
    df_tmp.columns = [f'{args.output_col_basename}_pval']  # rename the column FRACTEL p-values

    # Apply FDR correction to FRACTEL p-values
    _, fdr_pvals_tmp = fdrcorrection(
        df_tmp[f'{args.output_col_basename}_pval'], alpha=args.fdr_thres, method='indep', is_sorted=False
    )
    df_tmp[f'{args.output_col_basename}_pval_fdr_corr'] = fdr_pvals_tmp

    # Return the resulting data frame
    return df_tmp

def save_df_to_tsv(df, output_basename, keep_index=False):
    """
    Save the results data frame to a compressed TSV file.
    """
    df.to_csv(f'{output_basename}.tsv.gz', sep='\t', index = keep_index, compression='gzip')
    logging.info("Results saved to %s.tsv.gz", output_basename)

def check_pvalues_uniform(pvalues, significance_level=0.05):
    """
    Test if a distribution of p-values is uniform using a chi-square test.

    Parameters:
    - pvalues: array-like, p-values to test
    - significance_level: float, significance level for the test (default: 0.05)
    - num_bins: int, number of bins for the chi-square test (default: 10)

    Returns:
    - dict containing:
        - is_uniform: bool, True if distribution appears uniform
        - statistic: float, chi-square test statistic
        - p_value: float, p-value from chi-square test
    """

    # # Perform chi-square test
    ks_stat, p_val = kstest(sorted(pvalues), 'uniform')

    return {
        'is_uniform': p_val > significance_level,
        'statistic': ks_stat,
        'p_value': p_val
    }

def save_qq_plot(df, pvalue_col, output_basename, minus_log10_pval_zero_rep=300, dpi=300):
    """
    Create and save a Q-Q plot comparing expected vs observed values using seaborn styling.

    Parameters:
    - df: DataFrame containing the values to plot
    - pvalue_col: Column name for expected values
    - output_basename: Base name for output file
    - minus_log10_pval_zero_rep: Value to replace -log10(0) (default: 300)
    - dpi: Resolution for output image (default: 300)

    Returns:
    - None
    """

    # Set seaborn style
    sns.set_style("whitegrid")
    sns.set_context("paper")

    # Define the number of samples to draw
    nnt = df.shape[0]

    # Uniform sampling between 0 and 1 (theoretical p-values dist)
    s = rng.uniform(0, 1, nnt)

    # Extract theoretical quantiles
    nquantiles = 1e3
    quantile_array = np.arange(0, 1+1./nquantiles, 1./nquantiles)
    x = np.quantile(-np.log10(sorted(s)), q=quantile_array)

    # recover observed quantiles for targeting
    y_targeting = np.quantile(
        -np.log10(sorted(df[pvalue_col])), # .sample(nnt, random_state=113) 
        q=quantile_array
    )
    y_targeting = np.nan_to_num(
        y_targeting, 
        nan=max(int(y_targeting[~np.isnan(y_targeting)].max()*1.05),
                minus_log10_pval_zero_rep)
    )

    # Create scatterplot
    _, ax = plt.subplots(figsize=[8, 8])
    ax.hist(df[pvalue_col], rasterized=True, bins=200, color='darkgray', alpha=.75)
    ax.set_ylabel(f'Frequency')
    ax.set_title('Histogram of p-values', pad=20)
    sns.despine()
    plt.tight_layout()
    plt.savefig(f'{output_basename}_background_pval_hist.pdf', dpi=dpi, bbox_inches='tight')
    plt.close()

    # Create scatterplot
    _, ax = plt.subplots(figsize=[8, 8])
    ax.scatter(x, y_targeting, s=10, color='darkorange', alpha=.75,
                rasterized=True)

    # Add diagonal line
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, 'k--', alpha=0.75, linewidth=1)
    
    # Labels and title
    ax.set_xlabel('Expected -log10(uniform p-values)')
    ax.set_ylabel(f'Observed -log10({pvalue_col})')
    ax.set_title('Q-Q plot for distribution of expected vs observed p-values', pad=20)

    # Adjust layout
    sns.despine()
    plt.tight_layout()

    # Save plot
    plt.savefig(f'{output_basename}_background_qq_plot.pdf', dpi=dpi, bbox_inches='tight')
    plt.close()

def run_calibration(args):
    """
    Interpolate p-values in a data frame based on a reference data frame and save the result.
    """
    df = pd.read_csv(args.data_frame, sep='\t')
    reference_df = pd.read_csv(args.reference_data_frame, sep='\t')
    if args.reference_df_select_col and args.reference_df_select_value:
        reference_df = reference_df[reference_df[args.reference_df_select_col] == args.reference_df_select_value]
        assert len(reference_df) > 0, "No rows found in the reference dataframe with the specified selection criteria."
    df = interpolate_pvalues(
        df,
        reference_df,
        pval_col=args.pval_col,
        interpolated_col=args.interpolated_col
    )
    return df

def compute_effect_sizes(df, aggregating_cols, pval_col, effect_size_col="Estimate",
                          pval_type="two-sided", output_col_basename="FRACTEL"):
    """
    Calculate effect size defined as the maximum weighted average across observations in an element.
    
    Parameters:
    - df: DataFrame containing the data
    - aggregating_cols: List of columns to group by
    - pval_col: Column name with p-values
    - effect_size_col: Column name with effect sizes
    - pval_type: Type of p-value ('left', 'right', or 'two-sided')
    - output_col_basename: Base name for the output column
    
    Returns:
    - DataFrame with computed effect sizes
    """
    
    # Add left and right p-values
    if pval_type == 'two-sided':
        df['pval_left'] = df[pval_col]/2 
        df.loc[df[effect_size_col]>0, 'pval_left'] = \
                (1 - df.loc[df[effect_size_col]>0, 'pval_left']).values
        df['pval_right'] = 1 - df['pval_left']
    else:
        df[f'pval_{pval_type}'] = df[pval_col]
        df[f'pval_{"right" if pval_type == "left" else "left" }_pval'] = 1 - df[f'pval_{pval_type}']

    # Compute TW estimator for left and right p-value
    df_tmp = df.reset_index().groupby(aggregating_cols)\
         [df.columns]\
        .apply(lambda x: ((1-x.pval_left)*x[effect_size_col]).sum()/
               max((1-x.pval_left).sum(), PSEUDOCOUNT))
    
    # Create a new DataFrame to store the results
    df_tmp = pd.DataFrame(df_tmp)
    df_tmp.columns = ['tw_estimator_pval_left']
    df_tmp['tw_estimator_pval_right'] = df.reset_index().groupby(aggregating_cols)\
        [df.columns]\
        .apply(lambda x: ((1-x.pval_right)*x[effect_size_col]).sum()/
               max((1-x.pval_right).sum(), PSEUDOCOUNT))
    df = None
    # Import and run garbage collector to free up memory
    gc.collect()

    # When tw_estimator is NA, assign 1 (this comes when the p-values are straight 1s)
    df_tmp.loc[df_tmp.tw_estimator_pval_left.isna(),'tw_estimator_pval_left'] = 0
    df_tmp.loc[df_tmp.tw_estimator_pval_right.isna(),'tw_estimator_pval_right'] = 0

    df_tmp = df_tmp.reset_index()

    # Set TW estimator as the largest effect size between left and right in absolute value
    # Get the effect size as the maximum absolute value between left and right estimators
    left_abs = np.abs(df_tmp['tw_estimator_pval_left'])
    right_abs = np.abs(df_tmp['tw_estimator_pval_right'])
    
    # Create mask for where left absolute values are greater than right
    left_greater_mask = left_abs > right_abs
    
    # Initialize with right estimator values
    df_tmp['tw_estimator'] = df_tmp['tw_estimator_pval_right']
    
    # Where left is greater, use left estimator values
    df_tmp.loc[left_greater_mask, 'tw_estimator'] = df_tmp.loc[left_greater_mask, 'tw_estimator_pval_left']
    
    # Drop tw_estimator_pval_left and tw_estimator_pval_right columns
    df_tmp.drop(columns=['tw_estimator_pval_left', 'tw_estimator_pval_right'], inplace=True)
    
    # Rename the column to match the output column basename
    df_tmp.rename(columns={'tw_estimator': f'{output_col_basename}_effect_size'}, inplace=True)
    
    # Last, set the index to the original index of the data frame
    df_tmp = df_tmp.set_index(aggregating_cols)
    return df_tmp

def validate_args(args):
    """
    Validate the command line arguments.
    """
    col = getattr(args, 'reference_df_select_col', None)
    val = getattr(args, 'reference_df_select_value', None)
    if (col is None and val is not None) or (col is not None and val is None):
        raise ValueError("--reference-df-select-col and --reference-df-select-value must be used together")
    if hasattr(args, 'bnd') and args.bnd is not None and args.bnd >= 1:
        args.bnd = int(args.bnd)
    if hasattr(args, 'data_frame'):
        if hasattr(args, 'effect_size_col') and args.effect_size_col is not None:
            try:
                # Try to check if effect_size_col is in the dataframe columns            
                df = pd.read_csv(args.data_frame, sep='\t', nrows=1)
                if args.effect_size_col not in df.columns:
                    raise ValueError(f"Specified effect_size_col '{args.effect_size_col}' not found in the data frame columns.")
            except Exception as e:
                raise ValueError(f"Error checking effect_size_col: {e}") from e
    else:
        # If using MuData, check that the file exists
        if args.command != 'simulate':
            if not hasattr(args, 'mudata'):
                raise ValueError("Please provide a valid MuData object file path with --mudata")
            try:
                mu_data = mu.read_h5mu(args.mudata)
                # Check that the Mudata object contains results, layer and guide metadata
                if args.mu_uns_key_results not in mu_data.uns_keys():
                    raise ValueError(f"MuData object does not contain the specified uns key '{args.mu_uns_key_results}'. Available keys: {mu_data.uns_keys()}")
                if args.mu_guide_mod not in mu_data.mod:
                    raise ValueError(f"MuData object does not contain the specified guide modality '{args.mu_guide_mod}'. Available modalities: {mu_data.mod}")
                # Check if the guide ID column exists in the MuData object
                if args.mu_guide_id_col not in mu_data[args.mu_guide_mod].var_keys():
                    raise ValueError(f"MuData object does not contain the specified guide ID column '{args.mu_guide_id_col}'. Available columns: {mu_data[args.mu_guide_mod].var_names}")
                # Check if the element ID columns exist in the MuData object
                for col in args.mu_element_id_cols:
                    if col not in mu_data[args.mu_guide_mod].var_keys():
                        raise ValueError(f"MuData object does not contain the specified element ID column '{col}'. Available columns: {mu_data[args.mu_guide_mod].var_names}")
            except Exception as e:
                raise ValueError(f"Error reading MuData object: {e}") from e
    return args

def update_mudata(args, df):
    """
    Update the MuData object with the FRACTEL results.
    """
    mudata = mu.read_h5mu(args.mudata)
    mudata.uns[f'{args.output_col_basename}_results'] = df.to_dict(orient='list')
    mudata.write(filename=f'{args.output_basename}_{args.output_col_basename}.h5mu', compression='gzip')
    logging.info(f"Updated MuData object saved to {args.output_basename}_{args.output_col_basename}.h5mu")

def main():
    """
    Main function to parse arguments and execute the appropriate sub-command.
    """
    parser = argparse.ArgumentParser(
        description='FRACTEL: Framework for Rank Aggregation of CRISPR Tests within ELements',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    pkg_version = version("fractel")
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {pkg_version}",
    )

    subparsers = parser.add_subparsers(dest='command', required=True, help='Sub-command to execute')

    run_parser = subparsers.add_parser(
        'run',
        help='Run FRACTEL test on a given dataframe with p-values of grouped elements'
    )

    
    # Create a mutually exclusive group for data input options
    data_input_group = run_parser.add_mutually_exclusive_group(required=True)
    data_input_group.add_argument('-df', '--data-frame', type=str,
                             help='File path to the data frame with the data to aggregate')
    data_input_group.add_argument('-mu', '--mudata', type=str,
                             help='MuData object file path in IGVF format to use instead of a data frame')
    
    # Group the reference dataframe selection arguments for MuData
    mudata_group = run_parser.add_argument_group('mudata specific options')
    mudata_group.add_argument('--mu-uns-key-results', type=str, default='test_results',
                              help='Uns key in the MuData object, found in the obj.uns_keys(), to use ' \
                              'for the results (default: %(default)s)')
    mudata_group.add_argument('--mu-guide-mod', type=str, default='guide',
                              help='Name of the modality in MuData object with the guide data layer %(default)s)')
    mudata_group.add_argument('--mu-guide-id-col', type=str, default='guide_id',
                              help='Column in the MuData guide metadata (obj[\'guide\'].var) and results ' \
                              'identifying each gRNA (default: %(default)s)')
    mudata_group.add_argument('--mu-element-id-cols', type=str, nargs='+', 
                              default=['intended_target_name', 'intended_target_chr', 'intended_target_start', 
                                       'intended_target_end'], 
                                       help='Columns in the guide metadata data frame that together uniquely ' \
                                       'identify a genomic element (default: %(default)s)')
    mudata_group.add_argument('--save-updated-mu-data', action='store_true',
                                help='If specified, update and save the MuData object with the FRACTEL results. ')
    run_parser.add_argument('--aggregating-cols', required=True, type=str, nargs="+",
                             help='List of columns in data frame to use for the aggregation. The first column is ' \
                             'implicitely considered the genomic element ID column (e.g. dhs)')
    run_parser.add_argument('--usecols', type=str, nargs="+",
                             help='If specified, only these columns will be used from the data frame')
    run_parser.add_argument('--discard-values-in-aggr-cols', type=str, nargs="+",
                             help='If specified, these values will be discarded from the data frame in the aggregating columns')
    run_parser.add_argument('--keyword-for-background-values', type=str,
                             help='If specified, this value will be used to identify background values in the aggregating columns. ' \
                             'For example, if aggregating gRNA p-values in genomic elements, this should be the value used to identify non-targeting controls')
    run_parser.add_argument('--ignore-miscalibration', action='store_true',
                             help='If specified, ignore miscalibration of background p-values and proceed with the FRACTEL test')
    run_parser.add_argument('--sim-data', required=True, type=str, nargs="+",
                             help='File paths to the numpy dictionary of simulated data to use for the test. ' \
                             'The key should be the number of guides, and the values should be the p-values of the simulations')
    run_parser.add_argument('--pval-col', default='pvalue', type=str,
                             help='Column name in the data frame with the p-values to use for the test (default: %(default)s)')
    run_parser.add_argument('--pval-type', choices=('left', 'right', 'two-sided'), type=str, default='two-sided',
                            help='Type of p-value to use: "left" for left-tailed, "right" for right-tailed, or "two-sided" for two-tailed tests (default: %(default)s)')
    run_parser.add_argument('--bnd', type=float,
                             help='Bound for FRACTEL test. If < 1, it is interpreted as a fraction of the number of guides')
    run_parser.add_argument('--bnd-min', type=int, default = 3,
                            help='Minimum number of singletons for the simulation, in case using a variable/proportional bound (default: %(default)s)')
    run_parser.add_argument('-o', '--output-basename', required=True, type=str,
                             help='Path to the output file to save the results (will be saved as a compressed tsv file)')
    run_parser.add_argument('--output-col-basename', type=str, default='FRACTEL',
                             help='Base name for the `*_pval` and `*_pval_fdr_corr` output columns (default: %(default)s)')
    run_parser.add_argument('--fdr-thres', type=float, default=0.5,
                             help='Threshold for FDR correction of FRACTEL p-values (default: %(default)s)')
    run_parser.add_argument('--row-id-col', type=str,
                             help='If specified, this column will be used to uniquely identify each element per group in the data frame. ' \
                             'For example, if aggregating gRNA p-values in genomic elements, this should be the gRNA ID column')
    run_parser.add_argument('--effect-size-col', type=str,
                             help='If specified, use the data frame values under this column to compute effect sizes defined as ' \
                             'a maximum weighted average across observations in an element. ')

    simulate_parser = subparsers.add_parser(
        'simulate',
        help='Simulate data for FRACTEL analysis'
    )
    simulate_parser.add_argument('--num-guides', required=True, type=int, nargs="+",
                                 help='Number of guides to simulate')
    simulate_parser.add_argument('--num-simulations', required=True, type=int,
                                 help='Number of simulations to generate')
    simulate_parser.add_argument('--output-basename', required=True, type=str,
                                 help='Base name for the output files')
    simulate_parser.add_argument('--bnd', type=float, required=True,
                                 help='Bound for FRACTEL test. If < 1, it is interpreted as a fraction of the number of guides')
    simulate_parser.add_argument('--bnd-min', type=int, default = 3,
                                 help='Minimum number of singletons for the simulation, in case using a variable/proportional bound (default: %(default)s)')

    calibrate_parser = subparsers.add_parser(
        'calibrate',
        help='''Calibrate p-values in a data frame based on a background distribution. Currently, the calibration 
        is done by adjusting an empirical cumulative distribution function (ECDF) from the background/reference set 
        and evaluating the observed p-values against it.'''
    )
    calibrate_parser.add_argument('--data-frame', required=True, type=str,
                                    help='File path to the data frame with background p-values (e.g. non-targeting control p-values)')
    calibrate_parser.add_argument('--pval-col', default='pvalue', type=str,
                                    help='Column name in the data frames with the p-values to interpolate (default: %(default)s)')
    calibrate_parser.add_argument('--interpolated-col', default='interpolated_pvalue', type=str,
                                    help='Column name for the interpolated p-values (default: %(default)s)')
    calibrate_parser.add_argument('--output-basename', required=True, type=str,
                                    help='Base name for the output file to save the results (will be saved as a compressed tsv file)')
    # Group the reference dataframe selection arguments
    reference_df_select_group = calibrate_parser.add_argument_group('reference dataframe')
    reference_df_select_group.add_argument('--reference-data-frame', required=True, type=str,
                                           help='File path to the reference data frame with p-values')
    reference_df_select_group.add_argument('--reference-df-select-col', type=str,
                                           help='Column to use for filtering the reference dataframe')
    reference_df_select_group.add_argument('--reference-df-select-value', type=str,
                                           help='Value to use for filtering the reference dataframe')

    args = parser.parse_args()
    args = validate_args(args)

    match args.command:
        case 'run':
            df = run_fractel_analysis(args)
            if args.effect_size_col is not None:
                # Add fractel_df to the final results
                df = df.join(compute_effect_sizes(
                    load_and_filter_data(args),
                    args.aggregating_cols,
                    args.pval_col,
                    args.effect_size_col,
                    args.pval_type,
                    args.output_col_basename
                ), on=args.aggregating_cols)
            save_df_to_tsv(df, args.output_basename, keep_index=True)
            if args.mudata and args.save_updated_mu_data:
                update_mudata(args, df)
        case 'simulate':
            sim_dict = run_simulation(args)
            save_simulation_data(sim_dict, args.output_basename)
        case 'calibrate':
            df = run_calibration(args)
            save_df_to_tsv(df, args.output_basename)


if __name__ == "__main__":
    main()
