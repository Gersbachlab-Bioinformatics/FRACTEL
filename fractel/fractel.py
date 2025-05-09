#!/usr/bin/env python
# coding: utf-8
"""
FRACTEL: Framework for Rank Aggregation of CRISPR Tests within ELements
"""

import argparse
from functools import reduce

import numpy as np
import pandas as pd
from scipy.stats import beta
from statsmodels.stats.multitest import fdrcorrection
from functools import reduce

def element_test(dhs_df, sim_data, pval_col='pvalue', bnd=None,
                 row_id_col='grna', return_guide=False, return_index_min_grna=False):
    """
    Perform the FRACTEL test on a given dataframe of grouped data.
    """
    num_guides = dhs_df.shape[0]

    if num_guides not in sim_data:
        raise ValueError(f"Number of guides {num_guides} not found in simulated data.\
                          Please run the simulation with this number of guides.")
    aux = dhs_df.sort_values(by=pval_col)
    pvals = aux[pval_col]
    aux = np.arange(1, num_guides + 1)
    beta_evals = beta.cdf(pvals, aux, num_guides - aux + 1)
    if bnd < 1:
        bnd = max(1, int(np.round(num_guides * bnd)))
    pmin = np.min(beta_evals[:bnd])

    if return_guide or return_index_min_grna:
        grnas = dhs_df[row_id_col]
        if return_index_min_grna:
            return np.argmin(beta_evals)
        grna = grnas.values[np.argmin(beta_evals[:bnd])]
        return grna
    return (sim_data[num_guides] < pmin).sum() / sim_data[num_guides].size


def run_simulation(args):
    """
    Simulate data for FRACTEL analysis.
    """
    sim_dict = {}
    for m in [int(m) for m in args.num_guides]:
        if args.bnd < 1:
            bnd = np.max(int(round(args.bnd*m)),min(m,args.bnd_min))
        else:
            bnd = min(args.bnd, m)
        pvals = np.sort(np.random.random((args.num_simulations, m)))
        aux = np.arange(1, m + 1)
        beta_evals = beta.cdf(pvals, aux, m - aux + 1)
        sim_dict[m] = np.min(beta_evals[:,:bnd], axis=1)
    np.savez(f'{args.output_basename}', simulations = sim_dict)
    print(f"Simulated dictionary data saved to {args.output_basename}.npz under the key 'simulations'.")

def run_fractel_analysis(args):
    """
    Run the FRACTEL analysis based on the provided arguments.
    """
    
    # Load the simulated data
    sim_data = list(map(lambda x: np.load(x, allow_pickle = True)['simulations'].item(), args.sim_data))
    if len(sim_data) > 1:
        reduce(lambda x, y: x.update(y), sim_data)
    sim_data = sim_data[0]
    
    # Load the data frame
    df = pd.read_csv(
        args.data_frame,
        sep='\t',
        usecols=args.usecols
    )

    # Discard rows with NaN values in the aggregating columns
    df = df[~df[args.aggregating_cols[0]].isna()]

    # If specified, discard rows with certain values in the aggregating columns
    if args.discard_values_in_aggr_cols:
        discard_mask = reduce(
            lambda x, y: x | y,
            [df[col].isin(args.discard_values_in_aggr_cols) for col in args.aggregating_cols]
        )
        df = df[~discard_mask]

    # Group the data frame by the specified columns and apply the FRACTEL test
    df_tmp = (
        df.groupby(args.aggregating_cols)
        [df.columns]
        .apply(lambda x: element_test(
            x,
            sim_data,
            pval_col=args.pval_col,
            bnd=args.bnd,
            row_id_col=args.row_id_col,
            return_guide=False
        ))
        .to_frame()
    )
    df_tmp.columns = [args.output_col_basename]  # rename the column FRACTEL p-values

    # Apply FDR correction to FRACTEL p-values
    _, fdr_pvals_tmp = fdrcorrection(
        df_tmp[args.output_col_basename], alpha=args.fdr_thres, method='indep', is_sorted=False
    )
    df_tmp[f'{args.output_col_basename}_fdr_corr'] = fdr_pvals_tmp

    # Save the results to a compressed TSV file
    df_tmp.to_csv(f'{args.output_tsv_basename}.tsv.gz', sep='\t')
    print(f"Results saved to {args.output_tsv_basename}.tsv.gz")


def main():
    """
    Main function to parse arguments and execute the appropriate sub-command.
    """
    parser = argparse.ArgumentParser(
        description='Run FRACTEL test or simulate data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', required=True, help='Sub-command to execute')

    run_parser = subparsers.add_parser(
        'run',
        help='Run FRACTEL test on a given dataframe with p-values of grouped elements'
    )
    run_parser.add_argument('-df', '--data-frame', required=True, type=str,
                             help='File path to the data frame with the data to aggregate')
    run_parser.add_argument('--aggregating-cols', required=True, type=str, nargs="+",
                             help='List of columns in data frame to use for the aggregation.')
    run_parser.add_argument('--usecols', type=str, nargs="+",
                             help='If specified, only these columns will be used from the data frame')
    run_parser.add_argument('--discard-values-in-aggr-cols', type=str, nargs="+",
                             help='If specified, these values will be discarded from the data frame in the aggregating columns')
    run_parser.add_argument('--sim-data', required=True, type=str, nargs="+",
                             help='File paths to the numpy dictionary of simulated data to use for the test. ' \
                             'The key should be the number of guides, and the values should be the p-values of the simulations')
    run_parser.add_argument('--pval-col', default='pvalue', type=str,
                             help='Column name in the data frame with the p-values to use for the test (default: %(default)s)')
    run_parser.add_argument('--bnd', type=float,
                             help='Bound for FRACTEL test. If < 1, it is interpreted as a fraction of the number of guides')
    run_parser.add_argument('-o', '--output-tsv-basename', required=True, type=str,
                             help='Path to the output file to save the results (will be saved as a compressed tsv file)')
    run_parser.add_argument('--output-col-basename', type=str, default='FRACTEL_pval',
                             help='Base name for the output columns (default: %(default)s)')
    run_parser.add_argument('--fdr-thres', type=float, default=0.5,
                             help='Threshold for FDR correction of FRACTEL p-values (default: %(default)s)')
    run_parser.add_argument('--row-id-col', type=str,
                             help='If specified, this column will be used to uniquely identify each element per group in the data frame. ' \
                             'For example, if aggregating gRNA p-values in genomic elements, this should be the gRNA ID column')

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

    args = parser.parse_args()
    if args.bnd and args.bnd >= 1:
        args.bnd = int(args.bnd)

    if args.command == 'run':
        run_fractel_analysis(args)
    elif args.command == 'simulate':
        run_simulation(args)


if __name__ == "__main__":
    main()
