from shiny import reactive, session
import shiny.reactive

import pandas as pd
import numpy as np
import re
import os

from config import Config

distribution_types = Config.server_config('distribution_types')


# TODO try to implement the distributions as generators

def get_data_files(data_path: str = None) -> list[tuple[str, str]]:
    """
    Return all file names given a path to a folder with data files
    :param data_path: list of PathLike file names
    :return:
    """
    data_dir = ''

    if data_path is None:
        data_dir = os.path.join(os.path.dirname(__file__), 'data')

    file_names = [(file[:-4], os.path.join(data_dir, file)) for file in os.listdir(data_dir) if
                  os.path.isfile(os.path.join(data_dir, file))]

    return file_names


def create_summary_df(data_frame: pd.DataFrame, group_by: str, aggregators: tuple[str] | list,
                      functions: list[str] | str, fallback_functions: list[str] | str = None) -> pd.DataFrame:
    """
    Create a summary DataFrame by grouping based on `group_by`, aggregating by columns from `aggregator` and
    applying a function on all found columns with `actions`
    By default functions is: ['min', 'max', 'mean']
    :param fallback_functions: if any columns from aggregators are not numeric, do the fallback function 'count' instead
    :param data_frame: DataFrame to summarize
    :param group_by: Column to group by
    :param aggregators: Columns to aggregate by
    :param functions: possible functions to apply e.g. [np.sum, 'mean']
    :return:
    """
    # Revert to default values if empty or not provided
    if functions is None or not functions:
        functions = ['min', 'max', 'mean']

    if fallback_functions is None or not fallback_functions:
        fallback_functions = ['count']

    df = data_frame

    aggs = {k: list(functions) if pd.api.types.is_numeric_dtype(df[k]) else fallback_functions for k in aggregators}

    summarized_df = (df.groupby(group_by).agg(
        aggs
    ).reset_index()
                     )
    summarized_df.columns = [re.sub('^_|_$', '', '_'.join(col)) for col in summarized_df.columns.values]

    return summarized_df


def create_distribution_df(dist_type: str, dist_args: dict[str | int | shiny.reactive.Value],
                           cond_input: shiny.reactive.Value, column: str = 'value'):
    """
    Create a distribution function given a distribution type and reactive inputs
    :param dist_type: Must be a distribution type found within `numpy.random`
    :param dist_args: Arguments to be given. Consult `numpy.random.distribution` for more info
    :param cond_input: Conditional input to convert the 2D DataFrame to 1D
    :param column: name to give to columns
    :return: a DataFrame created from a numpy random distribution array
    """

    if dist_type not in distribution_types:
        print(dist_type)
        print(distribution_types)
        raise ValueError(f'Selected distribution "{dist_type}" is not a valid distribution in numpy.random .')

    # basically calls `numpy.random.dist_type` and unpacks the contents of `dist_args` as arguments
    np_dist = getattr(np.random, dist_type)(*[val for key, val in dist_args.items() if key not in ['min', 'max']])
    dist_df = pd.DataFrame(data=np_dist, index=range(1, len(np_dist) + 1),
                           columns=[column])

    if cond_input():
        np_dist = getattr(np.random, dist_type)(
            *[val for key, val in dist_args.items() if key not in ['min', 'max', 'obs']],
            (dist_args['min'](), dist_args['max']()))
        dist_df = pd.DataFrame(data=np_dist[:, :], index=range(1, len(np_dist) + 1),
                               columns=[f'{column}_{x}' for x in range(1, np_dist.shape[1] + 1)])

    return dist_df


def two_dim_to_one_dim(data_frame: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Converts a 2D matrix of n-columns to a single 1 column 1D vector.
    :param data_frame: DataFrame to be reduced
    :param column_name: Column name to pass to the final DataFrame
    :return: reduced DataFrame
    """
    # main functions that convert from n-columns to 1-column
    one_dim_df = data_frame.stack().reset_index()
    # drop all columns except last column - which holds the data
    one_dim_df.drop(one_dim_df.columns[:-1], axis=1, inplace=True)
    # rename last column
    one_dim_df.columns = [column_name]

    return one_dim_df


# This is a hacky workaround to help Plotly plots automatically
# resize to fit their container. In the future we'll have a
# built-in solution for this.
def synchronize_size(output_id):
    def wrapper(func):
        input = session.get_current_session().input

        @reactive.Effect
        def size_updater():
            func(
                input[f".clientdata_output_{output_id}_width"](),
                input[f".clientdata_output_{output_id}_height"](),
            )

        # When the output that we're synchronizing to is invalidated,
        # clean up the size_updater Effect.
        reactive.get_current_context().on_invalidate(size_updater.destroy)

        return size_updater

    return wrapper
