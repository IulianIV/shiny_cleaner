import pandas
import re
import os


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


def create_summary_df(data_frame: pandas.DataFrame, group_by: str, aggregators: tuple[str] | list,
                      functions: list[str] | str, fallback_functions: list[str] | str = None) -> pandas.DataFrame:
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

    aggs = {k: list(functions) if pandas.api.types.is_numeric_dtype(df[k]) else fallback_functions for k in aggregators}

    summarized_df = (df.groupby(group_by).agg(
        aggs
    ).reset_index()
                     )
    summarized_df.columns = [re.sub('^_|_$', '', '_'.join(col)) for col in summarized_df.columns.values]

    return summarized_df
