from pathlib import Path
from os import scandir
import numpy
import pandas

from shiny import reactive, session
from starlette.routing import Mount
from starlette.staticfiles import StaticFiles

import scipy
import pandas as pd
import numpy as np
import re
import os

from typing import Literal, Any

from config import Config

config = Config()
dist_defaults = config.input_config('distributions')

cont_dist = dist_defaults['continuous']
discrete_dist = dist_defaults['discrete']
divergence_doc_names = config.input_config('divergence')['doc_names']


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


def create_distribution_df(dist_name: str, continuous_dist: bool, dist_size: int, user_options: tuple,
                           conditional: reactive.Value,
                           dist_params: [list | dict],
                           stat_moments: str = 'mvsk',
                           random_state: int = None) -> dict[pandas.DataFrame | numpy.ndarray | dict[float]]:
    """
    Create distribution data frame and array automatically using the scipy.stats package.
    :param random_state: Random seed value used in random distribution value creation
    :param dist_name: Distribution to generate
    :param continuous_dist: Whether it is continuous or not
    :param dist_size: Used in RV generation, the number of RVs to generate
    :param user_options: Methods passed by the user to generate: SF, ISF etc.
    :param conditional: Conditional argument for extra options to generate
    :param dist_params: Distribution parameters: scale, loc, trials etc.
    :param stat_moments: 'Mean, Variance, Skewness, Kurtosis' - mvsk
    :return:
    """
    # TODO make this work with any type of given moments. Only works with 'mvsk' at the moment

    dist_data = {
        'distribution_array': None,
        'distribution_df': None,
        'stats': None
    }
    dist = None
    standard_cols = cont_dist['standard'] if continuous_dist else discrete_dist['standard']

    if isinstance(dist_params, dict):
        dist = getattr(scipy.stats, dist_name)(**dist_params)
    elif isinstance(dist_params, list):
        dist = getattr(scipy.stats, dist_name)(*dist_params)

    dist_rvs = dist.rvs(size=dist_size, random_state=random_state)

    if continuous_dist:
        pdf_pmf = dist.pdf(dist_rvs)
    else:
        pdf_pmf = dist.pmf(dist_rvs)

    cdf = dist.cdf(dist_rvs)

    stats = dist.stats(moments=stat_moments)
    entropy = (dist.entropy(),)
    stats = stats + entropy

    if continuous_dist:
        fit_stats = getattr(scipy.stats, dist_name).fit(dist_rvs)
        stats = stats + fit_stats

    calc_user_option = getattr(dist, user_options[0]().replace(' ', '').lower())(dist_rvs)

    if conditional():
        calc_extra_option = getattr(dist, user_options[1]().replace(' ', '').lower())(cdf)
        dist_array = np.vstack((dist_rvs, pdf_pmf, cdf, calc_user_option, calc_extra_option))
    else:
        dist_array = np.vstack((dist_rvs, pdf_pmf, cdf, calc_user_option))

    dist_df = pandas.DataFrame(dist_array.T)

    if conditional():
        dist_df.columns = [*standard_cols, user_options[0](), user_options[1]()]
    else:
        dist_df.columns = [*standard_cols, user_options[0]()]

    dist_data['distribution_array'] = dist_array
    dist_data['distribution_df'] = dist_df
    dist_data['stats'] = {k: round(v, 4) for k, v in
                          zip(['mean', 'variance', 'skewness', 'kurtosis', 'entropy', 'loc', 'scale'], stats)}

    return dist_data


def compare_dist_types(dist1_name, dist2_name,
                       check_type: Literal['discrete', 'continuous', None] = None) -> tuple[Any, Any] | bool:
    """
    Check if the types of two distributions are the same. Otherwise, return their types
    :param dist1_name: Name of first distribution
    :param dist2_name: Name of second distribution
    :param check_type: Type to check against
    :return:
    """
    if check_type is None:
        return config.get_dist_type(dist1_name), config.get_dist_type(dist2_name)

    if config.get_dist_type(dist1_name) == config.get_dist_type(dist2_name) == check_type:
        return True
    else:
        return False


def gen_starlette_mounts(module):
    mounts = tuple()

    root_path = Path(__file__).parent.joinpath(f'modules/{module}')
    module_subfolders = [f.path for f in scandir(root_path) if f.is_dir() and not f.path.endswith('__pycache__')]
    url_paths = [sub_path.split('/')[-1] for sub_path in module_subfolders]

    for sub_folder, url_path in zip(module_subfolders, url_paths):
        mounts = mounts + (Mount(f'/docs/{url_path}', app=StaticFiles(directory=sub_folder, html=True)),)

    return mounts


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
