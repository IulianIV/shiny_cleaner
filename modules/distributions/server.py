from __future__ import annotations
import numpy as np
import pandas
import pandas as pd
import shiny.reactive

from shiny import Inputs, Outputs, Session, module, render, ui, reactive
from shinywidgets import render_widget
import shiny.experimental as x

import plotly.express as px
import plotly.graph_objs as go

from utils import synchronize_size, two_dim_to_one_dim, create_distribution_df

from config import Config

graph_height = Config.ui_config('graph_height')
distribution_types = Config.server_config('distribution_types')


@module.server
def create_distribution_inputs(input: Inputs, output: Outputs, session: Session):
    @output
    @render.ui
    @reactive.event(input.distributions)
    def inputs():
        min_val = Config.input_config('distributions_min')
        max_val = Config.input_config('distributions_max')
        sd = Config.input_config('distributions_standard_deviation_sigma')
        mean = Config.input_config('distributions_mean_mu')
        events = Config.input_config('distributions_events')
        scale = Config.input_config('distributions_scale')

        if input.distributions() == 'Gaussian':
            return (
                ui.row(
                    ui.column(3, ui.input_numeric('min', 'Min', value=min_val)),
                    ui.column(3, ui.input_numeric('max', 'Max', value=max_val)),
                    ui.column(3, ui.input_numeric('mean', 'μ', value=mean)),
                    ui.column(3, ui.input_numeric('sd', 'σ', value=sd))
                ), x.ui.tooltip(
                    ui.input_switch('matrix', 'Matrix'),
                    # TODO find a way to show this as `(input.min(), input.max())` with actual values
                    "(Min, Max) matrix of values",
                    id="matrix_tip", placement='left'
                ),
                ui.input_slider('observations', 'Observations', min=min_val, max=max_val,
                                value=max_val / 2),
                ui.input_action_button('plot_distribution', 'Plot')

            )
        if input.distributions() == 'Poisson':
            return (
                ui.row(
                    ui.column(3, ui.input_numeric('min', 'Min', value=min_val)),
                    ui.column(3, ui.input_numeric('max', 'Max', value=max_val)),
                    ui.column(3, ui.input_numeric('events', 'Events', value=events)),
                ), x.ui.tooltip(
                    ui.input_switch('matrix', 'Matrix'),
                    # TODO find a way to show this as `(input.min(), input.max())` with actual values
                    "(Min, Max) matrix of values",
                    id="matrix_tip", placement='left'
                ),
                ui.input_slider('observations', 'Observations', min=min_val, max=max_val,
                                value=max_val / 2),
                ui.input_action_button('plot_distribution', 'Plot')
            )

        if input.distributions() == 'Exponential':
            return (
                ui.row(
                    ui.column(3, ui.input_numeric('min', 'Min', value=min_val)),
                    ui.column(3, ui.input_numeric('max', 'Max', value=max_val)),
                    ui.column(3, ui.input_numeric('scale', 'Scale', value=scale)),
                ), x.ui.tooltip(
                    ui.input_switch('matrix', 'Matrix'),
                    # TODO find a way to show this as `(input.min(), input.max())` with actual values
                    "(Min, Max) matrix of values",
                    id="matrix_tip", placement='left'
                ),
                ui.input_slider('observations', 'Observations', min=min_val, max=max_val,
                                value=max_val / 2),
                ui.input_action_button('plot_distribution', 'Plot')
            )


# TODO add a way to show data about the distribution: set mean and sd, calculated mean and sd etc
"""
ui.output_text_verbatim("test", placeholder=False)
@module.server
def test_text(input: Inputs, output: Outputs, session: Session):
    @output
    @render.text
    def test():
        return input.distributions()
    return test

"""


@module.server
def update_distribution_inputs(input: Inputs, output: Outputs, session: Session):
    @reactive.Effect
    @reactive.event(input.min, input.max)
    def update():
        current_value = input.observations()
        min_val = input.min()
        max_val = input.max()

        if min_val > max_val:
            ui.update_numeric('min', value=max_val)
            ui.update_numeric('max', value=min_val)
            ui.update_slider('observations', min=max_val, max=min_val, value=current_value)

        ui.update_slider('observations', min=min_val, max=max_val, value=current_value)


@module.server
def create_distribution_data_set(input: Inputs, output: Outputs, session: Session, data_frame: reactive.Value):
    @reactive.Effect
    def data_set():
        obs = input.observations()

        if input.distributions() == 'Gaussian':
            sd = input.sd()
            mean = input.mean()

            distribution_df = create_distribution_df('normal',
                                                     {'mean': mean, 'sd': sd, 'obs': obs, 'min': input.min,
                                                      'max': input.max},
                                                     input.matrix)

            data_frame.set(distribution_df)

        if input.distributions() == 'Poisson':
            events = input.events()

            distribution_df = create_distribution_df('poisson',
                                                     {'events': events, 'obs': obs, 'min': input.min, 'max': input.max},
                                                     input.matrix)

            data_frame.set(distribution_df)

        if input.distributions() == 'Exponential':
            scale = input.scale()

            distribution_df = create_distribution_df('exponential',
                                                     {'scale': scale, 'obs': obs, 'min': input.min,
                                                      'max': input.max},
                                                     input.matrix)

            data_frame.set(distribution_df)


@module.server
def load_distribution_data(input: Inputs, output: Outputs, session: Session, data_frame: reactive.Value):
    @output
    @render.data_frame
    def data():
        return render.DataGrid(
            data_frame().round(3),
            row_selection_mode='multiple',
            width='100%',
            height='100%',
        )


@module.server
def distribution_graph(input: Inputs, output: Outputs, session: Session, data_frame):
    @output
    @render_widget
    @reactive.event(input.plot_distribution)
    def graph():
        plot_data = data_frame()
        widget = None

        if input.matrix():
            plot_data = two_dim_to_one_dim(plot_data, 'value')

        # Create the plot
        hist = px.histogram(
            plot_data,
            x='value',
            title=f'Histogram of {input.distributions()} distribution', height=graph_height)

        # TODO Add Kernel Density Estimation KDE or some sort of Probability Density Function
        #   like the one from: https://numpy.org/doc/stable/reference/random/generated/numpy.random.normal.html
        """
        from scipy.stats import gaussian_kde
        import numpy as np
        array_plot_data = plot_data[plot_data.columns[0]].to_numpy()
        kde = gaussian_kde(array_plot_data)

        estimate1 = numpy.array([1 / (array_plot_data.std() * np.sqrt(2 * np.pi)) * np.exp(- (x -
        array_plot_data.mean()) ** 2 / (2 * array_plot_data.std() ** 2)) for x in array_plot_data]) print(
        plot_data) print(kde(array_plot_data)) kde_trace = go.Scatter(x=plot_data, y=kde(array_plot_data),
        mode='lines', name='markers')
        hist.add_trace(kde_trace)
        """

        widget = go.FigureWidget(hist)

        @synchronize_size("graph")
        def on_size_changed(width, height):
            widget.layout.width = width
            widget.layout.height = height

        return widget
