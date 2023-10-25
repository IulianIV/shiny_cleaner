from __future__ import annotations
import asyncio
from shiny import Inputs, Outputs, Session, module, render, ui, reactive
from shinywidgets import render_widget
import shiny.experimental as x

import plotly.express as px
import plotly.graph_objs as go

from utils import synchronize_size, two_dim_to_one_dim, create_distribution_df

from config import Config

graph_height = Config.ui_config('graph_height')


@module.server
def create_distribution_inputs(input: Inputs, output: Outputs, session: Session):
    @output
    @render.ui
    @reactive.event(input.distributions)
    def inputs():
        min_val = Config.input_config('distributions_min')
        max_val = Config.input_config('distributions_max')

        distributions_ui_defaults = {
            'numerics': (ui.column(3, ui.input_numeric('min', 'Min', value=min_val)),
                         ui.column(3, ui.input_numeric('max', 'Max', value=max_val))),
            'switch': (x.ui.tooltip(
                ui.input_switch('matrix', 'Matrix'),
                # TODO find a way to show this as `(input.min(), input.max())` with actual values
                "(Min, Max) matrix of values",
                id="matrix_tip", placement='left'
            )),
            'slider': (ui.input_slider('observations', 'Observations', min=min_val, max=max_val,
                                       value=max_val / 2)),
            'plot_btn': (ui.input_action_button('plot_distribution', 'Plot'))
        }

        sd = Config.input_config('distributions_standard_deviation_sigma')
        mean = Config.input_config('distributions_mean_mu')
        events = Config.input_config('distributions_events')
        scale = Config.input_config('distributions_scale')
        prob = Config.input_config('distributions_probability')
        trials = Config.input_config('distributions_trials')

        if input.distributions() == 'Normal':
            return (
                ui.row(
                    distributions_ui_defaults['numerics'],
                    ui.column(3, ui.input_numeric('mean', 'μ', value=mean)),
                    ui.column(3, ui.input_numeric('sd', 'σ', value=sd))
                ), distributions_ui_defaults['switch'],
                distributions_ui_defaults['slider'],
                distributions_ui_defaults['plot_btn']

            )
        if input.distributions() == 'Poisson':
            return (
                ui.row(
                    distributions_ui_defaults['numerics'],
                    ui.column(4, ui.input_numeric('events', 'Events', value=events)),
                ), distributions_ui_defaults['switch'],
                distributions_ui_defaults['slider'],
                distributions_ui_defaults['plot_btn']
            )

        if input.distributions() == 'Exponential':
            return (
                ui.row(distributions_ui_defaults['numerics'],
                       ui.column(4, ui.input_numeric('scale', 'Scale', value=scale)),
                       ), distributions_ui_defaults['switch'],
                distributions_ui_defaults['slider'],
                distributions_ui_defaults['plot_btn']
            )

        if input.distributions() == 'Geometric':
            return (
                ui.row(
                    distributions_ui_defaults['numerics'],
                    ui.column(4, ui.input_numeric('prob', 'Probability', value=prob)),
                ), distributions_ui_defaults['switch'],
                distributions_ui_defaults['slider'],
                distributions_ui_defaults['plot_btn']
            )
        if input.distributions() == 'Binomial':
            return (
                ui.row(distributions_ui_defaults['numerics'],
                       ui.column(4, ui.input_numeric('trials', 'Trials', value=trials)),
                       ui.column(4, ui.input_numeric('prob', 'Probability', value=prob)),
                       ), distributions_ui_defaults['switch'],
                distributions_ui_defaults['slider'],
                distributions_ui_defaults['plot_btn']
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


# TODO check #1 issue in GitHub. Apparently, having the `input.prob()` check in place, and inside the function, breaks
#   functionality and the function fails is called on a distribution which does NOT generate the UI with that input.
# TODO Also, even if the UI is generated, providing values >1 or <0 in the input, even with the functionality below to
#   update it to permitted values, it first runs the DataFrame generation with the actual wrong values.
#   This raises a ValueError
@module.server
def update_distribution_inputs(input: Inputs, output: Outputs, session: Session):
    @reactive.Effect
    @reactive.event(input.min, input.max, input.prob)
    # all probability checks removed until #1 is fixed
    # @reactive.event(input.min, input.max, input.prob)
    def update():
        current_value = input.observations()
        c_min_val = input.min()
        c_max_val = input.max()
        # prob_val = input.prob()

        if c_min_val > c_max_val:
            ui.update_numeric('min', value=c_max_val)
            ui.update_numeric('max', value=c_min_val)
            ui.update_slider('observations', min=c_max_val, max=c_min_val, value=current_value)

        # if prob_val > 1 or 0 > prob_val:
        #     ui.update_numeric('prob', value=1)

        ui.update_slider('observations', min=c_min_val, max=c_max_val, value=current_value)


@module.server
def create_distribution_data_set(input: Inputs, output: Outputs, session: Session, data_frame: reactive.Value):
    @reactive.Effect
    def data_set():
        obs = input.observations()

        if input.distributions() == 'Normal':
            sd = input.sd()
            mean = input.mean()

            distribution_df = create_distribution_df(input.distributions().lower(),
                                                     {'mean': mean, 'sd': sd, 'obs': obs, 'min': input.min,
                                                      'max': input.max},
                                                     input.matrix)

            data_frame.set(distribution_df)

        if input.distributions() == 'Poisson':
            events = input.events()

            distribution_df = create_distribution_df(input.distributions().lower(),
                                                     {'events': events, 'obs': obs, 'min': input.min, 'max': input.max},
                                                     input.matrix)

            data_frame.set(distribution_df)

        if input.distributions() == 'Exponential':
            scale = input.scale()

            distribution_df = create_distribution_df(input.distributions().lower(),
                                                     {'scale': scale, 'obs': obs, 'min': input.min,
                                                      'max': input.max},
                                                     input.matrix)

            data_frame.set(distribution_df)

        if input.distributions() == 'Geometric':
            prob = input.prob()

            distribution_df = create_distribution_df(input.distributions().lower(),
                                                     {'prob': prob, 'obs': obs, 'min': input.min,
                                                      'max': input.max},
                                                     input.matrix)

            data_frame.set(distribution_df)

        if input.distributions() == 'Binomial':
            prob = input.prob()
            trials = input.trials()

            distribution_df = create_distribution_df(input.distributions().lower(),
                                                     {'trials': trials, 'prob': prob, 'obs': obs, 'min': input.min,
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
