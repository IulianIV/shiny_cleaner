import numpy
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from shiny import Inputs, Outputs, Session, module, render, ui, reactive
from shinywidgets import render_widget

import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff

from utils import synchronize_size

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
        sd = Config.input_config('distributions_standard_deviation_sigma')
        mean = Config.input_config('distributions_mean_mu')

        if input.distributions() == 'Gaussian':
            return (
                ui.row(
                    ui.column(3, ui.input_numeric('min', 'Min', value=min_val)),
                    ui.column(3, ui.input_numeric('max', 'Max', value=max_val)),
                    ui.column(3, ui.input_numeric('mean', 'μ', value=mean)),
                    ui.column(3, ui.input_numeric('sd', 'σ', value=sd))
                ), ui.input_switch('matrix', 'Min x Max matrix'),
                ui.input_slider('observations', 'Observations', min=min_val, max=max_val,
                                value=max_val / 2),
                ui.input_action_button('plot_distribution', 'Plot')
            )


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
        sd = input.sd()
        mean = input.mean()

        if input.distributions() == 'Gaussian':
            distribution = np.random.normal(mean, sd, obs)
            distribution_df = pd.DataFrame(data=distribution, index=range(1, len(distribution) + 1),
                                           columns=['value'])

            if input.matrix():
                distribution = np.random.normal(mean, sd, size=(input.min(), input.max()))
                distribution_df = pd.DataFrame(data=distribution[:, :], index=range(1, len(distribution) + 1),
                                               columns=[f'value_{x}' for x in range(1, distribution.shape[1] + 1)])

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
        # widget = go.FigureWidget()

        if input.distributions() == 'Gaussian':
            plot_data = data_frame()

            if input.matrix():
                one_dim_df = plot_data.stack().reset_index()
                one_dim_df.drop(one_dim_df.columns[:-1], axis=1, inplace=True)
                one_dim_df.columns = ['value']
                plot_data = one_dim_df

            # Create the plot
            hist = px.histogram(
                plot_data,
                x='value',
                title=f'Histogram of {input.distributions()} distribution', height=graph_height)

            # TODO Add Kernel Density Estimation KDE or some sort of Probability Density Function
            #   like the one from: https://numpy.org/doc/stable/reference/random/generated/numpy.random.normal.html
            # array_plot_data = plot_data[plot_data.columns[0]].to_numpy()
            # kde = gaussian_kde(array_plot_data)
            #
            # estimate1 = numpy.array([1 / (array_plot_data.std() * np.sqrt(2 * np.pi)) * np.exp(- (x -
            # array_plot_data.mean()) ** 2 / (2 * array_plot_data.std() ** 2)) for x in array_plot_data]) print(
            # plot_data) print(kde(array_plot_data)) kde_trace = go.Scatter(x=plot_data, y=kde(array_plot_data),
            # mode='lines', name='markers')
            # hist.add_trace(kde_trace)

            widget = go.FigureWidget(hist)

            @synchronize_size("graph")
            def on_size_changed(width, height):
                widget.layout.width = width
                widget.layout.height = height

            return widget
