from __future__ import annotations

from shiny import Inputs, Outputs, Session, module, render, ui, reactive
from shinywidgets import render_widget

import plotly.express as px
import plotly.graph_objs as go

from utils import synchronize_size, create_distribution_df

from config import Config

config = Config()
graph_height = config.ui_config('graph_height')
dist_defaults = config.input_config('distributions')

cont_dist = dist_defaults['continuous']
discrete_dist = dist_defaults['discrete']
dist_names = cont_dist['names'] + discrete_dist['names']


@module.server
def create_dist_inputs(input: Inputs, output: Outputs, session: Session):
    @output
    @render.ui
    @reactive.event(input.distributions)
    def inputs():
        min_val = dist_defaults['min_obs']
        max_val = dist_defaults['max_obs']
        sd = dist_defaults['sd']
        mean = dist_defaults['mean']
        events = dist_defaults['events']
        scale = dist_defaults['scale']
        prob = dist_defaults['probability']
        trials = dist_defaults['trials']
        low = dist_defaults['low']
        high = dist_defaults['high']

        distribution_ui_body = None

        if input.distributions() == 'Normal':
            distribution_ui_body = (ui.column(3, ui.input_numeric('mean', 'μ', value=mean)),
                                    ui.column(3, ui.input_numeric('sd', 'σ', value=sd)))

        elif input.distributions() == 'Poisson':
            distribution_ui_body = ui.column(4, ui.input_numeric('events', 'Events', value=events))

        elif input.distributions() == 'Exponential':
            distribution_ui_body = ui.column(4, ui.input_numeric('scale', 'Scale', value=scale))

        elif input.distributions() == 'Geometric':
            distribution_ui_body = ui.column(4, ui.input_numeric('prob', 'Probability', value=prob))

        elif input.distributions() == 'Binomial':
            distribution_ui_body = (ui.column(3, ui.input_numeric('trials', 'Trials', value=trials)),
                                    ui.column(3, ui.input_numeric('prob', 'Probability', value=prob)))

        elif input.distributions() == 'Uniform':
            distribution_ui_body = (ui.column(3, ui.input_numeric('low', 'Low', value=low)),
                                    ui.column(3, ui.input_numeric('high', 'High', value=high)))

        return (
            ui.row(ui.column(3, ui.input_numeric('min', 'Min', value=min_val)),
                   ui.column(3, ui.input_numeric('max', 'Max', value=max_val)),
                   distribution_ui_body,
                   ),
            ui.input_selectize('prop', 'Properties', discrete_dist['methods'], multiple=False),
            ui.row(ui.column(6, ui.input_checkbox('enbl_extra', 'Extra Properties'))),
            ui.panel_conditional('input.enbl_extra',
                                 ui.input_selectize('extra_prop', 'Extra Properties', discrete_dist['extra_methods'],
                                                    multiple=False)),
            ui.input_slider('observations', 'Observations', min=min_val, max=max_val,
                            value=max_val / 2),
            ui.input_action_button('plot_distribution', 'Plot'),
        )


@module.server
def create_dist_details(input: Inputs, output: Outputs, session: Session, data_frame):
    @output
    @render.text
    def details():
        stats_body = '\n'.join([f'{k}: {v}' for k, v in data_frame.items()])

        return (
            f'\t~~~{input.distributions()} Distribution details~~~\n'
            f'{stats_body}'
        )


@module.server
def update_dist_min_max(input: Inputs, output: Outputs, session: Session):
    @reactive.Effect
    @reactive.event(input.min, input.max)
    def update():
        current_value = input.observations()
        c_min_val = input.min()
        c_max_val = input.max()

        if c_min_val > c_max_val:
            ui.update_numeric('min', value=c_max_val)
            ui.update_numeric('max', value=c_min_val)
            ui.update_slider('observations', min=c_max_val, max=c_min_val, value=current_value)

        ui.update_slider('observations', min=c_min_val, max=c_max_val, value=current_value)


@module.server
def update_dist_prop_select(input: Inputs, output: Outputs, session: Session):
    @reactive.Effect
    def update():
        props = config.get_dist_methods(input.distributions())
        extra_props = config.get_dist_methods(input.distributions(), extra=True)

        ui.update_selectize('prop', choices=props, selected=None)
        ui.update_selectize('extra_prop', choices=extra_props, selected=None)


@module.server
def update_dist_prob(input: Inputs, output: Outputs, session: Session):
    @reactive.Effect
    @reactive.event(input.prob)
    def update():
        prob_value = input.prob()

        if prob_value > 1:
            ui.update_numeric('prob', value=1)
        elif prob_value <= 0:
            ui.update_numeric('prob', value=0.1)


@module.server
def create_dist_df(input: Inputs, output: Outputs, session: Session, data_frame: reactive.Value):
    @output
    @render.data_frame
    @reactive.Calc
    def data():
        obs = input.observations()
        dist_data = dict()

        # TODO Features of the distribution to implement, tabular and graphical, will be shown in each
        #   distributions if-conditional
        #   The graphical representation should be made by drop-down choice, where possible.
        #   Add a Comprehensive view of Distribution Details in the side-bar
        # TODO Discrete, Continuous feature list
        # TODO Research about the usage of loc, scale arguments in the distribution methods.
        #   In some cases it seems to be the only way to actually increase the range of generated
        #   distribution
        """
        ==== Discrete ====
        To add functionalities:
        * interval;
        * entropy;

        More complex functionality to implement:
        * expect
        
        ==== Continuous ====
        To add functionalities:
        * interval;
        * moment;
        * entropy;

        More complex functionality to implement:
        * expect
        """

        if input.distributions() == 'Normal':
            sd = input.sd()
            mean = input.mean()

            norm_dist = create_distribution_df('norm', True, obs,
                                               (input.prop, input.extra_prop),
                                               input.enbl_extra, {'loc': mean, 'scale': sd})

            dist_data = norm_dist

        if input.distributions() == 'Poisson':
            events = input.events()

            poisson_dist = create_distribution_df('poisson', False, obs,
                                                  (input.prop, input.extra_prop),
                                                  input.enbl_extra, [events])

            dist_data = poisson_dist

        if input.distributions() == 'Exponential':
            scale = input.scale()

            expon_dist = create_distribution_df('expon', True, obs,
                                                (input.prop, input.extra_prop),
                                                input.enbl_extra, {'scale': scale})

            dist_data = expon_dist

        if input.distributions() == 'Geometric':
            prob = input.prob()

            geom_dist = create_distribution_df('geom', False, obs,
                                               (input.prop, input.extra_prop),
                                               input.enbl_extra, [prob])

            dist_data = geom_dist

        if input.distributions() == 'Binomial':
            prob = input.prob()
            trials = input.trials()

            binom_dist = create_distribution_df('binom', False, obs,
                                                (input.prop, input.extra_prop),
                                                input.enbl_extra, [trials, prob])

            dist_data = binom_dist

        if input.distributions() == 'Uniform':
            low = input.low()
            high = input.high()

            uniform_dist = create_distribution_df('uniform', True, obs,
                                                (input.prop, input.extra_prop),
                                                input.enbl_extra, {'loc': low, 'scale': high})

            dist_data = uniform_dist

        data_frame.set(dist_data)

        return render.DataGrid(
            dist_data['distribution_df'].round(3),
            row_selection_mode='multiple',
            width='100%',
            height='100%',
        )

    return data


@module.server
def dist_graph(input: Inputs, output: Outputs, session: Session, data_frame: reactive.Value):
    @output
    @render_widget
    @reactive.event(input.plot_distribution)
    def graph():
        plot_data = data_frame()
        widget = None

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
