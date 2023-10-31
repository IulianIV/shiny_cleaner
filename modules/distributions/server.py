from __future__ import annotations

from shiny import Inputs, Outputs, Session, module, render, ui, reactive

from shinywidgets import render_widget

import plotly.graph_objs as go
from plotly.subplots import make_subplots

from utils import synchronize_size, create_distribution_df
from .ui import label_with_tooltip

from config import Config

config = Config()
graph_height = config.ui_config('graph_height')
dist_defaults = config.input_config('distributions')

cont_dist = dist_defaults['continuous']
discrete_dist = dist_defaults['discrete']
dist_names = cont_dist['names'] + discrete_dist['names']


@module.server
def create_dist_settings(input: Inputs, output: Outputs, session: Session):
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

        properties_tooltip_text = 'Extra properties to Table & Plot'
        seed_tooltip = 'By setting a seed the data set can be frozen'

        dist_inputs = None

        if input.distributions() == 'Normal':
            dist_inputs = (ui.column(3, ui.input_numeric('mean', 'μ', value=mean)),
                           ui.column(3, ui.input_numeric('sd', 'σ', value=sd)))

        elif input.distributions() == 'Poisson':
            dist_inputs = ui.column(4, ui.input_numeric('events', 'Events', value=events))

        elif input.distributions() == 'Exponential':
            dist_inputs = ui.column(4, ui.input_numeric('scale', 'Scale', value=scale))

        elif input.distributions() == 'Geometric':
            dist_inputs = ui.column(4, ui.input_numeric('prob', 'Probability', value=prob))

        elif input.distributions() == 'Binomial':
            dist_inputs = (ui.column(3, ui.input_numeric('trials', 'Trials', value=trials)),
                           ui.column(3, ui.input_numeric('prob', 'Probability', value=prob)))

        elif input.distributions() == 'Uniform':
            dist_inputs = (ui.column(3, ui.input_numeric('low', 'Low', value=low)),
                           ui.column(3, ui.input_numeric('high', 'High', value=high)))

        dist_head = (ui.column(3, ui.input_numeric('seed', label_with_tooltip('Seed ', True, seed_tooltip, 'right',
                                                                              'seed_tooltip'), value=0)),
                     ui.column(3, ui.input_numeric('max', 'Max Obs.', value=max_val)))

        dist_options = (ui.input_selectize('prop',
                                           label_with_tooltip('Properties ', True, properties_tooltip_text, 'right',
                                                              'card_tooltip'),
                                           discrete_dist['methods'], multiple=False),
                        ui.row(ui.column(6, ui.input_checkbox('enbl_extra', 'Extra Properties'))),
                        ui.panel_conditional('input.enbl_extra',
                                             ui.input_selectize('extra_prop', 'Extra Properties',
                                                                discrete_dist['extra_methods'],
                                                                multiple=False)),
                        ui.input_slider('observations', 'Observations', min=min_val, max=max_val,
                                        value=max_val / 2))

        dist_plot = (ui.row(ui.column(5, ui.input_action_button('plot_distribution', 'Plot Histogram')),
                            ui.column(7, ui.input_checkbox('enbl_plot', 'Other Plots'))),
                     ui.panel_conditional('input.enbl_plot', ui.input_checkbox_group(
                         "plot_props",
                         "Other plots:", [],
                     ), ui.input_action_button('plot_other', 'Plot Other')))

        return (
            ui.row(dist_head,
                   dist_inputs,
                   ),
            dist_options,
            dist_plot
        )

# TODO fix this MathJax. It works only once. On change of distribution the TypeSet is no
#   longer evaluated
# @module.server
# def dist_eq(input: Inputs, output: Outputs, session: Session):
#     @output
#     @render.ui
#     @reactive.event(input.distributions)
#     def eq():
#         equation = ''
#
#         if input.distributions() == 'Binomial':
#             equation = '''
#             $$\\text{Binomial Distribution Formula}$$
#             $$P_x = {n \choose x}p^xq^{n-x}$$
#             $$\\text{Values}$$
#             $$\\begin{cases}n \\to \\text{Observations}\\\\x\\to\\text{Trials}\\\\p\\to\\text{Probability}\\end{cases}$$
#             '''
#
#         if input.distributions() == 'Exponential':
#             equation = '''
#             $$\\text{Exponential Distribution Formula}$$
#             $$f(x) = exp(-x), \\text{ for } x \geq 0$$
#             $$\\text{Values}$$
#             $$\\begin{cases}n \\to \\text{Observations}\\\\x\\to\\text{Trials}\\\\p\\to\\text{Probability}\\end{cases}$$
#             '''
#
#         return ui.p(equation)


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
    @reactive.event(input.max)
    def update():
        current_value = input.observations()
        c_max_val = input.max()

        ui.update_slider('observations', max=c_max_val, value=current_value)


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
def update_plot_prop(input: Inputs, output: Outputs, session: Session):
    @reactive.Effect
    @reactive.event(input.distributions, input.prop, input.extra_prop)
    def update():
        choices = []
        if input.distributions() in cont_dist['names']:
            [choices.append(_) for _ in cont_dist['standard'][1:]]
        else:
            [choices.append(_) for _ in discrete_dist['standard'][1:]]

        choices.append(input.prop())

        ui.update_checkbox_group('plot_props', choices=choices)


@module.server
def create_dist_df(input: Inputs, output: Outputs, session: Session, data_frame: reactive.Value):
    @output
    @render.data_frame
    @reactive.Calc
    def data():
        obs = input.observations()
        dist_data = dict()
        random_state = None

        if input.seed() > 0:
            random_state = input.seed()

        # TODO Discrete, Continuous feature list
        # TODO Research about the usage of loc, scale arguments in the distribution methods.
        #   In some cases it seems to be the only way to actually increase the range of generated
        #   distribution
        # TODO there is a small glitch, that does not actually impair functionality.
        #   when switching distribution, because the UI loads after, it still grabs the previous dists options
        #   and tries to generate the dist with those, giving an error that fies itself after updating the input
        """
        ==== Discrete ====
        To add functionalities:
        * interval;

        More complex functionality to implement:
        * expect
        
        ==== Continuous ====
        To add functionalities:
        * interval;
        * moment;

        More complex functionality to implement:
        * expect
        """

        if input.distributions() == 'Normal':
            sd = input.sd()
            mean = input.mean()

            norm_dist = create_distribution_df('norm', True, obs,
                                               (input.prop, input.extra_prop),
                                               input.enbl_extra, {'loc': mean, 'scale': sd},
                                               random_state=random_state)

            dist_data = norm_dist

        if input.distributions() == 'Poisson':
            events = input.events()

            poisson_dist = create_distribution_df('poisson', False, obs,
                                                  (input.prop, input.extra_prop),
                                                  input.enbl_extra, [events],
                                                  random_state=random_state)

            dist_data = poisson_dist

        if input.distributions() == 'Exponential':
            scale = input.scale()

            expon_dist = create_distribution_df('expon', True, obs,
                                                (input.prop, input.extra_prop),
                                                input.enbl_extra, {'scale': scale},
                                                random_state=random_state)

            dist_data = expon_dist

        if input.distributions() == 'Geometric':
            prob = input.prob()

            geom_dist = create_distribution_df('geom', False, obs,
                                               (input.prop, input.extra_prop),
                                               input.enbl_extra, [prob],
                                               random_state=random_state)

            dist_data = geom_dist

        if input.distributions() == 'Binomial':
            prob = input.prob()
            trials = input.trials()

            binom_dist = create_distribution_df('binom', False, obs,
                                                (input.prop, input.extra_prop),
                                                input.enbl_extra, [trials, prob],
                                                random_state=random_state)

            dist_data = binom_dist

        if input.distributions() == 'Uniform':
            low = input.low()
            high = input.high()

            uniform_dist = create_distribution_df('uniform', True, obs,
                                                  (input.prop, input.extra_prop),
                                                  input.enbl_extra, {'loc': low, 'scale': high},
                                                  random_state=random_state)

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
    @reactive.event(input.plot_distribution, input.plot_other)
    def graph():
        plot_data = data_frame()['distribution_df']
        subplot_titles = [f'Histogram of {input.distributions()} distribution']
        to_plots = input.plot_props()
        widget = None

        for plot in input.plot_props():
            subplot_titles.append(f'{input.distributions()} {plot} plot')

        fig = make_subplots(rows=1, cols=1 + len(to_plots), subplot_titles=subplot_titles)

        fig.layout.title = f'{input.distributions()} Distribution plots'
        fig.layout.width, fig.layout.height = 1500, 600

        hist_trace = go.Histogram(x=plot_data['Observations'], name='Observations')

        fig.add_trace(hist_trace, 1, 1)

        fig.update_xaxes(title_text="Observations", row=1, col=1)
        fig.update_yaxes(title_text='Count', row=1, col=1)

        for c in range(1, len(to_plots) + 1):
            scatter = go.Scatter(x=plot_data['Observations'], y=plot_data[to_plots[c - 1]],
                                 mode='markers', name=to_plots[c - 1])

            fig.add_trace(scatter, row=1, col=1 + c)
            fig.update_xaxes(title_text="Observations", row=1, col=1 + c)
            fig.update_yaxes(title_text=to_plots[c - 1], row=1, col=1 + c)

        widget = go.FigureWidget(fig)

        @synchronize_size("graph")
        def on_size_changed(width, height):
            widget.layout.width = width
            widget.layout.height = height

        return widget
