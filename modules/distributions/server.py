from __future__ import annotations

import numpy as np
import pandas
from scipy.stats import binom
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
safe_functions = config.server_config('safe_callable_dict')

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
        lower_bound = dist_defaults['lb']
        upper_bound = dist_defaults['ub']

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
            ui.row(ui.column(6, ui.input_checkbox('enbl_extra', 'Extra Properties')),
                   ui.column(6, ui.input_checkbox('enbl_expect', 'Expected Value'))),
            ui.panel_conditional('input.enbl_extra',
                                 ui.input_selectize('extra_prop', 'Extra Properties', discrete_dist['extra_methods'],
                                                    multiple=False)),
            ui.panel_conditional('input.enbl_expect',
                                 ui.input_text_area('expect_func', 'Function'),
                                 ui.input_checkbox('enbl_bounds', 'Enable Bounds'),
                                 ui.panel_conditional('input.enbl_bounds',
                                                      ui.input_numeric('expect_lb', 'Lower Bound', value=lower_bound),
                                                      ui.input_numeric('expect_ub', 'Upper Bound', value=upper_bound))),
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
def update_expect_bounds(input: Inputs, output: Outputs, session: Session):
    @reactive.Effect
    @reactive.event(input.expect_lb, input.expect_ub)
    def update():
        c_lb_val = input.expect_lb()
        c_ub_val = input.expect_ub()

        if c_lb_val > c_ub_val:
            ui.update_numeric('expect_lb', value=c_ub_val)
            ui.update_numeric('expect_ub', value=c_lb_val)


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
def update_dist_conf(input: Inputs, output: Outputs, session: Session):
    @reactive.Effect
    @reactive.event(input.confidence)
    def update():
        try:
            conf_value = float(input.confidence())

            if conf_value > 1:
                ui.update_numeric('confidence', value=1)
            elif conf_value < 0:
                ui.update_numeric('confidence', value=0)

        except ValueError as e:
            print(e)


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
        #   The calculation of some of these features (to be decided which) should be done automatically
        #   The graphical representation should be made by drop-down choice, where possible.
        #   Add a Comprehensive view of Distribution Details in the side-bar

        if input.distributions() == 'Normal':
            # TODO Normal feature list
            """
            Pre-rendered data: the table will be automatically rendered with this data:
            * pdf;
            * cdf;
            * stats -> only in side-bar

            Data to show by user choice:
            * logpdf;
            * logcdf;
            * sf;
            * logsf;

            Possibility of implementation:
            * ppf;
            * isf;
            * interval
            * moment
            * entropy;

            More complex functionality to implement:
            * expect
            * fit
            """
            sd = input.sd()
            mean = input.mean()

            distribution_df = create_distribution_df(input.distributions().lower(),
                                                     {'mean': mean, 'sd': sd, 'obs': obs, 'min': input.min,
                                                      'max': input.max},
                                                     input.matrix)
            dist_array.set(distribution_df)

        if input.distributions() == 'Poisson':
            # TODO Poisson feature list
            """
            Pre-rendered data: the table will be automatically rendered with this data:
            * pmf;
            * cdf;
            * stats -> only in side-bar

            Data to show by user choice:
            * logpmf;
            * logcdf;
            * sf;
            * logsf;

            Possibility of implementation:
            * ppf;
            * isf;
            * interval
            * entropy;

            More complex functionality to implement:
            * expect
            """
            events = input.events()

            distribution_df = create_distribution_df(input.distributions().lower(),
                                                     {'events': events, 'obs': obs, 'min': input.min, 'max': input.max},
                                                     input.matrix)

            dist_array.set(distribution_df)

        if input.distributions() == 'Exponential':
            # TODO Exponential feature list
            """
            Pre-rendered data: the table will be automatically rendered with this data:
            * pdf;
            * cdf;
            * stats -> only in side-bar

            Data to show by user choice:
            * logpdf;
            * logcdf;
            * sf;
            * logsf;

            Possibility of implementation:
            * ppf;
            * isf;
            * interval;
            * moment;
            * entropy;

            More complex functionality to implement:
            * expect
            * fit
            """
            scale = input.scale()

            distribution_df = create_distribution_df(input.distributions().lower(),
                                                     {'scale': scale, 'obs': obs, 'min': input.min,
                                                      'max': input.max},
                                                     input.matrix)

            dist_array.set(distribution_df)

        if input.distributions() == 'Geometric':
            # TODO Geometric feature list
            """
            Pre-rendered data: the table will be automatically rendered with this data:
            * pmf;
            * cdf;
            * stats -> only in side-bar

            Data to show by user choice:
            * logpmf;
            * logcdf;
            * sf;
            * logsf;

            Possibility of implementation:
            * ppf;
            * isf;
            * interval
            * entropy;

            More complex functionality to implement:
            * expect
            """
            prob = input.prob()

            distribution_df = create_distribution_df(input.distributions().lower(),
                                                     {'prob': prob, 'obs': obs, 'min': input.min,
                                                      'max': input.max},
                                                     input.matrix)

            dist_array.set(distribution_df)

        if input.distributions() == 'Binomial':
            # TODO Binomial feature list
            """
            Pre-rendered data: the table will be automatically rendered with this data:
            * pmf; Done
            * cdf; Done
            * stats -> only in side-bar Done

            Data to show by user choice:
            * logpmf; Done
            * logcdf; Done
            * sf; Done
            * logsf; Done

            Possibility of implementation:
            * ppf; Done
            * isf; Done
            * interval;
            * entropy;

            More complex functionality to implement:
            * expect
            """

            prob = input.prob()
            trials = input.trials()

            binomial = binom(trials, prob)
            binomial_rvs = binomial.rvs(size=obs)

            pmf = binomial.pmf(binomial_rvs)
            cdf = binomial.cdf(binomial_rvs)
            stats = binomial.stats(moments='mvsk')

            calc_user_option = getattr(binomial, input.prop().replace(' ', '').lower())(binomial_rvs)

            if input.enbl_extra():
                calc_extra_option = getattr(binomial, input.extra_prop().replace(' ', '').lower())(cdf)
                dist_array = np.vstack((binomial_rvs, pmf, cdf, calc_user_option, calc_extra_option))
            else:
                dist_array = np.vstack((binomial_rvs, pmf, cdf, calc_user_option))

            if input.enbl_expect():
                safe = ['acos', 'asin', 'atan', 'atan2', 'ceil', 'cos', 'cosh', 'degrees', 'e', 'exp', 'fabs',
                        'floor', 'fmod', 'frexp', 'hypot', 'ldexp', 'log', 'abs',
                        'log10', 'modf', 'pow', 'radians', 'sin', 'sinh', 'sqrt', 'tan', 'tanh']
                import compiler
                import math
                # safe_copy = dict([(k, locals().get(k, None)) for k in safe])
                x = 10
                # safe_copy['x'] = x
                # print(safe_copy)
                print(input.expect_func())

                # results = eval(input.expect_func(), {"__builtins__": None}, safe_copy)
                # print(results(x))

            new_df = pandas.DataFrame(dist_array.T)

            if input.enbl_extra():
                new_df.columns = ['Observations', 'CDF', 'PMF', input.prop(), input.extra_prop()]
            else:
                new_df.columns = ['Observations', 'CDF', 'PMF', input.prop()]

            dist_data['distribution_array'] = dist_array
            dist_data['distribution_df'] = new_df
            dist_data['stats'] = {k: round(v, 4) for k, v in zip(['mean', 'variance', 'skewness', 'kurtosis'], stats)}

        if input.distributions() == 'Uniform':
            # TODO Exponential feature list
            """
            Pre-rendered data: the table will be automatically rendered with this data:
            * pdf;
            * cdf;
            * stats -> only in side-bar

            Data to show by user choice:
            * logpdf;
            * logcdf;
            * sf;
            * logsf;

            Possibility of implementation:
            * ppf;
            * isf;
            * interval;
            * moment;
            * entropy;

            More complex functionality to implement:
            * expect
            * fit
            """
            low = input.low()
            high = input.high()

            distribution_df = create_distribution_df(input.distributions().lower(),
                                                     {'low': low, 'high': high, 'obs': obs, 'min': input.min,
                                                      'max': input.max},
                                                     input.matrix)

            dist_array.set(distribution_df)

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
