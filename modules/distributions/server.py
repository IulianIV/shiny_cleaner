from __future__ import annotations

import numpy as np
import pandas
from scipy.stats import binom
from shiny import Inputs, Outputs, Session, module, render, ui, reactive
from shinywidgets import render_widget

import plotly.express as px
import plotly.graph_objs as go

from utils import synchronize_size, two_dim_to_one_dim, create_distribution_df

from config import Config

graph_height = Config.ui_config('graph_height')
distributions = Config.input_config('distributions')
distributions_prop = Config.input_config('distributions_prop')

@module.server
def create_dist_inputs(input: Inputs, output: Outputs, session: Session):
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
        prob = Config.input_config('distributions_probability')
        trials = Config.input_config('distributions_trials')
        low = Config.input_config('distributions_low')
        high = Config.input_config('distributions_high')

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
            ui.input_selectize('prop', 'Show Properties', distributions_prop['Binomial'], multiple=False),
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
        props = distributions_prop[input.distributions()]
        ui.update_selectize('prop', choices=props, selected=None)


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
            * logpmf;
            * logcdf;
            * sf;
            * logsf;

            Possibility of implementation:
            * ppf;
            * isf;
            * interval;
            * entropy;

            More complex functionality to implement:
            * expect
            """

            prob = input.prob()
            trials = input.trials()

            distribution_array = binom.rvs(trials, prob, size=obs)

            pmf = binom.pmf(distribution_array, trials, prob)
            cdf = binom.cdf(distribution_array, trials, prob)
            stats = binom.stats(trials, prob, moments='mvsk')

            dist_array = np.vstack((distribution_array, pmf, cdf))

            if input.prop() == 'Log PMF':
                log_pmf = binom.logpmf(distribution_array, trials, prob)
                dist_array = np.vstack((dist_array, log_pmf))
            elif input.prop() == 'Log CDF':
                log_cdf = binom.logcdf(distribution_array, trials, prob)
                dist_array = np.vstack((dist_array, log_cdf))

            new_df = pandas.DataFrame(dist_array.T)

            if input.prop() == 'Log PMF':
                new_df.columns = ['Observations', 'CDF', 'PMF', 'Log PMF']
            elif input.prop() == 'Log CDF':
                new_df.columns = ['Observations', 'CDF', 'PMF', 'Log CDF']
            else:
                new_df.columns = ['Observations', 'CDF', 'PMF']

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
