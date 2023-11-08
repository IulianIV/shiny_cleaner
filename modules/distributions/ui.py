from config import Config
from shiny import Inputs, Outputs, Session, module, render, ui, reactive
from modules.common_ui import label_with_tooltip

config = Config()
cont_dist = config.input_config('distributions')['continuous']
discrete_dist = config.input_config('distributions')['discrete']
qmark = config.ui_config('tooltip_q')
graph_height = config.ui_config('graph_height')
dist_defaults = config.input_config('distributions')


dist_names = cont_dist['names'] + discrete_dist['names']

@module.ui
def distribution_selection():
    return (
        ui.input_radio_buttons('distributions', f'Distributions', sorted(dist_names)),
        # ui.output_ui('eq'),
        ui.hr(),
        ui.output_ui('inputs'),
        ui.hr(),
        ui.output_text_verbatim('details'),
    )

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
            dist_inputs = ui.column(4, ui.input_numeric('prob', 'Probability', value=prob, step=0.05))

        elif input.distributions() == 'Binomial':
            dist_inputs = (ui.column(3, ui.input_numeric('trials', 'Trials', value=trials)),
                           ui.column(3, ui.input_numeric('prob', 'Probability', value=prob, step=0.05)))

        elif input.distributions() == 'Uniform':
            dist_inputs = (ui.column(3, ui.input_numeric('low', 'Low', value=low)),
                           ui.column(3, ui.input_numeric('high', 'High', value=high)))

        # elif input.distributions() == 'Cauchy':
        #     dist_inputs = (ui.column(3, ui.input_numeric('scale', 'Scale', value=scale)),
        #                    ui.column(3, ui.input_numeric('location', 'Location', value=0))
        #                    )

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

        dist_plot = (ui.row(ui.column(5, ui.input_action_button('plot_distribution', 'Plot Histogram', class_='btn-success')),
                            ui.column(7, ui.input_checkbox('enbl_plot', 'Other Plots'))),
                     ui.panel_conditional('input.enbl_plot', ui.input_checkbox_group(
                         "plot_props",
                         "Other plots:", [],
                     ), ui.input_action_button('plot_other', 'Plot Other', class_='btn-primary')))

        return (
            ui.row(dist_head,
                   dist_inputs,
                   ),
            dist_options,
            dist_plot
        )

