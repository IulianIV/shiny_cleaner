from config import Config
from shiny import Inputs, Outputs, Session, module, render, ui, reactive

config = Config()
divergence_names = config.input_config('divergence')['names']


@module.ui
def divergence_selection():
    return (
        ui.row(
            ui.column(6, ui.input_radio_buttons('divergences', f'Divergences', sorted(divergence_names))),
            ui.column(6, ui.input_action_button('compute', 'Compute Divergence', class_='btn-warning'),
                      style="display: flex; align-items: center;")
        ),
        ui.hr(),
        ui.input_switch('show_all', 'Show Elementwise calculations'),
        # related to the elementwise print of data
        # ui.panel_conditional('input.show_all', ui.p('stuff')),
        # ui.output_text_verbatim('details'),

    )
