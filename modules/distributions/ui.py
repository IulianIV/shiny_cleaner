from config import Config
from shiny import module, ui

cont_dist = Config.input_config('distributions')['continuous']['names']
discrete_dist = Config.input_config('distributions')['discrete']['names']

@module.ui
def distribution_selection():
    return (
        ui.input_radio_buttons('distributions', f'Distributions', sorted(cont_dist + discrete_dist)),
        ui.hr(),
        ui.p('Distribution settings'),
        ui.output_ui('inputs'),
        ui.hr(),
        ui.output_text_verbatim('details')
    )
