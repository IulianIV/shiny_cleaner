from config import Config
from shiny import module, ui

distributions = Config.input_config('distributions')


@module.ui
def distribution_selection():
    return (
        ui.input_radio_buttons('distributions', f'Distributions', sorted(distributions)),
        ui.hr(),
        ui.p('Distribution settings'),
        ui.output_ui('inputs'),
        ui.hr(),
        ui.output_text_verbatim('details')
    )
