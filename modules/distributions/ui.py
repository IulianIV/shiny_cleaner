from config import DISTRIBUTIONS
from shiny import module, ui


@module.ui
def distribution_selection():
    return (
        ui.input_radio_buttons("distributions", f"Distributions", sorted(DISTRIBUTIONS)), ui.hr(),
        ui.p('Distribution settings'),
        ui.output_ui('distribution_inputs')
    )

