from typing import Literal

from config import Config
from shiny import module, ui
import shiny.experimental as x

config = Config()
cont_dist = config.input_config('distributions')['continuous']['names']
discrete_dist = config.input_config('distributions')['discrete']['names']
qmark = config.ui_config('tooltip_q')


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


def label_with_tooltip(label: str, q_mark: bool, tooltip_text: str,
                       placement: Literal['right', 'left', 'top', 'down', 'auto'], tooltip_id: str):
    if q_mark:
        return x.ui.tooltip(
            ui.span(label, qmark),
            tooltip_text,
            placement=placement,
            id=tooltip_id,
        )
    else:
        return x.ui.tooltip(
            ui.span(label),
            tooltip_text,
            placement=placement,
            id=tooltip_id,
        )
