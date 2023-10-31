from typing import Literal

from shiny import module, ui
from shiny import experimental as x
from shinywidgets import output_widget

from config import Config

config = Config()

graph_height_percent = config.ui_config('graph_height_percent')
qmark = config.ui_config('tooltip_q')

@module.ui
def show_table():
    return x.ui.card(
        ui.output_data_frame('data')
    )


@module.ui
def show_graph():
    return x.ui.card(
        output_widget('graph', height=graph_height_percent)
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

