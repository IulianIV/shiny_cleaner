from shiny import module, ui
from shiny import experimental as x
from shinywidgets import output_widget

from config import Config

graph_height_percent = Config.ui_config('graph_height_percent')

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
