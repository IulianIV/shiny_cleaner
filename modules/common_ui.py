from shiny import module, ui
from shiny import experimental as x
from shinywidgets import output_widget


@module.ui
def show_table():
    return x.ui.card(
        ui.output_data_frame('data')
    )


@module.ui
def show_graph():
    return x.ui.card(
        output_widget('graph', height='100%')
    )
