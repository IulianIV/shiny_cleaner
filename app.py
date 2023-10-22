from shiny import App, Inputs, Outputs, Session
from shiny import experimental as x
from shiny import reactive, ui

from modules.summary.ui import summary_inputs, summary_graph, summary_table
from modules.summary.server import (get_files, load_data_frame, update_aggregator_input,
                                    update_graph_input, load_summary_data, show_graph, filter_df)

# column choices
app_ui = x.ui.page_fillable(
    ui.layout_sidebar(
        summary_inputs('summary_inputs'),
        x.ui.panel_main(
            x.ui.layout_column_wrap(
                1,
                summary_table('summary_inputs'),
                x.ui.layout_column_wrap(
                    1,
                    summary_graph('summary_inputs')
                ),
            )
        )
    ),
)


def server(input: Inputs, output: Outputs, session: Session):
    original_df = reactive.Value()
    grouper = reactive.Value()
    data_frame = reactive.Value()

    get_files('summary_inputs')

    load_data_frame('summary_inputs', grouper, original_df)

    update_aggregator_input('summary_inputs', grouper)

    update_graph_input('summary_inputs', grouper)

    load_summary_data('summary_inputs', original_df, data_frame)

    show_graph('summary_inputs', filter_df('summary_inputs', original_df, data_frame))


app = App(app_ui, server)
