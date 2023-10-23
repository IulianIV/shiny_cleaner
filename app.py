from shiny import App, Inputs, Outputs, Session, render
from shiny import experimental as x
from shiny import reactive, ui

from modules.common import show_table, show_graph

from modules.summary.ui import summary_inputs
from modules.summary.server import (update_filename_input, load_data_frame, update_aggregator_input,
                                    update_graph_input, load_summary_data, create_graph, filter_df)

from modules.distributions.ui import distribution_selection
from modules.distributions.server import (create_distribution_inputs, load_distribution_data,
                                          update_distribution_inputs, create_distribution_data_set)

# column choices
app_ui = x.ui.page_fillable(
    ui.layout_sidebar(
        ui.panel_sidebar(
            {"class": "p-3"},
            ui.navset_tab_card(
                ui.nav("Data Summarizer", summary_inputs('summary')),
                ui.nav('Distributions', distribution_selection('distributions'), ui.output_ui('distribution_inputs')),
            ),
        ),
        x.ui.panel_main(
            x.ui.layout_column_wrap(
                1,
                ui.output_ui('_table'),
                x.ui.layout_column_wrap(
                    1,
                    ui.output_ui('_graph')
                ),
            ),

            # x.ui.layout_column_wrap(
            #     1,
            #     show_table('summary'),
            #     x.ui.layout_column_wrap(
            #         1,
            #         show_graph('summary')
            #     ),
            # )
        )

    ),
)


def server(input: Inputs, output: Outputs, session: Session):
    original_df = reactive.Value()
    grouper = reactive.Value()
    data_frame = reactive.Value()
    distribution_df = reactive.Value()

    update_filename_input('summary')

    load_data_frame('summary', grouper, original_df)

    update_aggregator_input('summary', grouper)

    update_graph_input('summary', grouper)

    load_summary_data('summary', original_df, data_frame)

    create_graph('summary', filter_df('summary', original_df, data_frame))

    @output
    @render.ui
    def _table():
        return show_table('summary')

    @output
    @render.ui
    def _graph():
        return show_graph('summary')

    @output
    @render.ui
    def distribution_inputs():
        create_distribution_inputs('distributions')

    update_distribution_inputs('distributions')

    create_distribution_data_set('distributions', distribution_df)

    load_distribution_data('distributions', distribution_df)


app = App(app_ui, server, debug=True)
