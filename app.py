from shiny import App, Inputs, Outputs, Session, render
from shiny import experimental as x
from shiny import reactive, ui

from config import Config

from modules.common_ui import show_table, show_graph

from modules.summary.ui import summary_inputs
from modules.summary.server import (update_filename_input, load_data_frame, update_aggregator_input,
                                    update_graph_input, load_summary_data, create_graph, filter_df)

from modules.distributions.ui import distribution_selection
from modules.distributions.server import (create_distribution_inputs, load_distribution_data,
                                          update_distribution_inputs, create_distribution_data_set, distribution_graph)


# TODO Check import management
#   if there are some imports, such as numpy, that are used only in certain functions, import the library inside that
#   function. This way the global namespace is less cluttered.

app_width = Config.ui_config('width')
app_height = Config.ui_config('height')

# column choices
app_ui = x.ui.page_fillable(
    ui.navset_tab_card(
        ui.nav(
            'Data Summarizer',
            x.ui.layout_sidebar(
                x.ui.sidebar(
                    {'class': 'p-3'},
                    summary_inputs('summary'),
                    width=app_width
                ),
                x.ui.layout_column_wrap(
                    1,
                    show_table('summary'),
                    x.ui.layout_column_wrap(
                        1,
                        show_graph('summary')
                    ),
                ),
                height=app_height
            )
        ),
        ui.nav(
            'Distributions',
            x.ui.layout_sidebar(
                x.ui.sidebar(
                    {'class': 'p-3'},
                    distribution_selection('distributions'), ui.output_ui('distribution_inputs'),
                    width=app_width
                ),
                x.ui.layout_column_wrap(
                    1,
                    show_table('distributions'),
                    show_graph('distributions')
                ),
                height=app_height
            )
        ),
    )
)


def server(input: Inputs, output: Outputs, session: Session):
    original_df = reactive.Value()
    grouper = reactive.Value()
    summary_df = reactive.Value()
    distribution_df = reactive.Value()

    update_filename_input('summary')

    load_data_frame('summary', grouper, original_df)

    update_aggregator_input('summary', grouper)

    update_graph_input('summary', grouper)

    load_summary_data('summary', original_df, summary_df)

    create_graph('summary', filter_df('summary', original_df, summary_df))

    @output
    @render.ui
    def distribution_inputs():
        create_distribution_inputs('distributions')

    update_distribution_inputs('distributions')

    create_distribution_data_set('distributions', distribution_df)

    load_distribution_data('distributions', distribution_df)

    distribution_graph('distributions', distribution_df)


app = App(app_ui, server, debug=False)
