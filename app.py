from shiny import App, Inputs, Outputs, Session, render
from shiny import experimental as x
from shiny import reactive, ui

from config import Config

from modules.common_ui import show_table, show_graph

from modules.summary.ui import summary_inputs
from modules.summary.server import (update_filename_input, load_data_frame, update_aggregator_input,
                                    update_graph_input, load_summary_data, create_graph, filter_df)

from modules.distributions.ui import distribution_selection, create_dist_settings
from modules.distributions.server import (update_dist_prob, update_plot_prop,
                                          update_dist_min_max, create_dist_df, update_dist_prop_select,
                                          create_dist_details, dist_graph)  # dist_eq


# TODO Check import management
#   if there are some imports, such as numpy, that are used only in certain functions, import the library inside that
#   function. This way the global namespace is less cluttered.

app_width = Config.ui_config('width')
app_height = Config.ui_config('height')
dist_id = 'distributions'
summary_id = 'summary'
statistics_id = 'statistics'

# column choices
app_ui = x.ui.page_fillable(
    # ui.head_content(
    #     ui.tags.script(
    #         src="https://mathjax.rstudio.com/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"
    #     ),
    #     ui.tags.script(
    #         "if (window.MathJax) MathJax.Hub.Queue(['Typeset', MathJax.Hub]);"
    #     )
    # ),
    ui.navset_tab_card(
        ui.nav(
            'Distributions',
            x.ui.layout_sidebar(
                x.ui.sidebar(
                    {'class': 'p-3'},
                    distribution_selection(dist_id),
                    # ui.output_ui('print_dist_eq'),
                    ui.output_ui('distribution_inputs'),
                    ui.output_ui('distribution_details'),
                    width=app_width
                ),
                x.ui.layout_column_wrap(
                    1,
                    show_table(dist_id),
                    show_graph(dist_id),
                ),
                height=app_height
            )
        ),
        ui.nav(
            'Data Summarizer',
            x.ui.layout_sidebar(
                x.ui.sidebar(
                    {'class': 'p-3'},
                    summary_inputs(summary_id),
                    width=app_width
                ),
                x.ui.layout_column_wrap(
                    1,
                    show_table(summary_id),
                    x.ui.layout_column_wrap(
                        1,
                        show_graph(summary_id)
                    ),
                ),
                height=app_height
            )
        )
    )
)


def server(input: Inputs, output: Outputs, session: Session):
    orig_summary_df = reactive.Value()
    grouper = reactive.Value()
    summary_df = reactive.Value()
    dist_data = reactive.Value()

    # Distributions Section ####
    @output
    @render.ui
    def distribution_inputs():
        create_dist_settings(dist_id)

    @output
    @render.ui
    def distribution_details():
        create_dist_details(dist_id, dist_data()['stats'])

    # @output
    # @render.ui
    # def print_dist_eq():
    #     dist_eq(dist_id)

    create_dist_df(dist_id, dist_data)

    update_dist_min_max(dist_id)

    update_dist_prob(dist_id)

    update_dist_prop_select(dist_id)

    update_plot_prop(dist_id)

    dist_graph(dist_id, dist_data)

    # Summary Section ####

    update_filename_input(summary_id)

    load_data_frame(summary_id, grouper, orig_summary_df)

    update_aggregator_input(summary_id, grouper)

    update_graph_input(summary_id, grouper)

    load_summary_data(summary_id, orig_summary_df, summary_df)

    create_graph(summary_id, filter_df(summary_id, orig_summary_df, summary_df))


app = App(app_ui, server, debug=False)
