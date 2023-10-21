import pandas  # noqa: F401 (this line needed for Shinylive to load plotly.express)
import re
import plotly.express as px
import plotly.graph_objs as go
from shinywidgets import output_widget, render_widget

from shiny import App
from shiny import experimental as x
from shiny import reactive, render, req, session, ui

# TODO before any filtering is done and 'submit' is pressed, show the original data in the page
#   after submit is pressed show the filtered out data.
# TODO Try and see if you can implement tabs as you tried before
# TODO in the side bar create one section for reporting and one for editing.

# example taken & adapted from: https://shiny.posit.co/py/api/render.data_frame.html

# Load data from CSV file
new_df = pandas.read_csv('data/Life Expectancy Data.csv')
new_df = new_df.dropna()

# Strip columns of trailing whitespace, lower the name of the column and replace all spaces and '-' with an underscore.
new_df.columns = [re.sub('[ -]{1,}', '_', col.lower().strip()) for col in new_df.columns]


def create_summary_df(data_frame: pandas.DataFrame, group_by: str, aggregators: tuple[str],
                      functions: list[str], fallback_functions: list[str] = None):
    """
    Create a summary DataFrame by grouping based on `group_by`, aggregating by columns from `aggregator` and
    applying a function on all found columns with `actions`
    By default functions is: ['min', 'max', 'mean']
    :param fallback_functions: if any columns from aggregators are not numeric, do the fallback function 'count' instead
    :param data_frame: DataFrame to summarize
    :param group_by: Column to group by
    :param aggregators: Columns to aggregate by
    :param functions: possible functions to apply e.g. [np.sum, 'mean']
    :return:
    """
    if functions is None:
        functions = ['min', 'max', 'mean']

    if fallback_functions is None:
        fallback_functions = ['count']

    df = data_frame

    aggs = {k: list(functions) if pandas.api.types.is_numeric_dtype(df[k]) else fallback_functions for k in aggregators}
    summarized_df = (df.groupby(group_by).agg(
        aggs
    ).reset_index()
                     )
    summarized_df.columns = [re.sub('^_|_$', '', '_'.join(col)) for col in summarized_df.columns.values]

    return summarized_df


# drop NaN values and join aggregated columns

grouper = [col for col in new_df.columns.values]
operations = ['min', 'max', 'mean']
fallback = ['count']

# column choices
app_ui = x.ui.page_fillable(
    x.ui.layout_sidebar(
        x.ui.panel_sidebar(
            {"class": "p-3"},
            ui.p(
                ui.strong("Instructions:"),
                " Select an grouper col and one or more columns to summarize the data by.",
            ),
            ui.input_selectize("group_by", f"Select Columns to Group By", grouper),
            ui.input_selectize("aggregator", f"Select Columns to Aggregate", [], multiple=True),
            ui.input_selectize("operations", f"Select Operations", operations, multiple=True),
            ui.input_selectize("fallbacks", f"Fallback Operations", fallback, multiple=True),
            ui.input_action_button("submit", "See Summary"),
            ui.panel_conditional(
                "input.submit",
                ui.input_selectize("x_ax", f"X-axis", []),
                ui.input_selectize("y_ax", f"Y-axis", []),
                ui.input_action_button("plot", "Plot Graph"),

            )

        ),
        x.ui.panel_main(
            x.ui.layout_column_wrap(
                1,
                x.ui.card(
                    ui.output_data_frame("summary_data"),
                ),
                x.ui.layout_column_wrap(
                    1,
                    x.ui.card(
                        output_widget("show_summary_graph", height="100%"),
                    )
                ),
            )
        )
    ),
)


def server(input, output, session):
    data_frame = reactive.Value(None)

    @reactive.Effect()
    def update_aggregator_input():
        cols = grouper[:]
        cols.remove(input.group_by())

        return ui.update_selectize(
            "aggregator",
            label=f"Select columns ({len(cols)} options)",
            choices=cols,
            selected=None
        )

    @reactive.Effect()
    def update_graph_input():
        x_axis = grouper
        y_axis = input.aggregator()

        ui.update_selectize('x_ax', choices=x_axis, selected=None)
        ui.update_selectize('y_ax', choices=y_axis, selected=None)

    @reactive.Effect
    @reactive.event(input.submit)
    def summary_df():
        values = [input.group_by(), input.aggregator(), input.operations(), input.fallbacks()]

        for value in values:
            if isinstance(value, str):
                new_value = ''.join([x for x in value])
                values[values.index(value)] = new_value

        selected_df = create_summary_df(new_df, values[0], values[1], values[2], values[3])

        data_frame.set(selected_df)

    @output
    @render.data_frame
    @reactive.event(input.submit)
    def summary_data():
        return render.DataGrid(
            data_frame().round(2),
            row_selection_mode="multiple",
            width="100%",
            height="100%",
        )

    @reactive.Calc
    def filtered_df():
        selected_idx = list(req(input.summary_data_selected_rows()))
        countries = data_frame()[input.group_by()][selected_idx]
        # Filter data for selected countries
        return new_df[new_df[input.group_by()].isin(countries)]

    @output
    @render_widget
    @reactive.event(input.plot)
    def show_summary_graph():
        # Create the plot
        fig = px.line(
            filtered_df(),
            x=input.x_ax(),
            y=input.y_ax(),
            color=input.group_by(),
            title=f"{input.x_ax().title()} vs. {input.y_ax().replace('_', ' ').title()}",
        )
        widget = go.FigureWidget(fig)

        @synchronize_size("show_summary_graph")
        def on_size_changed(width, height):
            widget.layout.width = width
            widget.layout.height = height

        return widget


# This is a hacky workaround to help Plotly plots automatically
# resize to fit their container. In the future we'll have a
# built-in solution for this.
def synchronize_size(output_id):
    def wrapper(func):
        input = session.get_current_session().input
        print(input[f".clientdata_output_{output_id}_width"]())
        print(input[f".clientdata_output_{output_id}_height"]())
        @reactive.Effect
        def size_updater():
            func(
                input[f".clientdata_output_{output_id}_width"](),
                input[f".clientdata_output_{output_id}_height"](),
            )

        # When the output that we're synchronizing to is invalidated,
        # clean up the size_updater Effect.
        reactive.get_current_context().on_invalidate(size_updater.destroy)

        return size_updater

    return wrapper


app = App(app_ui, server)
