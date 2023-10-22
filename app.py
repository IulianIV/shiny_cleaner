import os.path
import re

import pandas  # noqa: F401 (this line needed for Shinylive to load plotly.express)
import plotly.express as px
import plotly.graph_objs as go
from shiny import App
from shiny import experimental as x
from shiny import reactive, render, req, session, ui
from shinywidgets import output_widget, render_widget

from utils import create_summary_df, get_data_files

operations = ['min', 'max', 'mean']
fallback = ['count']

# column choices
app_ui = x.ui.page_fillable(
    ui.layout_sidebar(
        ui.panel_sidebar(
            {"class": "p-3"},
            ui.input_selectize("file_name", f"Select File", []),
            ui.input_action_button("load_file", "Load File"),
            ui.panel_conditional('input.load_file',
                                 ui.p(
                                     ui.strong("Instructions:"),
                                     " Select columns to group and aggregate by.",
                                 ),
                                 ui.input_selectize("group_by", f"Group By", []),
                                 ui.input_selectize("aggregator", f"Aggregate", [], multiple=True),
                                 ui.input_selectize("operations", f"Operations", operations, multiple=True),
                                 ui.input_selectize("fallbacks", f"Fallback Operations", fallback, multiple=True),
                                 ui.input_action_button("submit", "Summarize"),
                                 ui.panel_conditional(
                                     "input.submit",
                                     ui.input_selectize("x_ax", f"X-axis", []),
                                     ui.input_selectize("y_ax", f"Y-axis", []),
                                     ui.input_action_button("plot", "Plot Graph"),

                                 )
                                 )
        ),
        x.ui.panel_main(
            x.ui.layout_column_wrap(
                1,
                x.ui.card(
                    ui.output_data_frame("summary_data")
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
    original_df = reactive.Value()
    grouper = reactive.Value()
    data_frame = reactive.Value()

    @reactive.Effect
    def update_file_names():
        files = [name[0] for name in get_data_files()]

        ui.update_selectize(
            'file_name',
            choices=files,
            selected=None
        )

    # also updates the 'group_by' input
    @reactive.Effect
    @reactive.event(input.load_file)
    def load_data_frame():
        name = os.path.join('data', input.file_name() + '.csv')
        df = pandas.read_csv(name)
        df = df.dropna()
        df.columns = [re.sub('[ -]{1,}', '_', col.lower().strip()) for col in df.columns]

        col_names = [col for col in df.columns.values]

        grouper.set(col_names)
        original_df.set(df)

        ui.update_selectize(
            'group_by',
            choices=grouper(),
            selected=grouper()[0]
        )

    # Remove from the columns input the currently selected group_by value

    @reactive.Effect
    @reactive.event(input.group_by)
    def update_aggregator_input():
        cols = grouper()[:]

        cols.remove(input.group_by())

        ui.update_selectize(
            "aggregator",
            label=f"Aggregate by",
            choices=cols,
            selected=None
        )

    # Update x_axis and y_axis inputs according to the group_by and aggregator inputs
    @reactive.Effect
    def update_graph_input():
        x_axis = grouper()
        y_axis = input.aggregator()

        ui.update_selectize('x_ax', choices=x_axis, selected=None)
        ui.update_selectize('y_ax', choices=y_axis, selected=None)

    @output
    @render.data_frame
    @reactive.event(input.submit)
    def summary_data():

        values = [input.group_by(), input.aggregator(), input.operations(), input.fallbacks()]

        # Make sure that strings get converted to list to work with pandas.aggregate
        for value in values:
            if isinstance(value, str):
                new_value = ''.join([x for x in value])
                values[values.index(value)] = new_value

        selected_df = create_summary_df(original_df(), values[0], values[1], values[2], values[3])

        data_frame.set(selected_df)

        return render.DataGrid(
            data_frame().round(2),
            row_selection_mode="multiple",
            width="100%",
            height="100%",
        )

    @reactive.Calc
    def filtered_df():
        selected_idx = list(req(input.summary_data_selected_rows()))
        selection = data_frame()[input.group_by()][selected_idx]

        # Filter data for selected countries
        return original_df()[original_df()[input.group_by()].isin(selection)]

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
