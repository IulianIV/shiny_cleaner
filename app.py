import pandas  # noqa: F401 (this line needed for Shinylive to load plotly.express)
import re
import plotly.express as px
import plotly.graph_objs as go
from shinywidgets import output_widget, render_widget

from shiny import App
from shiny import experimental as x
from shiny import reactive, render, req, session, ui

# example taken & adapted from: https://shiny.posit.co/py/api/render.data_frame.html

# Load data from CSV file
new_df = pandas.read_csv('data/Life Expectancy Data.csv')

# Strip columns of trailing whitespace, lower the name of the column and replace all spaces and '-' with an underscore.
new_df.columns = [re.sub('[ -]{1,}', '_', col.lower().strip()) for col in new_df.columns]

# Create summary dataframe
new_summary_df = (
    new_df.groupby('country')
    .agg(
        {
            'life_expectancy': ['min', 'max', 'mean'],
            'adult_mortality': ['min', 'max', 'mean'],
            'population': ['min', 'max', 'mean'],
            'gdp': ['min', 'max', 'mean']
        }
    ).reset_index()
)

# drop NaN values and join aggregated columns
new_summary_df = new_summary_df.dropna()
new_summary_df.columns = ['_'.join(col) for col in new_summary_df.columns.values]
new_summary_df.rename(columns={"country_": "country"}, inplace=True)

app_ui = x.ui.page_fillable(
    {"class": "p-3"},
    ui.p(
        ui.strong("Instructions:"),
        " Select one or more countries in the table below to see more information.",
    ),
    x.ui.layout_column_wrap(
        1,
        x.ui.card(
            ui.output_data_frame("summary_data"),
        ),
        x.ui.layout_column_wrap(
            2 / 4,
            x.ui.card(
                output_widget("country_detail_pop", height="100%"),
            ),
            x.ui.card(
                output_widget("country_detail_gdp", height="100%"),
            ),
            x.ui.card(
                output_widget("country_detail_life_expectancy", height="100%"),
            ),
            x.ui.card(
                output_widget("country_detail_adult_mortality", height="100%"),
            )
        ),
    ),
)


def server(input, output, session):
    @output
    @render.data_frame
    def summary_data():
        return render.DataGrid(
            new_summary_df.round(2),
            row_selection_mode="multiple",
            width="100%",
            height="100%",
        )

    @reactive.Calc
    def filtered_df():
        selected_idx = list(req(input.summary_data_selected_rows()))
        countries = new_summary_df["country"][selected_idx]
        # Filter data for selected countries
        return new_df[new_df["country"].isin(countries)]

    @output
    @render_widget
    def country_detail_pop():
        # Create the plot
        fig = px.line(
            filtered_df(),
            x="year",
            y="population",
            color="country",
            title="Population Over Time",
        )
        widget = go.FigureWidget(fig)

        @synchronize_size("country_detail_pop")
        def on_size_changed(width, height):
            widget.layout.width = width
            widget.layout.height = height

        return widget

    @output
    @render_widget
    def country_detail_gdp():
        # Create the plot
        fig = px.line(
            filtered_df(),
            x="year",
            y="gdp",
            color="country",
            title="GDP per Capita Over Time",
        )
        widget = go.FigureWidget(fig)

        @synchronize_size("country_detail_percap")
        def on_size_changed(width, height):
            widget.layout.width = width
            widget.layout.height = height

        return widget

    @output
    @render_widget
    def country_detail_life_expectancy():
        # Create the plot
        fig = px.line(
            filtered_df(),
            x="year",
            y="life_expectancy",
            color="country",
            title="Life Expectancy Over Time",
        )
        widget = go.FigureWidget(fig)

        @synchronize_size("country_detail_life_expectancy")
        def on_size_changed(width, height):
            widget.layout.width = width
            widget.layout.height = height

        return widget

    @output
    @render_widget
    def country_detail_adult_mortality():
        # Create the plot
        fig = px.line(
            filtered_df(),
            x="year",
            y="adult_mortality",
            color="country",
            title="Adult Mortality Over Time",
        )
        widget = go.FigureWidget(fig)

        @synchronize_size("country_detail_adult_mortality")
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
