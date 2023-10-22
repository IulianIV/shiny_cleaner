import pandas

import plotly.express as px
import plotly.graph_objs as go
from shiny import Inputs, Outputs, Session, module, render, ui, reactive, req
from shinywidgets import render_widget

import os
import re

from utils import get_data_files, create_summary_df, synchronize_size


@module.server
def get_files(input: Inputs, output: Outputs, session: Session):
    @reactive.Effect
    def update_file_names():
        files = [name[0] for name in get_data_files()]

        ui.update_selectize(
            'file_name',
            choices=files,
            selected=None
        )


@module.server
def load_data_frame(input: Inputs, output: Outputs, session: Session, grouper, original_df):
    # also updates the 'group_by' input
    @reactive.Effect
    @reactive.event(input.load_file)
    def load_frame():
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


@module.server
def update_aggregator_input(input: Inputs, output: Outputs, session: Session, grouper):
    @reactive.Effect
    @reactive.event(input.group_by)
    def update_aggregator_input():
        cols = grouper()[:]

        # Remove from the columns input the currently selected group_by value
        cols.remove(input.group_by())

        ui.update_selectize(
            "aggregator",
            choices=cols,
            selected=None
        )


@module.server
def update_graph_input(input: Inputs, output: Outputs, session: Session, grouper):
    # Update x_axis and y_axis inputs according to the group_by and aggregator inputs
    @reactive.Effect
    def update_graph_input():
        x_axis = grouper()
        y_axis = input.aggregator()

        ui.update_selectize('x_ax', choices=x_axis, selected=None)
        ui.update_selectize('y_ax', choices=y_axis, selected=None)


@module.server
def load_summary_data(input: Inputs, output: Outputs, session: Session, original_df, data_frame):
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


@module.server
def filter_df(input: Inputs, output: Outputs, session: Session, original_df, data_frame):
    @reactive.Calc
    def filtered_df():
        selected_idx = list(req(input.summary_data_selected_rows()))
        selection = data_frame()[input.group_by()][selected_idx]

        # Filter data for selected countries
        return original_df()[original_df()[input.group_by()].isin(selection)]

    return filtered_df


@module.server
def show_graph(input: Inputs, output: Outputs, session: Session, filtered_df):
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


