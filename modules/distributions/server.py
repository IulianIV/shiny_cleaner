import numpy as np
import pandas as pd
from shiny import Inputs, Outputs, Session, module, render, ui, reactive
from config import MAX, MIN, STANDARD_DEVIATION, MEAN_SIGMA


@module.server
def create_distribution_inputs(input: Inputs, output: Outputs, session: Session):
    @output
    @render.ui
    @reactive.event(input.distributions)
    def distribution_inputs():
        min_val = MIN
        max_val = MAX
        sd = STANDARD_DEVIATION
        mean = MEAN_SIGMA

        if input.distributions() == 'Gaussian':
            return (
                ui.row(
                    ui.column(3, ui.input_numeric("min", "Min", value=min_val)),
                    ui.column(3, ui.input_numeric("max", "Max", value=max_val)),
                    ui.column(3, ui.input_numeric("mean", "Mean", value=mean)),
                    ui.column(3, ui.input_numeric("sd", "SD", value=sd))
                ), ui.input_switch("matrix", "Min x Max matrix"),
                ui.input_slider("observations", None, min=min_val, max=max_val, value=max_val / 2),
                ui.input_action_button("plot_distribution", "Plot")
            )


@module.server
def update_distribution_inputs(input: Inputs, output: Outputs, session: Session):
    # Update x_axis and y_axis inputs according to the group_by and aggregator inputs
    @reactive.Effect
    @reactive.event(input.min, input.max)
    def update_slider():
        current_value = input.observations()
        min_val = input.min()
        max_val = input.max()

        if min_val > max_val:
            ui.update_numeric('min', value=max_val)
            ui.update_numeric('max', value=min_val)
            ui.update_slider('observations', min=max_val, max=min_val, value=current_value)

        ui.update_slider('observations', min=min_val, max=max_val, value=current_value)


@module.server
def create_distribution_data_set(input: Inputs, output: Outputs, session: Session, data_frame: reactive.Value):
    @output
    @render.data_frame
    @reactive.Effect
    def summary_data():
        obs = input.observations()
        sd = input.sd()
        mean = input.mean()

        if input.distributions() == 'Gaussian':
            distribution = np.random.normal(mean, sd, obs)
            distribution_df = pd.DataFrame(data=distribution, index=range(1, len(distribution) + 1),
                                           columns=['valuee'])

            if input.matrix():
                distribution = np.random.normal(mean, sd, size=(input.min(), input.max()))
                distribution_df = pd.DataFrame(data=distribution[:, :], index=range(1, len(distribution) + 1),
                                               columns=[f'value_{x}' for x in range(1, distribution.shape[1] + 1)])

            data_frame.set(distribution_df)
            return render.DataGrid(
                data_frame().round(3),
                row_selection_mode="multiple",
                width="100%",
                height="100%",
            )
