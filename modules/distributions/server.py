from shiny import Inputs, Outputs, Session, module, render, ui, reactive


@module.server
def create_distribution_inputs(input: Inputs, output: Outputs, session: Session):
    @output
    @render.ui
    @reactive.event(input.distributions)
    def distribution_inputs():
        if input.distributions() == 'Gaussian':
            return ui.input_slider("distributions", None, min=0, max=100, value=50)

