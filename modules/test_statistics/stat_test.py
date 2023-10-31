from shiny import Inputs, Outputs, Session, module, render, ui, reactive


class Test():
    def ui(self):
        return (ui.p('Distribution 1'),
        ui.row(
            ui.column(4, ui.input_numeric('dist1_n', 'n', value=100)),
            ui.column(4, ui.input_numeric('dist1_m', 'µ', value=100)),
            ui.column(4, ui.input_numeric('dist1_s', 'σ', value=100))
        ))
