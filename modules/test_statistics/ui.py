from typing import Literal

from config import Config
from shiny import Inputs, Outputs, Session, module, render, ui, reactive

import shiny.experimental as x

from modules.test_statistics.stat_test import Test

config = Config()
stat_tests = config.input_config('statistical_testing')

qmark = config.ui_config('tooltip_q')

test = Test()

@module.ui
def test_statistic_inputs():
    return (
        ui.input_radio_buttons('tests', f'Statistical Tests', sorted(stat_tests['tests'])),
        ui.hr(),
        ui.output_ui('inputs'),
    )


@module.server
def create_statistic_settings(input: Inputs, output: Outputs, session: Session):
    @output
    @render.ui
    def inputs():
        return test.ui()
