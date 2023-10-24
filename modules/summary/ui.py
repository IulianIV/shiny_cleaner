from shiny import module, ui
from config import Config

operations = Config.input_config('summary_operations')
fallback = Config.input_config('summary_fallback')

@module.ui
def summary_inputs():
    return (ui.input_selectize('file_name', f'Select File', []),
            ui.input_action_button('load_file', 'Load File'),
            ui.panel_conditional('input.load_file',
                                 ui.p(
                                     ui.strong('Instructions:'),
                                     ' Select columns to group and aggregate by.',
                                 ),
                                 ui.input_selectize('group_by', f'Group By', []),
                                 ui.input_selectize('aggregator', f'Aggregate By', [], multiple=True),
                                 ui.input_selectize('operations', f'Operations', operations, multiple=True),
                                 ui.input_selectize('fallbacks', f'Fallback Operations', fallback, multiple=True),
                                 ui.input_action_button('submit', 'Summarize'),
                                 ui.panel_conditional(
                                     'input.submit',
                                     ui.row(
                                         ui.column(6, ui.input_selectize('x_ax', f'X-axis', [])),
                                         ui.column(6, ui.input_selectize('y_ax', f'Y-axis', []))
                                     ),
                                     ui.row(
                                         ui.column(12, ui.input_action_button('plot', 'Plot Graph'),
                                                   style='display:inline-block')
                                     )

                                 )
                                 )
            )
