from config import Config
from shiny import module, ui
import shiny.experimental as x

config = Config()
divergence_names = config.input_config('divergence')['names']
divergence_docs = config.input_config('divergence')['doc_names']


@module.ui
def divergence_selection():
    return (
        ui.row(
            ui.column(6, ui.input_radio_buttons('divergences', f'Divergences', sorted(divergence_names))),
            ui.column(6, ui.input_action_button('compute', 'Compute Divergence', class_='btn-warning'),
                      style="display: flex; align-items: center;")
        ),
        ui.hr(),
        ui.input_switch('show_all', 'Show elementwise calculations'),
        # related to the elementwise print of data
        ui.output_text_verbatim('details'),

    )


@module.ui
def divergence_doc_urls():
    docs_urls = tuple()

    for doc, name in zip(divergence_docs, divergence_names):
        docs_urls = docs_urls + (ui.nav_control(ui.a(name, href=f'/docs/{doc}', target='_blank')),)

    return (
        ui.nav_menu(
            x.ui.tooltip("See More", "Redirects to Jupyter Notebook", id="distances_info"),
            *docs_urls
        )
    )
