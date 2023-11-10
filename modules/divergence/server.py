from __future__ import annotations

import numpy as np
import pandas

from shiny import Inputs, Outputs, Session, module, render, ui, reactive

import scipy

from config import Config
from utils import compare_dist_types

config = Config()
graph_height = config.ui_config('graph_height')
dist_defaults = config.input_config('distributions')

cont_dist = dist_defaults['continuous']
discrete_dist = dist_defaults['discrete']


@module.server
def divergence_results(input: Inputs, output: Outputs, session: Session, divergence_results: reactive.Value):
    @output
    @render.text
    def details():
        selected_divergence = input.divergences()

        details_body = []

        if divergence_results.is_set():
            for k, v in divergence_results().items():

                if k == 'element_wise_data':
                    continue

                details_body.append(f'{k}: {v}\n')

        return (
            f'\t~~~{selected_divergence} Divergence results~~~\n'
            f'{"".join(details_body)}\n'
        )


@module.server
def compute_divergences(input: Inputs, output: Outputs, session: Session, distributions: tuple,
                        diverge_results: reactive.Value):
    @reactive.Effect
    @reactive.event(input.compute)
    def compute():

        diverge_data = dict()
        current_distributions = tuple()

        dist1_data = distributions[0]
        dist2_data = distributions[1]

        dist_values = (dist1_data()['distribution_array'][1], dist2_data()['distribution_array'][1])

        if len(dist_values[0]) != len(dist_values[1]):
            diverge_data['Array length error'] = 'Selected distributions must have same length'
            diverge_results.set(diverge_data)
            return None

        for value in input.__dict__.values():
            if isinstance(value, dict):
                if 'dist1-distributions' in value.keys():
                    current_distributions = (value['dist1-distributions'](),)

                if 'dist2-distributions' in value.keys():
                    current_distributions = current_distributions + (value['dist2-distributions'](),)

        if input.divergences() == 'Discrete Bhattacharyya':

            if not compare_dist_types(current_distributions[0], current_distributions[1], 'discrete'):
                diverge_data['Distribution type error'] = ('Discrete Bhattacharyya can not be '
                                                           'calculated for continuous distributions.')
                diverge_results.set(diverge_data)
                return None

            square_roots = []
            for pi, qi in zip(dist_values[0], dist_values[1]):
                square_roots.append(np.sqrt(pi * qi))

            bhat = np.round(-np.log(sum(square_roots)), 5)

            diverge_data[f'{input.divergences()} divergence'] = bhat

            diverge_results.set(diverge_data)

        if input.divergences() == 'Discrete Hellinger':

            if not compare_dist_types(current_distributions[0], current_distributions[1], 'discrete'):
                diverge_data['Distribution type error'] = ('Discrete Hellinger can not be '
                                                           'calculated for continuous distributions.')
                diverge_results.set(diverge_data)
                return None

            squares = []

            for pi, qi in zip(dist_values[0], dist_values[1]):
                squares.append((np.sqrt(pi) - np.sqrt(qi)) ** 2)

            sq_sum = np.sum(squares)

            hellinger = np.round(sq_sum / np.sqrt(2), 5)

            diverge_data[f'{input.divergences()} divergence'] = hellinger

            diverge_results.set(diverge_data)

        if input.divergences() == 'Kullback–Leibler':

            # this is the actual Kullback-Liebler as per the scipy.special documentation
            kl = scipy.special.rel_entr(dist_values[0], dist_values[1])

            kl_by_formula = []

            for i in range(len(dist_values[0])):
                if dist_values[0][i] > 0 and dist_values[1][i] > 0:
                    kl_by_formula.append(dist_values[0][i] * np.log(dist_values[0][i] / dist_values[1][i]))
                elif dist_values[0][i] == 0 and dist_values[1][i] > 0:
                    kl_by_formula.append(0)
                else:
                    kl_by_formula.append(0)

            kl_by_formula = np.array(kl_by_formula)

            diverge_data['SciPy Kullback–Leibler avg.'] = np.round(np.average(kl), 5)
            diverge_data['Formula Based Kullback-Leibler avg.'] = np.round(np.average(kl_by_formula), 5)

            # related to the print of elementwise data
            elementwise_dict = {
                'd1_data': dist_values[0],
                'd2_data': dist_values[1],
                'kl': kl,
                'custom_kl': kl_by_formula
            }

            diverge_data['element_wise_data'] = pandas.DataFrame(elementwise_dict)

            diverge_results.set(diverge_data)

        pass


@module.server
def show_compute_extra(input: Inputs, output: Outputs, session: Session, diverge_results: reactive.Value):
    @reactive.Effect
    @reactive.event(input.show_all)
    def extra():

        if input.divergences() in ['Discrete Hellinger', 'Discrete Bhattacharyya']:
            return None

        data_frame = diverge_results()['element_wise_data']
        details_body = ['| Dist. 1 | Dist. 2 | SciPy KL | Formula KL |\n',
                        '|:--:|:--:|:--:|:--:|\n']

        for x in range(len(data_frame['d1_data'])):
            details_body.append(
                f'| {np.round(data_frame["d1_data"][x], 3)} | {np.round(data_frame["d2_data"][x], 3)} | {np.round(data_frame["kl"][x], 3)} | {np.round(data_frame["custom_kl"][x], 3)} |\n')

        details_body = ''.join(details_body)

        m = ui.modal(ui.markdown(details_body)
                     ,
                     title="Elementwise divergence table",
                     easy_close=True,
                     footer=None,
                     )

        ui.modal_show(m)
        pass
