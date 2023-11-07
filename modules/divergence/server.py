from __future__ import annotations

import numpy
import numpy as np
import pandas
from shiny import Inputs, Outputs, Session, module, render, ui, reactive
import scipy

from config import Config

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
        # elementwise_df = None

        # TODO implement the elementwise print of data

        # if divergence_results.is_set():
        #     # details_body = ''.join([f'{title}: {value}\n' for title, value in divergence_results().items()[-1]])
        #     for k, v in divergence_results().items():
        #         details_body.append(f'{k}: {v}\n')
        #
        #         if k == 'element_wise_data':
        #             elementwise_df = v
        #             # details_body.append('Dist. 1 Data\tDist. 2 Data\tSciPy KL\tFormula KL\n')
        #             # for x in range(len(v['d1_data'])):
        #             #     details_body.append(
        #             #         f'{v["d1_data"][x]}\t{v["d2_data"][x]}\t{v["kl"][x]}\t{v["custom_kl"][x]}\n')

        return (
            f'\t~~~{selected_divergence} Divergence results~~~\n'
            f'{"".join(details_body)}\n'
            # f'{elementwise_df}'
        )


@module.server
def compute_divergences(input: Inputs, output: Outputs, session: Session, distributions: tuple,
                        diverge_results: reactive.Value):
    @reactive.Effect
    @reactive.event(input.compute)
    def compute():

        current_distributions = tuple()

        diverge_data = dict()

        for value in input.__dict__.values():
            if isinstance(value, dict):
                if 'dist1-distributions' in value.keys():
                    current_distributions = (value['dist1-distributions'](),)

                if 'dist2-distributions' in value.keys():
                    current_distributions = current_distributions + (value['dist2-distributions'](),)

        dist1_data = distributions[0]
        dist2_data = distributions[1]

        dist_values = (dist1_data()['distribution_array'][1], dist2_data()['distribution_array'][1])

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

            kl_by_formula = numpy.array(kl_by_formula)

            diverge_data['SciPy Kullback–Leibler avg.'] = np.average(kl)
            diverge_data['Formula Based Kulback-Leibler avg.'] = np.average(kl_by_formula)

            # related to the print of elementwise data
            if input.show_all():
                elementwise_dict = {
                    'd1_data': dist_values[0],
                    'd2_data': dist_values[1],
                    'kl': kl,
                    'custom_kl': kl_by_formula
                }

                diverge_data['element_wise_data'] = pandas.DataFrame(elementwise_dict)

                print(diverge_data['element_wise_data'])

            diverge_results.set(diverge_data)

        pass
