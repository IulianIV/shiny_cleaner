from numpy import random


class Config:
    __configs = ['ui', 'input']

    __ui_setters = ['width', 'height']
    __input_setters = ['summary_operations', 'summary_fallback', 'distributions_mean_sigma',
                       'distributions_standard_deviation', 'distributions_min', 'distributions_max']
    __server_setter = []

    __ui_config = {
        'width': 500,
        'height': 1200,
        'graph_height': 550,
        'graph_height_percent': '100%'
    }
    __server_config = {}
    __input_config = {
        'summary': {
            'operations': ['min', 'max', 'mean'],
            'fallback': ['count']
        },
        'distributions': {
            'continuous': {
                'names': ['Uniform', 'Normal', 'Exponential'],
                'methods': ['Log PDF', 'Log CDF', 'Survival Function', 'Log SF']
            },
            'discrete': {
                'names': ['Binomial', 'Geometric', 'Poisson'],
                'methods': ['Log PMF', 'Log CDF', 'Survival Function', 'Log SF']
            },
            'multivariate': {

            },
            'mean': 1,
            'sd': 1.1,
            'min_obs': 10,
            'max_obs': 100,
            'events': 5,
            'scale': 5,
            'probability': 0.35,
            'trials': 10,
            'low': -1,
            'high': 0
        }
    }

    def get_dist_methods(self, distribution: str):
        continuous = self.__input_config['distributions']['continuous']
        discrete = self.__input_config['distributions']['discrete']
        multivariate = self.__input_config['distributions']['multivariate']

        if distribution in continuous['names']:
            return continuous['methods']
        elif distribution in discrete['names']:
            return discrete['methods']
        elif distribution in multivariate['names']:
            return multivariate['methods']

    @staticmethod
    def ui_config(name):
        return Config.__ui_config[name]

    @staticmethod
    def server_config(name):
        return Config.__server_config[name]

    @staticmethod
    def input_config(name):
        return Config.__input_config[name]

    # TODO reimplement this to work with the new Config structure
    #   at the moment it wll not work as expected.
    # @staticmethod
    # def set(config, name, value):
    #     if config not in Config.__configs:
    #         raise NameError(
    #             f'"{config}" is not a valid config. Valid configs: {", ".join([c for c in Config.__configs]).strip()}')
    #
    #     if config == 'ui':
    #         if name in Config.__ui_setters:
    #             Config.__ui_config[name] = value
    #         else:
    #             raise NameError('Name not accepted in set() method')
    #
    #     if config == 'input':
    #         if name in Config.__input_setters:
    #             if isinstance(Config.__input_config[name], list):
    #                 Config.__input_config[name].append(value)
    #             else:
    #                 Config.__input_config[name] = value
    #         else:
    #             raise NameError('Name not accepted in set() method')
