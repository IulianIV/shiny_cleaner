import math

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
                'standard': ['Observations', 'CDF', 'PDF'],
                'names': ['Uniform', 'Normal', 'Exponential'],
                'methods': ['Log PDF', 'Log CDF', 'SF', 'Log SF'],
                'extra_methods': ['PPF', 'ISF']
            },
            'discrete': {
                'standard': ['Observations', 'CDF', 'PMF'],
                'names': ['Binomial', 'Geometric', 'Poisson'],
                'methods': ['Log PMF', 'Log CDF', 'SF', 'Log SF'],
                'extra_methods': ['PPF', 'ISF']
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
            'low': 0,
            'high': 1,
            'confidence': 0.1,
            'lb': 10,
            'ub': 100
        }
    }

    def get_dist_methods(self, distribution: str, extra: bool = False):
        continuous = self.__input_config['distributions']['continuous']
        discrete = self.__input_config['distributions']['discrete']
        multivariate = self.__input_config['distributions']['multivariate']

        if distribution in continuous['names']:
            if extra:
                return continuous['extra_methods']
            return continuous['methods']
        elif distribution in discrete['names']:
            if extra:
                return discrete['extra_methods']
            return discrete['methods']
        elif distribution in multivariate['names']:
            if extra:
                return multivariate['extra_methods']
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
