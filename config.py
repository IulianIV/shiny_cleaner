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
    __server_config = {
        'distribution_types': random.__all__
    }
    __input_config = {
        'summary_operations': ['min', 'max', 'mean'],
        'summary_fallback': ['count'],
        'distributions': ['Uniform', 'Binomial', 'Poisson', 'Normal', 'Exponential',
                          'Geometric'],
        'distributions_mean_mu': 1,
        'distributions_standard_deviation_sigma': 1.1,
        'distributions_min': 10,
        'distributions_max': 100,
        'distributions_events': 5,
        'distributions_scale': 5,
        'distributions_probability': 0.35,
        'distributions_trials': 10,
        'distributions_low': -1,
        'distributions_high': 0,
        'distributions_prop': {
            'Uniform': ['Log PDF', 'Log CDF', 'Survival Function', 'Log SF'],
            'Binomial': ['Log PMF', 'Log CDF', 'Survival Function', 'Log SF'],
            'Poisson': ['Log PMF', 'Log CDF', 'Survival Function', 'Log SF'],
            'Normal': ['Log PDF', 'Log CDF', 'Survival Function', 'Log SF'],
            'Exponential': ['Log PDF', 'Log CDF', 'Survival Function', 'Log SF'],
            'Geometric': ['Log PMF', 'Log CDF', 'Survival Function', 'Log SF']
        }
    }

    @staticmethod
    def ui_config(name):
        return Config.__ui_config[name]

    @staticmethod
    def server_config(name):
        return Config.__server_config[name]

    @staticmethod
    def input_config(name):
        return Config.__input_config[name]

    @staticmethod
    def set(config, name, value):
        if config not in Config.__configs:
            raise NameError(
                f'"{config}" is not a valid config. Valid configs: {", ".join([c for c in Config.__configs]).strip()}')

        if config == 'ui':
            if name in Config.__ui_setters:
                Config.__ui_config[name] = value
            else:
                raise NameError('Name not accepted in set() method')

        if config == 'input':
            if name in Config.__input_setters:
                if isinstance(Config.__input_config[name], list):
                    Config.__input_config[name].append(value)
                else:
                    Config.__input_config[name] = value
            else:
                raise NameError('Name not accepted in set() method')
