from shiny import ui


class Config:
    __configs = ['ui', 'input']

    __ui_setters = ['width', 'height']
    __input_setters = ['summary_operations', 'summary_fallback', 'distributions_mean_sigma',
                       'distributions_standard_deviation', 'distributions_min', 'distributions_max']

    __server_setter = []

    __ui_config = {
        'width': 500,
        'height': 1200,
        'graph_height': 500,
        'graph_height_percent': '100%',
        'tooltip_q': ui.HTML(
            '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi '
            'bi-question-circle-fill mb-1" viewBox="0 0 16 16"><path d="M16 8A8 8 0 1 1 0 8a8 8 0 0 1 16 0zM5.496 '
            '6.033h.825c.138 0 .248-.113.266-.25.09-.656.54-1.134 1.342-1.134.686 0 1.314.343 1.314 1.168 0 '
            '.635-.374.927-.965 1.371-.673.489-1.206 1.06-1.168 1.987l.003.217a.25.25 0 0 0 .25.246h.811a.25.25 0 0 0 '
            '.25-.25v-.105c0-.718.273-.927 1.01-1.486.609-.463 1.244-.977 1.244-2.056 '
            '0-1.511-1.276-2.241-2.673-2.241-1.267 0-2.655.59-2.75 2.286a.237.237 0 0 0 .241.247zm2.325 6.443c.61 0 '
            '1.029-.394 1.029-.927 0-.552-.42-.94-1.029-.94-.584 0-1.009.388-1.009.94 0 .533.425.927 1.01.927z"/></svg>'
        )
    }
    __server_config = {}
    __input_config = {
        'summary': {
            'operations': ['min', 'max', 'mean'],
            'fallback': ['count']
        },
        'distributions': {
            'continuous': {
                'standard': ['Observations', 'PDF', 'CDF'],
                'names': ['Uniform', 'Normal', 'Exponential', 'Cauchy'],
                'methods': ['Log PDF', 'Log CDF', 'SF', 'Log SF'],
                'extra_methods': ['PPF', 'ISF']
            },
            'discrete': {
                'standard': ['Observations', 'PMF', 'CDF'],
                'names': ['Binomial', 'Geometric', 'Poisson'],
                'methods': ['Log PMF', 'Log CDF', 'SF', 'Log SF'],
                'extra_methods': ['PPF', 'ISF']
            },
            'multivariate': {
            },
            'mean': 1,
            'sd': 1.1,
            'min_obs': 0,
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
        },
        'statistical_testing':{
            'tests': ['t-test', 'z-test', 'Wilcoxon', 'ANOVA']
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
