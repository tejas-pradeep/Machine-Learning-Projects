import pandas as pd
import statsmodels.api as sm

class FeatureReduction(object):
    def __init__(self):
        pass

    @staticmethod
    def forward_selection(data, target, significance_level=0.05): # 9 pts
        '''
        Implement forward selection using the steps provided in the notebook.
        You can use sm.OLS for your regression model.
        Do not forget to add a bias to your regression model. A function that may help you is the 'sm.add_constants' function.
        
        Args:
            data: data frame that contains the feature matrix
            target: target feature to search to generate significant features
            significance_level: the probability of the event occuring by chance
        Return:
            forward_list: list containing significant features (in order of selection)

        '''
        raise NotImplementedError

    @staticmethod
    def backward_elimination(data, target, significance_level = 0.05): # 9 pts
        '''
        Implement backward selection using the steps provided in the notebook.
        You can use sm.OLS for your regression model.
        Do not forget to add a bias to your regression model. A function that may help you is the 'sm.add_constants' function.

        Args:
            data: data frame that contains the feature matrix
            target: target feature to search to generate significant features
            significance_level: the probability of the event occuring by chance
        Return:
            backward_list: list containing significant features
        '''
        raise NotImplementedError
