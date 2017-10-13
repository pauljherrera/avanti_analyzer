# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 09:34:07 2016

@author: forex
"""

from __future__ import division
from __future__ import print_function



from .scripts.custom_functions import  max_argmax_df, min_argmin_df
from .scripts.custom_functions import max_argmax_panel, min_argmin_panel


class optimizationAnalyzer(object):
    """
    It's main method 'analyze()' analyzes an optimization object and returns
    the best parameters.
    """
    def analyze(self):
        """
        Main method.
        """
        raise NotImplementedError('To be implemented')
        
        
class TwoVariablesOptimizationAnalyzer(optimizationAnalyzer):
    
    def analyze(self, optimization, ordinal=1):
        # Searching best parameters. 
        value, (var1,var2) = max_argmax_df(optimization.performanceMatrix,
                                           ordinal)
        key1 = optimization.performanceMatrix.index.names[0]
        key2 = optimization.performanceMatrix.columns.names[0]
        
        # Searching order type and exit period.
        try:
            orderType = optimization.optimization['{}_{},{}_{}'.format(key1, 
                                                  var1, key2, var2)].orderType
            exitPeriod = optimization\
                         .optimization['{}_{},{}_{}'.format(key1, var1, key2, var2)]\
                         .eventsRelevantData['Period with best Sortino ratio']

        except KeyError:
            orderType = optimization.optimization['{}_{},{}_{}'.format(key1, 
                        int(var1), key2, int(var2))].orderType

            exitPeriod = optimization\
                         .optimization['{}_{},{}_{}'.format(key1, int(var1), 
                                                            key2, int(var2))]\
                         .eventsRelevantData['Period with best Sortino ratio']
        
        return {key1 : var1, key2 : var2}, orderType, exitPeriod
        
        
class ThreeVariablesOptimizationAnalyzer(optimizationAnalyzer):
    """
    """
        



