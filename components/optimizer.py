# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 16:03:23 2016

@author: Paul Herrera
"""

from __future__ import division
from __future__ import print_function


import copy
import datetime as dt
import gc

from functools import reduce

from .performance_visitor import SortinoTwoVariablesVisitor, SortinoThreeVariablesVisitor
from .strategies import SmaKnoxville


class Optimizer():
    """
    Parent class. 
    """
    def __init__(self, strategy, performanceVisitor = None, 
                 commission = 0, swap = (0,0), minEvents = 50,
                 orderTypeBias = None):
        self.strategy = strategy        
        self.minEvents = minEvents
        self._visitor = performanceVisitor
        self.orderTypeBias = orderTypeBias
        if orderTypeBias != None:
            self.forceOrderType = True
        else:
            self.forceOrderType = False
        self.commission = commission
        self.swap = swap

    def _acceptVisit(self):
        self._visitor.visit(self)
            
    def _check_commissions(self):
        """
        First part of the optimization algorithm.
        """
        if (self.commission > 0) | (self.swap[0] + self.swap[1] != 0):
            commission = True
            swap = True
        else:
            commission = False
            swap = False

        return commission, swap
    
    def optimize(self):
        """
        """
        raise NotImplementedError('To be implemented')
        
                                                         
                                                 
class TwoVariablesOpt(Optimizer):
    """
    Implements the optimization method for two variable strategies.
    """
    
    def optimize(self, fixedParameters, variableParameters, eventLookback = 1,
                 plot=False):
        """
        Main method.
        """
        # Checking the presence of commissions
        commission, swap = self._check_commissions()

        # Initializing some variables.               
        self.optimization = {}
        optNum = reduce(lambda x,y: x*y, 
                        [len(x) for x in variableParameters.values()])
        i=1

        # Optimization loop
        key1, key2 = tuple(variableParameters.keys())

        for var1 in list(variableParameters.values())[0]:

            for var2 in list(variableParameters.values())[1]:
                print('-----------------------------------------------------------')
                print('Optimization {} of {}'.format(i, optNum))
                print('Processing variables: {}, {}'.format(var1, var2))
                
                #Stablishing parameters for optimization.
                parameters = copy.copy(fixedParameters)
                parameters.update({x:y for (x,y) in 
                                  zip(variableParameters.keys(),[var1,var2])})
                
                # Initializing the strategy and its event studies.
                strategy = self.strategy.initialize_strategy(commission=self.commission,
                           swap=self.swap, **parameters)
                strategy.generate_optimized_event_study(commission=commission, 
                                                     swap=swap, plot=plot,
                                                     lookback = eventLookback)
                                                     
                # Storing the strategy.
                del strategy.base
                gc.collect()
                self.optimization['{}_{},{}_{}'.format(key1,var1,
                                  key2,var2)] = strategy
                
                i += 1

        # Accept a visit to generate performance matrix.
        self._acceptVisit()        
            

class ThreeVariablesOpt(Optimizer):
    """
    Implements the optimization method for three variable strategies.
    """    
    def optimize(self, fixedParameters, variableParameters, eventLookback = 1,
                 plot=False):
        """
        Main method.
        """
        # Checking the presence of commissions
        commission, swap = self._check_commissions()

        # Initializing some variables.               
        self.optimization = {}
        optNum = reduce(lambda x,y: x*y, 
                        [len(x) for x in variableParameters.values()])
        i=1

        # Optimization loop
        key1, key2, key3 = tuple(variableParameters.keys())
        
        for var1 in variableParameters.values()[0]:

            for var2 in variableParameters.values()[1]:

                for var3 in variableParameters.values()[2]:
                    
                    print('---------------------------------------------------')
                    print('Optimization {} of {}'.format(i, optNum))
                    print('Processing variables: {}, {}, {}'.format(var1, var2, 
                                                                    var3))
                    
                    #Stablishing parameters for optimization.
                    parameters = copy.copy(fixedParameters)
                    parameters.update({x:y for (x,y) in 
                                      zip(variableParameters.keys(),[var1,var2,
                                                                     var3])})
                    
                    # Initializing the strategy and its event studies.
                    strategy = self.strategy.initialize_strategy(commission=self.commission,
                               swap=self.swap, **parameters)
                    strategy.generate_optimized_event_study(commission=commission, 
                                                         swap=swap, plot=plot,
                                                         lookback = eventLookback)
                                                         
                    # Storing the strategy.  
                    del strategy.base
                    gc.collect()
                    self.optimization['{}_{},{}_{},{}_{}'.format(key1,var1,
                                      key2,var2,key3,var3)] = strategy
                    
                    i += 1

        # Accept a visit to generate performance matrix.
        self._acceptVisit()        
                    
                    

if __name__ == "__main__":
    
    opt = TwoVariablesOpt(SmaKnoxville(startDate = dt.date(2013,12,8), 
                           endDate = dt.date(2016,12,30)),
                           performanceVisitor = SortinoTwoVariablesVisitor(),
                           commission=20)
    
    fixed = {
             'symbol' : 'NZDUSD',
             'timeframe' : 'M5',
             'SMAmethod' : 'above',
             'KDmethod' : 'below'}
             
    variables = {
#                 'symbol' : ['EURUSD', 'AUDUSD'],                 
                 'SMAlookback' : [1000, 1250, 1500, 1750, 2000],
                 'KDlookback' : [30]
                 }
                 
    opt.optimize(fixed, variables, eventLookback=20, plot=True)
    print(opt.performanceMatrix)
    
    