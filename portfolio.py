# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 07:44:07 2016

@author: forex
"""

from __future__ import division
from __future__ import print_function


import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from pprint import pprint
import cPickle as pickle

from backtest import Backtest
#from optimizer import *  # Possibly just a provitional solution for the 
                            # unpickling problem.



class Portfolio():
    def __init__(self):
        self.walkforwards = []
        
    
    def calculate_stats(self):
        """
        """
        self.stats = Backtest._calculate_final_stats(self.backtestUnified)
        
    
    def get_optimization_parameters(self):
        """
        Returns a dictionary with the parameters needed for reoptimization
        of the strategy.
        """
            
    
    
    def load_walkforwards(self, method='load_pickles', path='walkforwards'):
        """
        The method specifies the function (implementation) used to load the
        pickle(s).
        """
        walkforwards = []
        namesParamDict = {}
        for name in os.listdir(path):
            wf = pickle.load(open(os.path.join(path, name)))
            walkforwards.append(wf)
            key = name.split('.')[0]
            performanceMatrix = wf[wf.keys()[0]][0]
            value = {performanceMatrix.columns.name : list(performanceMatrix.columns.values),
                     performanceMatrix.index.name : list(performanceMatrix.index.values)}
            namesParamDict.update({key: value})
            
        self.walkforwards = walkforwards
        self.optimizationParameters = namesParamDict
            
    
    def unify_backtests(self):
        """
        """
        # Merging all the backtests df
        backtest = pd.concat([x['backtest'].loc[:,'Symbol':'Profit_pips'] 
                   for x in self.walkforwards])
        backtest.sort_values('Exit_time', inplace=True)
        backtest.index = range(len(backtest.index))
        backtest = pd.concat([backtest,
                   Backtest.calculate_cumulative_stats(list(backtest.Profit_pips))], 
                   axis=1)
        self.backtestUnified = backtest



class PlotterPortfolio(object):
    def __init__(self, portfolio=None):
        self.set_portfolio(portfolio)
        
        
    def set_portfolio(self, portfolio):
        self.portfolio = portfolio
        self._p = portfolio
        
    def plot_backtest(self):
        self._p.backtestUnified.Cum_profit.plot()


    
if __name__ == "__main__":
    path='walkforwards/use'    
    
    pf = Portfolio()
    pf.load_walkforwards(path = path)
    pf.unify_backtests()
    pf.calculate_stats()
    
    pprint(pf.stats)
    
    plotter = PlotterPortfolio(pf)
    plotter.plot_backtest()
    
    
    
    
    
    
