# -*- coding: utf-8 -*-
"""
Created on Thu Oct 06 10:36:53 2016

@author: Paúl Herrera
"""

import datetime as dt


from components.optimizer import TwoVariablesOpt, ThreeVariablesOpt
from components.strategies import SmaKnoxville, KDDouble, RSI_SDC
from components.walkforward import SimpleWalkforward
from components.backtest import Backtest
from components.walkforward_visitor import LoopVisitor, WalkforwardVisitor
from components.optimization_analyzer import TwoVariablesOptimizationAnalyzer
from components.performance_visitor import SortinoTwoVariablesVisitor
from components.walkforward_persistance_builder import WalkforwardPersistanceBuilder
from components.optimization_reporter import OptimizationReporterBridge
from components.optimization_reporter import MatrixScreenReporter
from components.optimization_reporter import BacktestReporterBridge


if __name__ == "__main__":
    """
    Walkforward and results persistance
    """
    symbols = [
#        'AUDCAD', 
#        'AUDJPY', 
#        'AUDUSD', 
#        'EURAUD', 
#        'EURCAD', 
#        'EURGBP', 
#        'EURJPY', 
#        'BTCUSD', 
#        'GBPUSD', 
#        'NZDUSD', 
#        'USDCAD', 
#        'USDCHF', 
#        'USDJPY',
         'EURUSD'
    ]
    
    path = 'walkforwards'
    
    for s in symbols:
        print('\nWORKING ON {}'.format(s))        
        
        startDate = dt.date(2010,1,31)
        endDate = dt.date(2016,8,26)
        eventLookback = 15
        outOfSamples = 5
        fixed = {
                 'symbol' : s,
                 'timeframe' : 'D1',
                 'RSIlookback' : 14,
                 'RSImethod' : 'above',
                 }
                 
        variables = {
                    'RSIlevel' : [39,36],
                     'SDCrepetition' : [-3,-4],
                    }
        

    
        wf = SimpleWalkforward(RSI_SDC, TwoVariablesOpt, Backtest,
                         SortinoTwoVariablesVisitor, 
                         LoopVisitor(OptimizationReporterBridge(MatrixScreenReporter())),
                         WalkforwardVisitor(BacktestReporterBridge()),
                         TwoVariablesOptimizationAnalyzer(),
                         startDate, endDate, numOutOfSamples=outOfSamples,
                         commission=50)
        
        wf.walkforward(fixed, variables, eventLookback=eventLookback, 
                       preoptimize=False, plot=True)

        name = '{}'.format(fixed['symbol']) \
                + '_{}'.format(fixed['timeframe']) \
                + '_RSI{}{}'.format(fixed['RSIlookback'], fixed['RSImethod']) \
                + '_SDC' \
                + '_{}oos_lb{}.pkl'.format(outOfSamples, eventLookback)
        print(name)
        WalkforwardPersistanceBuilder(wf).save(path, name)
        
        