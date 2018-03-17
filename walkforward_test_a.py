# -*- coding: utf-8 -*-
"""
Created on Thu Oct 06 10:36:53 2016

@author: Paúl Herrera
"""

import datetime as dt

from components.optimizer import TwoVariablesOpt, ThreeVariablesOpt
from components.strategies import SmaKnoxville, KDDouble, RSI_SDC
from components.walkforward import SimpleWalkforward, Preoptimize
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
        
        startDate = dt.date(2006,1,2)
        endDate = dt.date(2018,3,9)
        eventLookback = 15
        outOfSamples = 6
        fixed = {
                 'symbol' : s,
                 'timeframe' : 'M15',
                 #bullish
                 #'KDmethod' : 'above',
                 #'SMAmethod' : 'below',
                 #bearish
                 'KDmethod' : 'below',
                 'SMAmethod' : 'above',
                 
                 }
                 
        """variables = {
                    'SMAlookback' : [3000,2750,2500,2250,2000,1750,1500,1250,1000],
                    'KDlookback' : [15,30,45,60,75,90,105,120],
                    }"""
        variables = {
                'SMAlookback' : [2250],
                    'KDlookback' : [30]                
                }
        
        """variables = {
                    'SMAlookback' : [1750,1500,1250,1000,750,500,250],
                    'KDlookback' : [60,90,120,150,180]
                    }"""
                

    
        wf = SimpleWalkforward(SmaKnoxville, TwoVariablesOpt, Backtest,
        #wf = Preoptimize(SmaKnoxville, TwoVariablesOpt, Backtest,
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
                + '_KD{}SMA{}'.format(fixed['KDmethod'], fixed['SMAmethod']) \
                + '_{}oos_lb{}.pkl'.format(outOfSamples, eventLookback)
        print(name)
        WalkforwardPersistanceBuilder(wf).save(path, name)
        
        