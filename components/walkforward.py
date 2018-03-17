# -*- coding: utf-8 -*-
"""
Created on Thu Oct 06 10:36:53 2016

@author: Paúl Herrera
"""

from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import datetime as dt
from pprint import pprint
import sys
import gc

from .optimizer import TwoVariablesOpt, ThreeVariablesOpt
from .backtest import Backtest
from .optimization_analyzer import TwoVariablesOptimizationAnalyzer
from .performance_visitor import SortinoTwoVariablesVisitor, SortinoThreeVariablesVisitor
from .strategies import SmaKnoxville, KDDouble, RSI_SDC
from .walkforward_visitor import LoopVisitor, WalkforwardVisitor
from .optimization_reporter import OptimizationReporterBridge, MatrixScreenReporter, BacktestReporterBridge
from .walkforward_persistance_builder import WalkforwardPersistanceBuilder
from .scripts.custom_functions import notification


class Walkforward(object):
    """
    """
    def __init__(self, strategy, optimizer, backtester, 
                 performanceVisitor, optimizationVisitor, backtestVisitor,
                 optimizationAnalyzer, startDate, endDate, 
                 numOutOfSamples=5, ratio = 3,
                 commission=0, swap=(0,0)):
        self.startDate = startDate
        self.endDate = endDate
        self._strategy = strategy
        self._optimizer = optimizer
        self._backtester = backtester
        self._performanceVisitor = performanceVisitor
        self._optimizationVisitor = optimizationVisitor
        self._backtestVisitor = backtestVisitor
        self._optimizationAnalyzer = optimizationAnalyzer
        self.commission = commission
        self.swap = swap        
        
        self._calculateCalendar(numOutOfSamples, ratio)
        print(self.calendar)


    def accept_visit_backtest(self):
        self._backtestVisitor.visit(self.backtests)
        

    def accept_visit_optimization(self):
        self._optimizationVisitor.visit(self.optimizations)


    def _backtest(self, fixedParameters, eventLookback=10, 
                  ordinal=None, threshold=0):
        raise NotImplementedError('To be implemented')


    def _calculateCalendar(self, numOutOfSamples=5, ratio=3):
        """
        """
        #Dividing the data for the Walkforward
        days = (self.endDate - self.startDate).days
        divisions = numOutOfSamples + ratio
        oosample = days / divisions
        insample = oosample * ratio

        # Creating the dates.
        InSampleStarts = [self.startDate + dt.timedelta(oosample * x)
                          for x in range(numOutOfSamples)]
        OutSampleStarts = [(self.startDate + dt.timedelta(insample)) + \
                           dt.timedelta(oosample * x) 
                           for x in range(numOutOfSamples)]
        OutSampleEnds = [x + dt.timedelta(oosample) for x in OutSampleStarts]
        
        #Creating a DataFrame with the InSample and OutofSample boundaries    
        walkf = pd.DataFrame()
            
        walkf['InSampleStarts'] = InSampleStarts
        walkf['OutSampleStarts'] = OutSampleStarts
        walkf['OutSampleEnds'] = OutSampleEnds
        
        self.calendar = walkf
    

    def _optimize(self, fixedParameters, variableParameters, eventLookback=10,
                  preoptimize=False, plot=False):    
        self.optimizations = []
        i = 1
        first = True
        for startDate, endDate in [x for x in zip(self.calendar.InSampleStarts,
                                                  self.calendar.OutSampleStarts)]:
            print('\n{}'.format('-'*60))
            print('Insample {}: {} - {}'.format(i, startDate, endDate))
            
            optimization = self._optimizer(self._strategy(startDate, endDate),
                                           self._performanceVisitor(),
                                           commission = self.commission,
                                           swap = self.swap)
            optimization.optimize(fixedParameters, variableParameters, 
                                  eventLookback, plot)
            
            # Storing.
            self.optimizations.append(optimization)
            
            # Uses the preoptimize method to get a preview of the optimization.
            if preoptimize and first:
                self._preoptimize(optimization)
                first = False
                if self.__class__ == Preoptimize: # Ends the optimization in
                    break                         # case of a preoptimization. (This is a hack :( )
            
            gc.collect()
            i += 1


    def _preoptimize(self, optimization):
        """
        Checks if there is too much correlation int he events and lets the 
        client decide if the optimization must go on.
        """
        # This benchmark must be under 1.
        benchmark = round(np.mean([x.eventsRelevantData['Period with best Sortino ratio']
                                   / x.signalsSpan['1st quartile'] for x in 
                                   optimization.optimization.values()]), 2)
        minEvents = min([x.eventsNumber for x in optimization.optimization.values()])
        maxEvents = max([x.eventsNumber for x in optimization.optimization.values()])

        # Play notification.
        notification()

        # Print report.
        self.accept_visit_optimization()
        print('-' * 60)
        print('Optimization correlation fitness is: {}'.format(benchmark))
        print('Minimum amount of events: {}'.format(minEvents))
        print('Maximum amount of events: {}'.format(maxEvents))
        
        if self.__class__ != Preoptimize:
            cont = input('Continue with the optimization? (y/n): ')
            if cont == 'n': sys.exit('')
        

    def walkforward(self):
        """
        Main method.
        """        
        raise NotImplementedError('To be implemented')



class ElectionTwoVariableWalkforward(Walkforward):
    """
    Walkforwards a two-variable optimization.
    Lets the client choose the criteria for choosing the parameters 
    for the backtest.
    """
    def _backtest(self, fixedParameters, eventLookback, 
                  ordinal=None, threshold=0):
        self.backtests = []
        
        # Initializes the ordinal and the backtest count (btCount) according
        # the client election on the type of backtest
        if ordinal:
            ordinals = range(1, ordinal + 1)
            btCount = [ordinal] * len(self.optimizations[0].optimization)
        else:
            btCount = [len(x.performanceMatrix.values[x.performanceMatrix.values
                       >= threshold]) for x in self.optimizations]
            ordinals = range(1, max(btCount) + 1)
            
        # The first loop backtests the best optimization, then the second best
        # and so on.
        for ordinal in ordinals:
            parameters = []
            
            # This loop appends to the parameter list the optimizations
            # that are above a threshold (if there is one).
            for optimization in self.optimizations:
                parameter = self._optimizationAnalyzer.analyze(optimization, 
                                                               ordinal)
                if optimization.performanceMatrix.loc[parameter[0].values()[0], 
                                        parameter[0].values()[1]] >= threshold:
                    parameters.append(parameter)
                else: 
                    parameters.append(None)
                 
            # Backtests.
            i = 1
            for startDate, endDate, parameter in [x for x 
                in zip(self.calendar.OutSampleStarts, 
                       self.calendar.OutSampleEnds,
                       parameters)]:
                print('\n{}'.format('-'*30))
                print('Backtest {}-{}: {} - {}'.format(ordinal, i, 
                                                  startDate, endDate))
                print(parameter)
                                                  
#                try: # Initialize backtest, generate backtest and stats. 
                bt = self._backtester(self._strategy(startDate, endDate),
                                      commission=self.commission)
                
                bt.generate_backtest(dict(fixedParameters, **parameter[0]), 
                                     eventLookback, 
                                     parameter[1], parameter[2], 
                                     divide=btCount[i-1], plot=False)
                bt.generate_stats(printIt = False)

                self.backtests.append(bt)
#                except TypeError: pass
                    
                i += 1


    def walkforward(self, fixedParameters, variableParameters, 
                    eventLookback=10, preoptimize=False, 
                    clientChoice=False, plot=False):
        """
        Main method.
        """        
        self._optimize(fixedParameters, variableParameters, eventLookback,
                       preoptimize=preoptimize, plot=plot)
        
        print('\n' + '-' * 60)
        print('Optimizations report')
        self.accept_visit_optimization()
        
        keepGoing = True
        while keepGoing:
            # Backtesting according to client decision.
            if clientChoice: 
                notification()
                
            chosen = str(raw_input('Choose method for backtesting ("best" or "threshold"): ')) \
                     if clientChoice else 'b'
            
            if chosen[0] == 'b':
                number = int(input('Number of best optimizations to backtest: ')) \
                         if clientChoice else 1
                self._backtest(fixedParameters, eventLookback, ordinal=number)
                    
            elif chosen[0] == 't':
                threshold = float(raw_input('The minimum Sortino Ratio to backtest must be: '))
                self._backtest(fixedParameters, eventLookback,  
                               ordinal = len(self.optimizations[0].optimization),
                               threshold=threshold)
                            
            # Merge backtests.
            self.backtestFinal = self.backtests[0].merge_backtests(self.backtests)
            
            # Show report            
            self.backtestFinal.plot_backtest()
            pprint(self.backtestFinal.stats)
            
            notification()
            if clientChoice:
                keepGoing = True if raw_input('Backtest again? (y/[n]): ').lower() \
                                    == 'y' else False


class Preoptimize(Walkforward):
    """
    Only preoptimizes.
    """
    def walkforward(self, fixedParameters, variableParameters, 
                    eventLookback=10, preoptimize=True, plot=False):
        self._optimize(fixedParameters, variableParameters, eventLookback,
                       preoptimize=True, plot=plot)



class SimpleWalkforward(Walkforward):
    """
    Walkforwards always choosing the best parameters of each optimization 
    for the backtest.
    """
    def _backtest(self, fixedParameters, eventLookback, ordinal=1):
        """
        Backtests.
        """
        self.backtests = []
        # Grabs best parameters using the optimizationAnalyzer.
        parameters = [self._optimizationAnalyzer.analyze(x, 1) for x
                      in self.optimizations]
                         
        # Backtests loop.
        i = 1
        for startDate, endDate, parameter \
            in [x for x in zip(self.calendar.OutSampleStarts, 
                               self.calendar.OutSampleEnds,
                               parameters)]:
            print('\n{}\n'.format('-'*30))
            print('Backtest {}: {} - {}'.format(i, startDate, endDate))
            print(parameter)

            # Instantiate, generate and statistify backtest.                                              
            bt = self._backtester(self._strategy(startDate, endDate),
                                  commission=self.commission)
            bt.generate_backtest(dict(fixedParameters, **parameter[0]), 
                                 eventLookback, parameter[1], parameter[2], 
                                 plot=False)
            bt.generate_stats(printIt = False)

            self.backtests.append(bt)
            i += 1


    def walkforward(self, fixedParameters, variableParameters, 
                    eventLookback=10, preoptimize=False, plot=False):
        """
        Main method.
        """        
        self._optimize(fixedParameters, variableParameters, eventLookback,
                       preoptimize=preoptimize, plot=plot)
        
        
        print('\n' + '-' * 60)
        print('Optimizations report')
        self.accept_visit_optimization()
        
        # Backtesting.
        self._backtest(fixedParameters, eventLookback, ordinal=1)
                        
        # Merge backtests.
        self.backtestFinal = self.backtests[0].merge_backtests(self.backtests)
        
        # Show report            
        self.backtestFinal.plot_backtest()
        pprint(self.backtestFinal.stats)
        
        notification()
        
    
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
        'BTCUSD', 
#        'GBPUSD', 
#        'NZDUSD', 
#        'USDCAD', 
#        'USDCHF', 
#        'USDJPY', 
    ]
    
    path = 'walkforwards'
    
    for s in symbols:
        print('\nWORKING ON {}'.format(s))        
        
        startDate = dt.date(2015,1,31)
        endDate = dt.date(2017,8,26)
        eventLookback = 15
        outOfSamples = 5
        fixed = {
                 'symbol' : s,
                 'timeframe' : 'M15',
                 'RSIlookback' : 14,
                 'RSImethod' : 'above',
                 }
                 
                 
        variables = {
                    'RSIlevel' : [39,36,33,30],
                     'SDCrepetition' : [-3,-4,-5],
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
        
        WalkforwardPersistanceBuilder(wf).save(path, name)
    
    notification()


    """
    Walkforwards loader and merger. Procedural by now.
    """
#    from os import walk
#    
#    path = 'walkforwards/use'
#    files = list(walk('walkforwards/use'))[0][2]
#    
#    backtests = []
#    for f in files:
#        wf = WalkforwardPersistanceBuilder()
#        wf.load(path,f)
#        backtests.append(wf.backtest)
#    backtests = pd.concat(backtests).loc[:,:'Profit_pips']
#    backtests.sort_values('Exit_time', inplace=True)
#    backtests.index = range(len(backtests))
#    backtests = pd.concat([backtests, 
#                           Backtest.calculate_cumulative_stats(
#                               backtests.Profit_pips)], axis=1)    
#
#    backtest = backtests
#    stats = []
#    stats.append(('Number of trades',len(backtest)))
#    stats.append(('Total profits (pips)', 
#                  round(backtest.Cum_profit.iloc[-1],1)))
#    stats.append(('Mean',round(backtest.Cum_mean.iloc[-1], 1)))
#    stats.append(('Std',round(backtest.Cum_std.iloc[-1], 2)))
#    stats.append(('Sharpe',round(backtest.Cum_sharpe.iloc[-1], 3)))
#    stats.append(('Delta_sharpe (30)',
#                  round(abs(np.diff(backtest.Cum_sharpe))[-30:].mean(), 4)))
#    stats.append(('SQN',round(backtest.Cum_sqn.iloc[-1], 2)))
#    stats.append(('Delta_sqn (30)',
#                  round(abs(np.diff(backtest.Cum_sqn))[-30:].mean(), 3)))
#    stats.append(('Max Drawdown',
#                  round(max(backtest.Drawdown), 1)))
#    stats.append(('Calmar', round(backtest.Cum_profit.iloc[-1]
#                                  / max(backtest.Drawdown), 1)))
#    stats.append(('Winner percentage',
#                  round(len(backtest.Profit_pips[backtest.Profit_pips > 0])\
#                        / len(backtest) * 100, 1)))
#    stats.append(('Average profit', 
#                   round(backtest.Profit_pips[backtest.Profit_pips > 0].mean(),1)))                
#    stats.append(('Average loss', 
#                   round(backtest.Profit_pips[backtest.Profit_pips < 0].mean(),1)))
#    stats.append(('PL Ratio',
#                  round(-backtest.Profit_pips[backtest.Profit_pips > 0].mean() / \
#                  backtest.Profit_pips[backtest.Profit_pips < 0].mean(),1)))
#    stats.append(('Max profit', round(max(backtest.Profit_pips),1)))
#    stats.append(('Max loss', round(min(backtest.Profit_pips),1)))
#  
#    backtest.Cum_profit.plot()
#    pprint(stats)
#    
    
