# -*- coding: utf-8 -*-
"""
Created on Wed Oct 05 18:11:23 2016

@author: forex
"""

from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from pprint import pprint
from copy import copy
import datetime as dt


from .strategies import SmaKnoxville


class Backtest(object):
    """
    """
    def __init__(self, strategy, commission = 0, swap = (0,0)):
        self.strategy = strategy
        self.commission = commission
        self.swap = swap    
    
    def apply_money_management(self, method='naive_fixed_loss', pipsLoss=100,
                               loss=100):
        """
        Applies a money management technique to the backtest.
        """
        if method == 'naive_fixed_loss':
            mmBacktest = copy(self.backtest.loc[:,:'Profit_pips'])
            mmBacktest ['Profit'] = np.round(mmBacktest.Profit_pips * \
                                             loss / pipsLoss,2)
        
        # Cumulative stats.
        mmBacktest = pd.concat([mmBacktest, self.calculate_cumulative_stats(mmBacktest.Profit)], axis=1)
        
        self.mmBacktest = mmBacktest
        
        
    @staticmethod
    def calculate_cumulative_stats(profitsColumn):
        """
        Calculate the cumulative stats based on a profit Series.
        """
        # Initializing DataFrame and converting the column in a pd.Series.
        df = pd.DataFrame()
        profitsColumn = pd.Series(profitsColumn)
        
        # Calculating the cumulatinve stats.
        df['Cum_profit'] = np.cumsum(profitsColumn)
        df['Drawdown'] = np.maximum.\
                              accumulate(df['Cum_profit']) - \
                              df['Cum_profit']
        df['Cum_mean'] = profitsColumn.expanding().mean()
        df['Cum_std'] = profitsColumn.expanding().std()
        df['Cum_sharpe'] = df.Cum_mean / df.Cum_std
        df['Cum_sqn'] = df.Cum_sharpe * np.sqrt(df.index)
                
        return df

    
    @staticmethod
    def _calculate_final_stats(df):
        backtest = df
        stats = []
        
        stats.append(('Number of trades',len(backtest)))
        stats.append(('Total profits (pips)', 
                      round(backtest.Cum_profit.iloc[-1],1)))
        stats.append(('Mean',round(backtest.Cum_mean.iloc[-1], 1)))
        stats.append(('Std',round(backtest.Cum_std.iloc[-1], 2)))
        stats.append(('Sharpe',round(backtest.Cum_sharpe.iloc[-1], 3)))
        stats.append(('Delta_sharpe (30)',
                      round(abs(np.diff(backtest.Cum_sharpe))[-30:].mean(), 4)))
        stats.append(('SQN',round(backtest.Cum_sqn.iloc[-1], 2)))
        stats.append(('Delta_sqn (30)',
                      round(abs(np.diff(backtest.Cum_sqn))[-30:].mean(), 3)))
        stats.append(('Max Drawdown',
                      round(max(backtest.Drawdown), 1)))
        stats.append(('Calmar', round(backtest.Cum_profit.iloc[-1]
                                      / max(backtest.Drawdown), 1)))
        stats.append(('Winner percentage',
                      round(len(backtest.Profit_pips[backtest.Profit_pips > 0])\
                            / len(backtest) * 100, 1)))
        stats.append(('Average profit', 
                       round(backtest.Profit_pips[backtest.Profit_pips > 0].mean(),1)))                
        stats.append(('Average loss', 
                       round(backtest.Profit_pips[backtest.Profit_pips < 0].mean(),1)))
        stats.append(('PL Ratio',
                      round(-backtest.Profit_pips[backtest.Profit_pips > 0].mean() / \
                      backtest.Profit_pips[backtest.Profit_pips < 0].mean(),1)))
        stats.append(('Max profit', round(max(backtest.Profit_pips),1)))
        stats.append(('Max loss', round(min(backtest.Profit_pips),1)))

        return stats
    
    
    def generate_backtest(self, entryParameters, eventLookback, orderType, 
                          exitPeriod, divide=1, plot=True):
        """
        Generates the backtest according to varios methods.
        
        """
        self._generateSignals(entryParameters, eventLookback,
                              orderType, exitPeriod)
        
        self.backtest = self._generateBacktestDataFrame(divide)
        
        # Plotting.
        if plot == True:
            self.plot_backtest()


    def _generateBacktestDataFrame(self, divide = 1):
        # Creating the backtest DataFrame and the basic components.
        backtest = pd.DataFrame(index=range(len(self.exitSignals)))
        length = len(backtest) # Number to limit the entry signals           
                               # in case they are more than the exit signals.
        
        backtest['Symbol'] = list(self.exitSignals.Symbol)
        backtest['Entry_time'] = list(self.entrySignals.Datetime[0:length])
        backtest['Entry_price'] = list(self.entrySignals.Price[0:length])
        backtest['Order_type'] = list(self.entrySignals.Order_type[0:length])
        backtest['Exit_time'] = list(self.exitSignals.Datetime)
        backtest['Exit_price'] = list(self.exitSignals.Price)
        
        # Calculating the profits in pips.
        # TODO: use numpy when all signals have the same orderType
        profits = []
        for i in backtest.index:
            if backtest.Order_type.iloc[i] == 'BUY':
                profits.append((backtest.iloc[i].Exit_price 
                                - backtest.iloc[i].Entry_price 
                                - self.commission) / 10)
            if backtest.Order_type.iloc[i] == 'SELL':
                profits.append((backtest.iloc[i].Entry_price 
                                - backtest.iloc[i].Exit_price 
                                - self.commission) / 10)

        # Cumulative stats.
        profits = list(np.array(profits) / divide)
        backtest['Profit_pips'] = profits
        backtest = pd.concat([backtest, self.calculate_cumulative_stats(profits)], 
                             axis=1)
                        
        return backtest


    def _generateSignals(self, entryParameters, eventLookback, orderType, 
                         exitPeriod):
        """
        Generates the entry and exit signals to calculate the backtesting
        DataFrame.
        """
        print(entryParameters)        
        
        initializedStrategy = self.strategy.initialize_strategy(orderType = orderType,
                                                                **entryParameters)
        self.entrySignals = initializedStrategy.generate_signals(lookback=eventLookback)
        
        try:        
            self.exitSignals = initializedStrategy.generate_exit_signals(exitPeriod)
        except UnboundLocalError:
            raise AttributeError('The strategy doesn''t have and orderType')


    def generate_stats(self, applyTo='backtest', printIt=False):
        backtest = getattr(self, applyTo)
        
        stats = self._calculate_final_stats(backtest)
        stats.append(('Commission', round(self.commission / 10, 2)))
        stats.append(('Percentage of commission', 
                      round((self.commission / 10) 
                             / (backtest.Cum_mean.iloc[-1] 
                             + (self.commission / 10)),2)))

        # Setting attribute.
        self.stats = stats

        # Printing.
        if printIt == True:
            pprint(stats)

    def merge_backtests(self, backtests):
        """
        Merges different backtests concatenating the signals.
        """
        # Merging all the backtests df
        backtest = pd.concat([x.backtest.loc[:,'Symbol':'Profit_pips'] 
                   for x in backtests])
        backtest.sort_values('Exit_time', inplace=True)
        backtest.index = range(len(backtest.index))
        backtest = pd.concat([backtest,
                   self.calculate_cumulative_stats(list(backtest.Profit_pips))], 
                   axis=1)
                   
        # Building the new backtest object.
        bt = Backtest(backtests[0].strategy, backtests[0].commission,
                      backtests[0].swap)
        bt.backtest = backtest
        bt.generate_stats(applyTo = 'backtest', printIt = False)      
            
        return bt
        
            
    def plot_backtest(self):
        """
        """
        for col in self.backtest.loc[:,'Profit_pips':'Drawdown'].columns:  
            self.backtest[col].plot(figsize=(11,6), title=col)
            plt.show()


    def plot_mm_backtest(self):
        """
        """
        for col in self.mmBacktest.loc[:,'Profit':'Drawdown'].columns:  
            self.mmBacktest[col].plot(figsize=(11,6), title=col)
            plt.show()


if __name__ == "__main__":
    """
    """
    parameters = {'SMAlookback' : 2000,
                  'KDlookback' : 100,
                  'symbol' : 'EURUSD',
                  'timeframe' : 'H1',
                  'SMAmethod' : 'below',
                  'KDmethod' : 'above'}

    bt = Backtest(SmaKnoxville(startDate = dt.date(2010,1,1), 
                                endDate = dt.date(2013,12,31)),
                  commission=20)
    bt.generate_backtest(parameters, 'BUY', 20, divide=1)
    bt.generate_stats()
    
    

    
             
                 




