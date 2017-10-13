# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 13:39:39 2016

@author: forex
"""
from __future__ import division
from __future__ import print_function



from .indicators import Knoxville_div, RSI, SDC, SMA
from .strategy_creator import StrategyCreator


class Strategy(object):
    def __init__(self, startDate = None, endDate =None):
        self.startDate = startDate
        self.endDate = endDate        

    def initialize_strategy(self):
        """
        Main method.
        Given a set of parameters, initializes a strategy.
        """
        raise NotImplementedError('To be implemented')



class KDDouble(Strategy):
    """
    """
    def initialize_strategy(self, symbol, timeframe1, lookback1, extension1,
                            timeframe2, lookback2, extension2, 
                            method, orderType = None,
                            commission=0, swap=(0,0)):
        # Indicators intantiation.
        extensionValue = 1 if method == 'above' else -1
                            
        indicator1 = Knoxville_div(symbol = symbol, 
                                              timeframe = timeframe1, 
                                              lookback = lookback1,
                                              startDate = self.startDate,
                                              endDate = self.endDate)
        indicator1.indicator = indicator1.extend_indicator_value(
                                  indicator1.indicator, extension1, 
                                  extensionValue)
        indicator1.filter_indicator(level = 0, method=method)
        indicator1.adapt_to_timeframe(timeframe2)

        indicator2 = Knoxville_div(symbol = symbol, 
                                              timeframe = timeframe2, 
                                              lookback = lookback2,
                                              startDate = self.startDate,
                                              endDate = self.endDate)
        indicator2.indicator = indicator2.extend_indicator_value(
                                  indicator2.indicator, extension2, 
                                  extensionValue)
        indicator2.filter_indicator(level = 0, method=method)

        # Deleting heavy dataframes.
        indicator1.save_memory()
        indicator2.save_memory()

        # Initializing the strategy.
        return StrategyCreator(symbol, timeframe1,
                               indicatorsList = [indicator1, indicator2],
                               adapt=timeframe2, orderType = orderType,
                               commission=commission, swap=swap)
    
    
    
class RSIDouble(Strategy):
    """
    """
    def initialize_strategy(self, symbol, timeframe1, lookback1,
                            timeframe2, lookback2, 
                            level1, method1, level2, method2, orderType = None,
                            commission=0, swap=(0,0)):

        # Indicators intantiation.
        indicator1 = RSI(symbol = symbol, 
                            timeframe = timeframe1, 
                            lookback = lookback1, 
                            startDate = self.startDate,
                            endDate = self.endDate)
        indicator1.adapt_to_timeframe(timeframe2)
        
        indicator2 = RSI(symbol = symbol, 
                                    timeframe = timeframe2, 
                                    lookback = lookback2,
                                    startDate = self.startDate,
                                    endDate = self.endDate)
                                    
        # Initializing OS/OB conditions according to the parameters
        indicator1.filter_indicator(level = level1, method = method1)
        indicator2.filter_indicator(level = level2, method = method2)
         
        # Deleting heavy dataframes.
        indicator1.save_memory()
        indicator2.save_memory()
                
        # Initializing the strategy.
        return StrategyCreator(symbol, timeframe1,
                               indicatorsList = [indicator1, indicator2],
                               adapt=timeframe2, orderType = orderType,
                               commission=commission, swap=swap)
        
        
                                    
class RSI_SDC(Strategy):
    def initialize_strategy(self, symbol, timeframe, RSIlookback, RSImethod,
                            RSIlevel, SDCrepetition, orderType=None,
                            commission=0, swap=(0,0)):
        #Indicator instantiation.
        indicator1 = RSI(symbol = symbol, 
                            timeframe = timeframe, 
                            lookback = RSIlookback, 
                            startDate = self.startDate,
                            endDate = self.endDate)
        indicator2 = SDC(symbol = symbol, 
                            timeframe = timeframe) 
                                    
        # OS/OB condition and number of repetitions.
        indicator1.filter_indicator(level = RSIlevel, method = RSImethod)
        indicator2.filter_indicator(level = SDCrepetition, method = 'equals')
        
        # Deleting heavy dataframes.
        indicator1.save_memory()
        indicator2.save_memory()

        # Initializing the strategy.
        return StrategyCreator(symbol, timeframe,
                               indicatorsList = [indicator1, indicator2],
                               orderType = orderType,
                               commission=commission, swap=swap)


class SmaKnoxville(Strategy):
    """
    """
    def initialize_strategy(self, symbol, timeframe, SMAlookback, KDlookback,
                            SMAmethod, KDmethod, orderType=None,
                            commission=0, swap=(0,0)):
        
        #Indicator instantiation.
        indicator1 = SMA(symbol = symbol, 
                            timeframe = timeframe, 
                            lookback = SMAlookback,
                            startDate = self.startDate,
                            endDate = self.endDate)

                                                                       
        indicator1.filter_indicator(level = indicator1.prices.Close, 
                                    method = SMAmethod)
                                    
        indicator2 = Knoxville_div(symbol = symbol, 
                                      timeframe = timeframe, 
                                      lookback = KDlookback,
                                      startDate = self.startDate,
                                      endDate = self.endDate)
        indicator2.filter_indicator(level = 0, method = KDmethod)

        # Deleting heavy dataframes.
        indicator1.save_memory()
        indicator2.save_memory()

        # Initializing the strategy.
        return StrategyCreator(symbol, timeframe,
                        indicatorsList = [indicator1, indicator2], 
                        orderType = orderType,
                        commission=commission, swap=swap)
                        
                        
if __name__ == '__main__':
    s = RSI_SDC()
    strat = s.initialize_strategy('GBPUSD','M15',14,'above',65,4)
    strat.generate_optimized_event_study(lookback=10)

