# -*- coding: utf-8 -*-
"""
Created on Mon Sep 05 13:46:39 2016

@author: forex
"""

from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import copy
import sys
import os
from os.path import exists
import datetime as dt
import gc

sys.path.append('scripts')
from historical_data import HistoricalData
from KnoxPy import relative_strength, Knoxpy_df2, MOM
from csv_handler import CsvHandler
import custom_functions as cf



"""
Superclass: Indicator. Used like an abstract superclass.
"""
class Indicator(HistoricalData):
    """
    """
    def __init__(self, symbol, timeframe, lookback=None, path='csv', 
                 startDate=None, endDate=None):
        HistoricalData.__init__(self, symbol, timeframe, path, startDate,
                 endDate)
        if lookback: 
            self.lookback = int(lookback)
        self.adapted = False
        
        # Indicator calculation
        self.indicator = self.get_prices_datetime() #HistoricalData method.
        self.indicator['Value'] = self.calculate_indicator() 
        
        
    def __repr__(self):
        try:        
            print(self.indicator)
        except:
            pass
    
    
    def adapt_to_timeframe(self, timeframe):
        """
        Adapts the 'Date, 'Time' and 'Datetime' columns of a df so it is 
        compatible with an indicator of higher timeframe.
        """
        # Making the variables easier to read.
        time1 = self.timeframe 
        time2 = timeframe
        df = self.indicator 
        
        # Adapting the prices.
        df = cf.adapt_to_timeframe(df, time1, time2)
        
        # Assigning the attributes.                
        self.indicator = df
        self.adapted = True
   

    def break_level(self, level, method='above'):
        """
        Leaves the moments in which a level is trespassed.
        """
        self.indicator['Shift'] = self.indicator.Value.shift()
        
        if method == 'above':
            self.indicator = self.indicator[(self.indicator.Value > level)\
                                            & (self.indicator.Shift < level)]
        elif method == 'below':
            self.indicator = self.indicator[(self.indicator.Value < level)\
                                            & (self.indicator.Shift > level)]
                           
        del self.indicator['Shift']
    
        return self.indicator    
    
      
    @staticmethod
    def extend_indicator_value(indicator, extensionPeriod, value=True):
        """
        Extends the passed value of an indicator for a period of time.
        Example: Extend for ten periods the True signal after a two-SMA
                 crossover.
        TODO: update. Doesn't have 'store' nor limit dates.
        """
        indexes = [(x, x+extensionPeriod) for x 
                   in indicator[indicator.Value == value].index]
        for x,y in indexes:
            indicator.loc[x:y,'Value'] = value
            
#        extended = copy.copy(indicator)
#        for i in extended.sort_index(ascending=False).index:
#            if (extended.Value.loc[i] != value):
#                if value in extended.Value.loc[i-extensionPeriod:i-1].values:
#                    extended.Value.loc[i] = value
                    
        return indicator
        
      
    def filter_indicator(self, level, method='above', filterName = None, 
                     forceCalculation=False, returns=False, delPrice=True):
        """
        Filters as True the values of a Series that are 
        above a Series of the same length or value.
        TODO: deal with starDate and endDate when it isn't the same as the
        stored df.
        TODO: store the adapted version. So it doesn't have to force calculation
        when it is adapted.
        """
        # Update name and csvFile
        self.update_name(level, method, filterName)

        # Indicator calculation
        if method == 'above':
            self.indicator =\
                self.indicator[self.indicator.Value > level]
        elif method == 'below':
            self.indicator =\
                self.indicator[self.indicator.Value < level]
        elif method == 'equals':
            try:
                self.indicator =\
                    self.indicator[self.indicator.Value == level]
            except TypeError:
                self.indicator =\
                    self.indicator[self.indicator.Value == float(level)]
                                                                
        if returns == True:
            return self.indicator
            
             
    def calculate_indicator(self):
        """
        Returns a Series with the indicator calculated
        """
        raise NotImplementedError('To be implemented')
        
        
    def save_memory(self):
        del self.prices
        gc.collect()
       

    def store_indicator(self):
       """
       Coded apart to have storage options in the future.
       """
       if len(self.indicator) > 0:
           self.indicator.to_csv(self.csvFile)

             
    def update_name(self, element, methodName="", filterName=None):
        """
        """
        if filterName == None:
            if type(element)==int:
                self.name += '_' + methodName + '%i'%element
            elif type(element)==type(self.prices.Close):
                self.name += '_' + methodName + '%s'%element.name
        else:
            self.name += '_' + methodName + filterName

        self.csvFile = os.path.join('db', '%s.csv'%self.name)



"""
Subclasses: all the indicators
"""    

class Knoxville_div(Indicator):
    """
    """
    def __init__(self, symbol, timeframe, lookback, path='csv',
                 startDate=None, endDate=None):       
        Indicator.__init__(self, symbol, timeframe, lookback, path, startDate,
                 endDate)
        self.name = (symbol + '_' + timeframe + '_' + 'Knoxville_Div{}'.format(lookback))

        
    def calculate_indicator(self):
        df = Knoxpy_df2(self.prices, lookback=self.lookback, verbose=True)
        return df.Knoxpy


class Momentum(Indicator):
    """
    """
    def __init__(self, symbol, timeframe, lookback=14, path='csv',
                 startDate=None, endDate=None):
        Indicator.__init__(self, symbol, timeframe, lookback, path, startDate,
                 endDate)
        self.name = (symbol + '_' + timeframe + '_' + 'Momentum{}'.format(lookback))

        
    def calculate_indicator(self):
        return MOM(self.prices, self.lookback).Momentum


                   
class RSI(Indicator):
    """
    TODO: the RSI calculation differs a little bit from the MT4 RSI calculation.
    """
    def __init__(self, symbol, timeframe, lookback=14, path='csv',
                 startDate=None, endDate=None):
        Indicator.__init__(self, symbol, timeframe, lookback, path, startDate,
                 endDate)
        self.name = (symbol + '_' + timeframe + '_' + 'RSI{}'.format(lookback))

        
    def calculate_indicator(self):
        return relative_strength(self.prices.Close, self.lookback)
                                            
              
# Same Direction Candles Indicator.
class SDC(Indicator):
    """
    This indicator counts the amount of candles that goes in 
    the same direction (up or down) in consecutive days.
    """
    def __init__(self, symbol, timeframe, path='csv',
                 startDate=None, endDate=None):
        Indicator.__init__(self, symbol, timeframe, None, path, startDate,
                 endDate)
        self.name = (symbol + '_' + timeframe + '_' + 'SDC')
        
    def calculate_indicator(self):
        result = pd.Series(np.where(self.prices.Close < self.prices.Close.shift(), 
                                    -1, 1))
        
        series = newSeries = result
        while any(newSeries):
            newSeries = pd.Series(np.where(series == series.shift(), 
                                           np.sign(series), 0))
            result += newSeries
            series = newSeries 
            
        return result
        
              
class SMA(Indicator):
    """
    """
    def __init__(self, symbol, timeframe, lookback, path='csv', 
                 startDate=None, endDate=None):
        Indicator.__init__(self, symbol, timeframe, lookback, path, startDate,
                 endDate)
        self.name = (symbol + '_' + timeframe + '_' + 'SMA{}'.format(lookback))

            
    def calculate_indicator(self):
        return self.prices.Close.rolling(window=self.lookback, 
                                         center=False).mean()


if __name__ == "__main__":
    """
    """
    ind = Knoxville_div('GBPUSD', 'D1', 30)
    mom = Momentum('GBPUSD', 'D1', 20)




    

    
    
    
    