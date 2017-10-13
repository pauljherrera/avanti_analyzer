# -*- coding: utf-8 -*-
"""
Created on Mon Sep 05 07:49:07 2016

@author: forex
"""

from __future__ import division
from __future__ import print_function

from pandas.io.parsers import read_csv
from os import path as pt
import datetime as dt
from copy import copy
from scipy import stats

from .scripts import custom_functions as cf


class HistoricalData(object):
    """
    Reads from the csv folder the historical data of a symbol and a timeframe.
    The date and time are columns and not part of the index.
    
    Future possibilities: - Union of two historical data for mean-reverting or
    pair trading.
    """
    def __init__(self, symbol, timeframe, path='csv', startDate=None,
                 endDate=None):
        self.symbol = symbol
        self.timeframe = timeframe
        self.barsPerDay = self.get_bars_per_day(timeframe)
        self.read_prices(symbol, timeframe, path)
        self.decimals = self.get_decimals(self.prices.Close.iloc[-300:]) 
                                # The [-300:] is ther to improve performance
        self.turn_prices_into_integers()
        
        #If a startDate is provided...
        if startDate != None:
            self.startDate = startDate
            self.prices = self.prices[self.prices.Date > startDate]
        else:
            self.startDate = self.prices.Date.iloc[0]
            
        #If an endDate is provided...
        if endDate != None:
            self.endDate = endDate
            self.prices = self.prices[self.prices.Date < endDate]
        else:
            self.endDate = self.prices.Date.iloc[-1]
            
        self.adapted = False


    def __repr__(self):
        try:        
            print(self.prices)
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
        
        # Adapting the prices.
        self.prices = cf.adapt_to_timeframe(self.prices, time1, time2)
        
        # Declaring as adapted and re-reading the prices.                
        self.adapted = True
        
        return self.prices
        
    @staticmethod    
    def get_bars_per_day(timeframe):
        if timeframe == 'D1': barsPerDay = 1
        elif timeframe == 'H4': barsPerDay = 6
        elif timeframe == 'H1': barsPerDay = 24
        elif timeframe == 'M15': barsPerDay = 24 * 4
        elif timeframe == 'M5': barsPerDay = 24 * 12
        elif timeframe == 'M1': barsPerDay = 24 * 60
        return barsPerDay


    def get_decimals(self, Series):
        """
        Gets the current symbol number of decimals after the comma.
        """
        decimals = map(lambda x: str(Series.iloc[x]).split('.')[1], 
                   range(len(Series))) 
        decimals = [len(x) for x in decimals]
        decimals = stats.mode(decimals)[0][0]
        return decimals


    def get_prices_datetime(self):
        """
        """
        df = copy(self.prices)
        df.drop('Open', axis=1, inplace=True)
        df.drop('High', axis=1, inplace=True)
        df.drop('Low', axis=1, inplace=True)
        df.drop('Close', axis=1, inplace=True)
        df.drop('Vol', axis=1, inplace=True)

        return df


    @staticmethod    
    def limit_dates(df, startDate=None, endDate=None):
        """
        Limits the start date and the end date of a DataFrame.
        """
        #If a startDate is provided...
        if startDate != None:
            df = df[df.Date > startDate]
            
        #If an endDate is provided...
        if endDate != None:
            df = df[df.Date < endDate]
    
        return df
    
    def read_prices(self, symbol, timeframe, path='csv'):
        """
        """
        fileName = symbol + '_' + timeframe + '_UTC+0_00_noweekends.csv'
        self.prices = read_csv(pt.join(path, fileName), header=None)

        # Assigning header.        
        self.prices.columns = ["Date","Time","Open","High","Low","Close","Vol"]
        
        # Converting Date and Time columns in Date and Time type        
        self.prices['Date'] = [dt.date(int(x[0:4]), int(x[5:7]), int(x[8:10])) 
                               for x in self.prices.Date]
        self.prices['Time'] = [dt.time(int(x[0:2]), int(x[3:5])) for x in 
                               self.prices.Time]
                               
        # Creating Datetime column.
        self.prices['Datetime'] = [dt.datetime.combine(x,y) for x,y in zip(
                               self.prices.Date, self.prices.Time)]
        
        # Rearranging columns
        cols = self.prices.columns.tolist()        
        cols = cols[-1:] + cols[:-1]
        self.prices = self.prices[cols]
        
        return self.prices
        
    def turn_prices_into_integers(self):
        self.prices['Open'] = self.prices.Open.apply(lambda x: int(x * pow(10, 
                                                            (self.decimals))))
        self.prices['High'] = self.prices.High.apply(lambda x: int(x * pow(10, 
                                                            (self.decimals))))
        self.prices['Low'] = self.prices.Low.apply(lambda x: int(x * pow(10, 
                                                            (self.decimals))))
        self.prices['Close'] = self.prices.Close.apply(lambda x: int(x * pow(10, 
                                                            (self.decimals))))



if __name__ == "__main__":
    
    a = HistoricalData('EURUSD', 'M15')
    
    
    