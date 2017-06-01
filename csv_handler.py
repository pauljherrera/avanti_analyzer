# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 11:20:23 2016

@author: Paul Herrera
"""

from __future__ import division
from __future__ import print_function

import pandas as pd
import os
from os.path import exists
import datetime as dt

class CsvHandler():
    """
    Acts as an abstract superclass
    """
    def __init__(self, forceCalculation=False, startDate=None, endDate=None):
        self.csvFile = os.path.join('db', '%s.csv'%self.name)
        
        if (forceCalculation == False) & (exists(self.csvFile) == True):
            self.indicator = pd.read_csv(self.csvFile, header=0, index_col=0,
                                         parse_dates=['Datetime'])
            self.parse_indicator_date_time() 
            self.filter_indicator_dates(startDate, endDate)
            
        else:
            # Indicator calculation
            self.indicator = self.get_prices_datetime() #HistoricalData method.
            self.indicator['Value'] = self.calculate_indicator() #Indicator method.
            
            # Indicator storage only if it hasn't date limits.
            if (startDate==None) and (endDate==None):
                self.indicator.to_csv(self.csvFile)
        
    def filter_indicator_dates(self, startDate=None, endDate=None):
        """
        """
        if startDate != None:
            self.startDate = startDate
            self.indicator = self.indicator[self.indicator.Date > startDate]
        else:
            self.startDate = self.prices.Date[0]
            
        if endDate != None:
            self.endDate = endDate
            self.indicator = self.indicator[self.indicator.Date < endDate]
        else:
            self.endDate = self.prices.Date.iloc[-1]

    def parse_indicator_date_time(self):
        # Converting Date and Time columns in Date and Time type
        self.indicator['Date'] =\
            map(lambda x: dt.date(int(x[0:4]), int(x[5:7]), int(x[8:10])), 
                self.indicator.Date)
        self.indicator['Time'] =\
            map(lambda x: dt.time(int(x[0:2]), int(x[3:5])), 
                self.indicator.Time)

 
        

    
    