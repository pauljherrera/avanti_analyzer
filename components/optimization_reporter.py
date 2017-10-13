# -*- coding: utf-8 -*-
"""
Created on Mon Jan 02 13:59:57 2017

@author: forex
"""

from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import datetime as dt
from copy import copy
from pprint import pprint


class ReporterBridge(object):
    """
    """
    def __init__(self, reporter=None):
        self._reporter = reporter
    
    def show_report(self):
        """
        Main method.
        """
        self._reporter.show_report()
        

class OptimizationReporterBridge(ReporterBridge):
    
    def __init__(self, reporter=None):
        if not reporter: 
            self._reporter = MatrixScreenReporter()
        else: 
            self._reporter = reporter
    
    def show_report(self, optimization):
        """
        Pass an Optimizer instance.
        """
        print('')
        self._reporter.show_report(optimization.performanceMatrix)


class BacktestReporterBridge(ReporterBridge):
    
    def show_report(self, backtest):
        """
        Pass a Backtest instance.
        """
        print('')
        backtest.backtest.Profit_pips.plot(figsize=(10,6))
        plt.show()



class Reporter(object):
    """
    """
    def show_report(self):
        """
        Main method.
        """
        raise NotImplementedError('To be implemented')


class ScreenReporter(Reporter):
    
    def show_report(self, reported):
        pprint(reported)


class MatrixScreenReporter(ScreenReporter):
    
    def show_report(self, reported):
        reported = reported[reported.columns].astype(float)
        sns.heatmap(reported, annot=True)
        plt.show()

        

if __name__ == "__main__":
    """
    """
    reporter = OptimizationReporterBridge(MatrixScreenReporter())
    reporter.show_report()
    
    