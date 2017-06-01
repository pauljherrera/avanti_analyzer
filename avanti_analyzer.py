# -*- coding: utf-8 -*-
"""
Created on Mon Sep 05 12:40:04 2016

@author: forex
"""

from __future__ import division
from __future__ import print_function

import numpy as np
from matplotlib import pyplot as plt
import seaborn
import shelve

import indicators
from strategy import Strategy


"""
run = True

if run==True:
    sma = indicators.SMA('EURUSD', 'H4', 50, forceCalculation=False)
    sma.filter_indicator(sma.prices.Close, method='below')    

    rsi = indicators.RSI('EURUSD', 'M15', 14, forceCalculation=False)
    rsi.filter_indicator(30, method='below')
    
    
    strat = Strategy('EURUSD', 'M15', 'BUY', commission = 15, swap=(-5,-9),
                     indicatorsList = [sma, rsi])
    strat.generate_optimized_event_study(eventwindow=(-100, 500), 
                                         commission = True, swap=True)
    #strat.plot_all_events(showExamples = 50, commission=True)
    print(strat.signalsSpan)
    print('')
    print(strat.eventsRelevantData)
    print('')
    print(strat.orderType)
    print('')
    print(strat.eventsNum)    
"""   
    
indicator = indicators.RSI('EURUSD', 'H1', 14, forceCalculation=False)
indicator.filter_indicator(70, method='above')

#    strategy = Strategy('EURUSD', 'H1', 'BUY', commission = 15, swap=(-5,-9),
#                     indicatorsList = [indicator])
#    strategy.generate_optimized_event_study(eventwindow=(-100, 500), 
#                                         commission = True, swap=True, 
#                                         plot=True)
#    optimization[v] = strategy
#    
#for k in np.sort(optimization.keys()):
#    optimization[k].plot_all_events(commission=True)
#    print(optimization[k].eventsRelevantData)
#    print(optimization[k].orderType)
#    print(optimization[k].signalsSpan)



    
"""
strat.generate_optimized_event_study(eventwindow=(-100, 500), commission = 10)
strat.plot_all_events(showExamples = 50, commission=True)
print(strat.signalsSpan)
print(strat.eventsRelevantData)
print(strat.eventsNum)

""
strat.generate_signals(1,25)
strat.generate_event_study(eventwindow = (-20,40), neighbours = 'auto')


""
plt.figure(1, figsize = (9, 7))
ylim = int(strat.eventsStats.Std.loc[0:].max() / 5)
plt.ylim(-ylim, ylim)
for e in strat.eventsStandard.columns[-50:]:
    plt.plot(strat.eventsStandard.loc[:,e] / 10, alpha=0.05,
             color='#793471')
plt.plot(strat.eventsStats.Mean / 10, 'r', linewidth=2.0)
plt.figure(2, figsize = (9, 3.5))
plt.plot(strat.eventsStats.Mean.loc[0:] / 10, 'r', linewidth=2.0)
plt.show()
"""

