# -*- coding: utf-8 -*-
"""
Created on Mon Sep 05 13:07:34 2016

@author: forex
"""

from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
from copy import copy
from matplotlib import pyplot as plt
import seaborn as sns

from .historical_data import HistoricalData
from .scripts.custom_functions import round_to_multiple, max_argmax_df



class StrategyCreator():
    """
    Output: a list of 'BUY' and 'SELL' signals for a symbol. These signals are 
    generated when certain indicators met or doesn't.
    """
    def __init__(self, outputSymbol, outputTimeframe, orderType=None, 
                 commission = 0, swap = (0,0), indicatorsList = None,
                 forceOrderType = False, adapt=False):
        self.base = HistoricalData(outputSymbol, outputTimeframe)

        self.orderType = orderType
        self.commission = commission
        self.swap = (round(float(swap[0])/self.base.barsPerDay, 3), 
                      round(float(swap[1])/self.base.barsPerDay, 3))
        self.indicators = {}
        if indicatorsList != None:
            for i in indicatorsList:
                self.add_indicator(i)
        self._minEventWindow = 70
        self.forceOrderType = forceOrderType
        self.adapted = False
        
        # Adapting prices. Opposed as the indicators and historical data, the
            # strategies keep both regular and adapted prices.
        if adapt != False:
            self.adaptedPrices= self.base.adapt_to_timeframe(adapt)
            self.base = HistoricalData(outputSymbol, outputTimeframe)
            self.adapted = True
        
    
    def add_indicator(self, indicator):
        """
        Adds a condition DataFrame to the 
        """
        self.indicators[(len(self.indicators) + 1)] = indicator

    def generate_event_study(self, eventwindow = (-100, 500), commission = False, 
                             swap = False, neighbours = 'auto'):
        """
        """
        # Setting security zones at the start and end of the signals
        eventSignals = self.signals[(self.signals.index > -eventwindow[0]) & \
                                    (self.signals.index < (self.base.prices.index[-1] - \
                                    eventwindow[1]))]
                                    
        self._eventsPrices = dict(zip(eventSignals.index, 
                            [self.base.prices.loc[i+eventwindow[0]:i+eventwindow[1]] 
                            for i in eventSignals.index]))
                                
        # Number of events
        self.eventsNumber = len(self._eventsPrices)
        
        # Dataframe with standardized prices (only Close)
        self._eventsStandard = {k : list(self._eventsPrices[k].Close - \
                               self._eventsPrices[k].Close.loc[k]) for k in
                               eventSignals.index}
                                                                                           
        self._eventsStandard = pd.DataFrame.from_dict(self._eventsStandard)
        self._eventsStandard.index = range(eventwindow[0],eventwindow[1]+1)
        
#        # Calculate stats without commissions
#        if (commission  == False) & (swap == False):
#            # Calculate stats
        self.eventsStats =\
            pd.DataFrame({'Mean': self._eventsStandard.mean(axis=1).round(1),
                          'Std': self._eventsStandard.std(axis=1).round(1),
                          }, index = self._eventsStandard.index)
                          # The unit here is pipos, not pips
        
        # Calculate stats with commissions. TODO: swaps and slippage.
#        else:
#            swaps = np.ones(eventwindow[1])
#            if self.orderType == 'SELL': 
#                self.commission = -self.commission
#                swaps [:] = - self.swap[1]
#            elif self.orderType == 'BUY':
#                swaps [:] = self.swap[0]
#            swaps = np.cumsum(swaps)
#            # Charging commissions only after the event.
#            commissionedEvents = pd.concat([self._eventsStandard.loc[eventwindow[0]:0],
#                              self._eventsStandard.loc[1:] - self.commission])
#            mean = commissionedEvents.mean(axis=1).round(1)
#            # Charging swaps only after the event.
#            mean = pd.concat([mean.loc[eventwindow[0]:0],
#                              mean.loc[1:] + swaps])
#
#            self.eventsStats = pd.DataFrame({'Mean': mean,
#                                             'Std': 
#                               commissionedEvents.std(axis=1).round(1),
#                                             }, index = self._eventsStandard.index)
                                             
        self.eventsStats['Sharpe'] = self.eventsStats.Mean / self.eventsStats.Std
        self.eventsStats['SQN'] = self.eventsStats.Sharpe * \
                                  np.sqrt(self.eventsNumber)
                                  
        # Calculating positive and negative sortino ratios.                          
        self.eventsStats['Sortino_positive'] =\
            (self.eventsStats.Mean - self.commission) / \
            np.sqrt(np.power((self._eventsStandard - commission).\
            clip(upper=0),2).mean(axis=1))
        self.eventsStats['Sortino_negative'] =\
            (-self.eventsStats.Mean - self.commission) / \
            np.sqrt(np.power((-self._eventsStandard - commission).\
            clip(upper=0),2).mean(axis=1))

        # Calculating neigh_SQN and neigSortino with or without optimization    
        if neighbours == 'auto':
            self.eventsStats['Neig_Sortino_Pos'] =\
                self._optimize_sortino_neighbours(self.eventsStats.Sortino_positive,
                                                 eventwindow=eventwindow)
            self.eventsStats['Neig_Sortino_Neg'] =\
                self._optimize_sortino_neighbours(self.eventsStats.Sortino_negative,
                                                 eventwindow=eventwindow)
            if (commission  == False) & (swap == False):
                neighbours, RSQN = self._optimize_neighbours(eventwindow)
            else:
                neighbours, RSQN = self._optimize_neighbours(eventwindow, 
                                                      commission = True)
        else:
            self.eventsStats['Neig_SQN'] = self.eventsStats.SQN.loc[1:].rolling(window = \
                        neighbours * 2 + 1, center=True).mean().round(3)
                          
        self.eventsStats.Sharpe.loc[0] = 0
        self.eventsStats.SQN.loc[0] = 0
        self.eventsStats.Sortino_positive.loc[0] = 0
        self.eventsStats.Sortino_negative.loc[0] = 0
        
        
        # Dictionary with relevant data       
        if (commission  == False) & (swap == False):   # Without commissions    
            self.eventsRelevantData = {'Best SQN' : 
                max(abs(self.eventsStats.loc[0:].SQN)),
                                       'Period with best SQN' : 
                np.argmax(abs(self.eventsStats.loc[0:].SQN)),
                                       'Best Neig_SQN' : 
                round(self.eventsStats.SQN.loc[np.argmax(abs(self.eventsStats.loc[neighbours+1:].Neig_SQN))], 3),
                                       'Period with best Neig_SQN' : 
                np.argmax(abs(self.eventsStats.loc[neighbours+1:].Neig_SQN)),
                                       'Sharpe of period with best Neig_SQN' : 
                round(self.eventsStats.Sharpe.loc[np.argmax(abs(self.eventsStats.loc[neighbours+1:].Neig_SQN))], 3),
                                       'Mean of period with best Neig_SQN' : 
                self.eventsStats.Mean.loc[np.argmax(abs(self.eventsStats.loc[neighbours+1:].Neig_SQN))]/10,
                                       'Neighbours': neighbours,
                                       'RSQN of period with best Neig_SQN' :
                RSQN}
            
        else:  # With commissions
            # Setting functions according to SELL or BUY strategies to calculate
                # the relevantData
            if self.orderType == 'BUY': 
                opt = max
                argopt = np.argmax
            elif self.orderType == 'SELL': 
                opt = min 
                argopt = np.argmin
            self.eventsRelevantData = {'Best SQN' : 
                opt(self.eventsStats.loc[0:].SQN),
                                       'Period with best SQN' : 
                argopt(self.eventsStats.loc[0:].SQN),
                                       'Best Neig_SQN' : 
                round(self.eventsStats.SQN.loc[argopt(self.eventsStats.loc[neighbours+1:].Neig_SQN)], 3),
                                       'Period with best Neig_SQN' : 
                argopt(self.eventsStats.loc[neighbours+1:].Neig_SQN),
                                       'Sharpe of period with best Neig_SQN' : 
                round(self.eventsStats.Sharpe.loc[argopt(self.eventsStats.loc[neighbours+1:].Neig_SQN)], 3),
                                       'Mean of period with best Neig_SQN' : 
                round(self.eventsStats.Mean.loc[argopt(self.eventsStats.loc[neighbours+1:].Neig_SQN)]/10, 2),  # In pips
                                       'Neighbours': neighbours,
                                       'RSQN of period with best Neig_SQN' :
                RSQN}
       
        # Common stats.
        (self.eventsRelevantData['Best Sortino ratio'], 
        (self.eventsRelevantData['Period with best Sortino ratio'], x)) =\
            max_argmax_df(self.eventsStats.loc[1:,'Sortino_positive':'Sortino_negative'])
        (x, 
        (self.eventsRelevantData['Period with best Neig_Sortino'], y)) =\
            max_argmax_df(self.eventsStats.loc[1:,'Neig_Sortino_Pos':'Neig_Sortino_Neg'])
        self.eventsRelevantData['Neighbours'] = self._optSortinoNeighbours
            
        self.eventsRelevantData['Mean of positive Sortino'] =\
            np.round(self.eventsStats.Sortino_positive.loc[1:].mean(),3)
        self.eventsRelevantData['Mean of negative Sortino'] =\
            np.round(self.eventsStats.Sortino_negative.loc[1:].mean(),3)
        self.eventsRelevantData['Sortino Bias'] =\
            self.eventsRelevantData['Mean of positive Sortino'] - \
            self.eventsRelevantData['Mean of negative Sortino']
        self.eventsRelevantData['SQS'] =\
            max(self.eventsRelevantData['Mean of positive Sortino'],
                self.eventsRelevantData['Mean of negative Sortino']) * \
            np.sqrt(self.eventsNumber)
            
        self.eventsRelevantData['Mean of SQNs'] =\
            round(self.eventsStats.loc[1:].SQN.mean(),2)
        self.eventsRelevantData['Mean of positive SQNs'] =\
            round(self.eventsStats.loc[1:].\
                  SQN[self.eventsStats.loc[1:].SQN > 0].mean(),2)
        self.eventsRelevantData['Mean of negative SQNs'] =\
            round(self.eventsStats.loc[1:].\
                  SQN[self.eventsStats.loc[1:].SQN < 0].mean(),2)
        self.eventsRelevantData['Sharpe of SQNs'] =\
            round(self.eventsStats.loc[1:].SQN.mean(),2) / \
            round(self.eventsStats.loc[1:].SQN.std(),2)
        try:
            self.eventsRelevantData['Sharpe of positive SQNs'] =\
                round(self.eventsStats.loc[1:].\
                      SQN[self.eventsStats.loc[1:].SQN > 0].mean(),2) / \
                round(self.eventsStats.loc[1:].\
                      SQN[self.eventsStats.loc[1:].SQN > 0].std(),2)
            self.eventsRelevantData['Sharpe of negative SQNs'] =\
                round(self.eventsStats.loc[1:].\
                      SQN[self.eventsStats.loc[1:].SQN < 0].mean(),2) / \
                round(self.eventsStats.loc[1:].\
                      SQN[self.eventsStats.loc[1:].SQN < 0].std(),2)
        except:
            pass
        
        # Stablishing orderType of the strategy if it isn't forced to stay
            # the same.
        if self.forceOrderType == False:
            if self.eventsRelevantData['Mean of positive Sortino'] > \
               self.eventsRelevantData['Mean of negative Sortino']:
                self.orderType = 'BUY'
            else: 
                self.orderType = 'SELL'
        
        # Filling the OrderType column of the signals.
            self.signals.Order_type = self.orderType


    def generate_exit_signals(self, exitPeriod):
        """
        Generates exit signals using the entry signals and the optimized
        exit period.
        """
        index = [x + exitPeriod for x in self.signals.index]
        df = self.base.prices[self.base.prices.index.isin(index)]
        
        if self.orderType == 'BUY':
            order = 'SELL'
        elif self.orderType == 'SELL':
            order = 'BUY'
        else:
            print('No orderType had been asssigned')
        
        # Creating Dataframe.
        exitSignals = pd.DataFrame()
        exitSignals['Datetime'] = df.Datetime
        exitSignals['Symbol'] = copy(self.base.symbol)
        exitSignals['Order_type'] = order
        exitSignals['Price'] = df.Close
        
        # As an attribute.
        self.exitSignals = exitSignals
        
        return self.exitSignals
        

    def generate_optimized_event_study(self, occurrenceNumber = 1, lookback = 1,
                                       eventwindow = (-30,500), 
                                       commission = False, swap = False,
                                       plot = True, verbose=True):
        """
        While loop until the best_sqn (or other performance measurement)
        is below the 1st decile (or other benchmark) of signalsSpan. Also the
        eventwindow is optimized.
        """       
        if verbose == True: 
            print('\nGenerating optimized Event Study\n')
        self.generate_signals(occurrenceNumber, lookback = lookback)
                
        self.generate_event_study(eventwindow)
        
        if lookback == 'auto':
            # A while loop to optimize the lookback.
            while self.eventsRelevantData['Period with best SQN'] >= \
                                       self.signalsSpan['1st decile']:
                if lookback == occurrenceNumber: lookback = 2
                else:
                    lookback = int(round_to_multiple(lookback * 2, 5))
                self.generate_signals(occurrenceNumber, lookback = lookback)
                self.generate_event_study(eventwindow)
            
        # Modify the eventwindow according to the 1st quartile.
        eventwindow = (int(-round_to_multiple(self.signalsSpan['1st quartile'] / 2, 5)),
                       int(round_to_multiple(self.signalsSpan['1st quartile'], 5)))
        if eventwindow[1] < self._minEventWindow:
            eventwindow = (-int(self._minEventWindow/2), self._minEventWindow)
        
        # Creating and plotting  the event study without commission.                
        self.generate_event_study(eventwindow=eventwindow, commission=True)

        if plot == True: 
            self.plot_all_events()
                    
        # Creating and plotting  the event study with commission after storing
            # the commissionless eventStats.
#        if (commission  == True) | (swap == True):
#            self.eventsStatsNC = copy(self.eventsStats)
#            self.generate_event_study(eventwindow=eventwindow, 
#                                      commission=commission, swap=swap)
#            if plot == True: 
#                self.plot_all_events(commission = True)
    
    def generate_signals(self, occurrenceNumber = 1, lookback = 0):
        """
        Merges all the indicators and generates a dataframe with the signals
        for the backtesting/event study.
        """
        if self.adapted == True:
            filtered = copy(self.adaptedPrices)
        else:
            filtered = copy(self.base.prices)
        keys = self.indicators.keys()
        
        for k in keys:
            # Main filter: coincidence of Datetimes
            filtered = filtered[(
                filtered.Datetime.isin(self.indicators[k].indicator.Datetime))]
        
        # Giving signals form.
        self.signals = pd.DataFrame({'Datetime' : filtered.Datetime,
                                     'Symbol' : copy(self.base.symbol),
                                     'Order_type' : copy(self.orderType),
                                     'Price' : filtered.Close})
        self.signals = self.signals[['Datetime', 'Symbol', 'Order_type', 
                                     'Price']]

        # In case the base prices of the strategy has been adapted,
            # it's necessary to change the Datetime to the original one.
        if self.base.adapted == True:
            self.signals.Datetime = filtered.Datetime_original                                  
                                     
        # Filter according to the occurence number and lookback.
        if lookback > 0:
            occurrence = []
            for i in range(len(self.signals.index)):
                counter = 1
                while counter != False:
                    try:
                        # Checking the presence of the index in the previous signals + the lookback.
                        if self.signals.index[i] in range(self.signals.index[i-counter], 
                                                          self.signals.index[i-counter] + \
                                                          lookback + 1):
                            # If it is present, the occurrence counter increases.
                            counter += 1
                        else:
                            # If it isn't, the occurrence number is stored.
                            occurrence.append(counter)
                            counter = False
                    except: 
                        print(self.signals)
                        raise IndexError('index -2 is out of bounds for axis 0 with size 1')
            self.signals['Occurrence'] = occurrence
            
            # Choosing only the signals with the appropriate occurrence number.
            self.signals = self.signals[self.signals.Occurrence == occurrenceNumber]
            self.signals.drop(['Occurrence'], axis=1, inplace=True)
              
        # Calculating average span between signals.
        if len(self.signals) > 0:
            self.signalsSpan = {'Mean' : round(np.diff(self.signals.index).mean(),1),
                                'Std' : round(np.diff(self.signals.index).std(),1),
                                '1st decile' : np.percentile(np.diff(self.signals.index), 10),
                                '1st quartile' : np.percentile(np.diff(self.signals.index), 25),
                                'Median' : np.percentile(np.diff(self.signals.index), 50),
                                '1st centile' : np.percentile(np.diff(self.signals.index), 1),
                                'Lookback' : lookback}
        else:
            print('No signals')
                   
        return self.signals
        
    def _optimize_neighbours(self, eventwindow=(-10,50), commission = False,
                            method='moderate', weights=(0.4,0.6)):
        """
        Input:
        Output:
        """
        if method == 'robust':
            flag = False
            optNeighboursPrev = 0
            optNeighbours = 0
            maxNeighbours = int(eventwindow[1] / 4)
            
            # Mean while loop to optimize the neighbours.
            while flag == False:
                self.eventsStats['Neig_SQN'] = self.eventsStats.SQN.loc[1:].rolling(window =\
                    optNeighboursPrev * 2 + 1, center=True).mean().round(3)
        
                # Setting functions according to SELL or BUY strategies to calculate
                    # the relevantData
                if self.orderType == 'BUY': 
                    argopt = np.argmax
                elif self.orderType == 'SELL': 
                    argopt = np.argmin
                
                if commission == False:
                    locBestSQN = np.argmax(abs(self.eventsStats.Neig_SQN.loc[1:]))
                else:
                    locBestSQN = argopt(self.eventsStats.Neig_SQN.loc[1:])
                
                minLocBestSQN = min([locBestSQN, eventwindow[1] - locBestSQN + 1])
                optNeighbours = minLocBestSQN - 1
                if optNeighbours > maxNeighbours:
                    optNeighbours = maxNeighbours
                    self.eventsStats['Neig_SQN'] = self.eventsStats.SQN.loc[1:].rolling(window =\
                        optNeighbours * 2 + 1, center=True).mean().round(3)
                    break
                if optNeighboursPrev == optNeighbours: flag = True
                optNeighboursPrev = optNeighbours
            self._optNeighbours = optNeighbours

        elif method == 'moderate':
            """
            This method uses a formula to merge the SQN with the robustness
            of the Neig_SQN. It iterates until the formula starts to decrease.
            """
            flag = False
            RSQNPrev = 0
            RSQN = 0
            neighbours = 0
            maxNeighbours = int(eventwindow[1] / 2) -1
            while flag == False:
                #print(neighbours) # For debugging
                self.eventsStats['Neig_SQN'] = self.eventsStats.SQN.loc[1:].rolling(window =\
                    neighbours * 2 + 1, center=True).mean().round(3)
        
                # Setting functions according to SELL or BUY strategies to calculate
                    # the relevantData
                if self.orderType == 'BUY': 
                    argopt = np.argmax
                elif self.orderType == 'SELL': 
                    argopt = np.argmin
                
                if commission == False:
                    locBestSQN = np.argmax(abs(self.eventsStats.Neig_SQN.loc[1:]))
#                    optNeigSQN = np.max(abs(self.eventsStats.Neig_SQN.loc[1:]))
                else:
                    locBestSQN = argopt(self.eventsStats.Neig_SQN.loc[1:])
#                    optNeigSQN = opt(self.eventsStats.Neig_SQN.loc[1:])
    
                minLocBestSQN = min([locBestSQN, eventwindow[1] - locBestSQN + 1])
                optNeighbours = minLocBestSQN - 1
                if optNeighbours > neighbours:
                    neighbours += 1
                    #print(locBestSQN) # For debugging
                    if neighbours == maxNeighbours: 
                        neighbours -= 2
                        break
                elif optNeighbours <= neighbours:   
                    neighbours -= 1
                    break
                try:
                    optSQN = abs(self.eventsStats.SQN.loc[locBestSQN])
                except:
                    print(self.eventsStats)
                    print(locBestSQN)
                    print(self.eventsNumber)
                    raise TypeError('Hay %i evento(s).'%self.eventsNumber)
                # Mean formula that merges SQN and robustness.
                RSQN = (optSQN * weights[0]) + \
                       (np.sqrt(neighbours) * weights[1])
                #print(RSQN) # For debugging
                if RSQN < RSQNPrev: 
                    neighbours -= 2
                    break
                RSQNPrev = RSQN
            
            if neighbours < 0: neighbours = 0 # To avoid bugs 
            self.eventsStats['Neig_SQN'] = self.eventsStats.SQN.loc[1:].rolling(window =\
                neighbours * 2 + 1, center=True).mean().round(3)
            self._optNeighbours = neighbours
            
       
        return self._optNeighbours, RSQNPrev

    
    def _optimize_sortino_neighbours(self, Series, eventwindow=(-10,50), 
                                    weights=(0.55,0.45)):                                    
        """
        """
        flag = False
        RSortinoPrev = 0
        RSortino = 0
        neighbours = 0
        maxNeighbours = int(eventwindow[1] / 2) -1
        while flag == False:
            neigSeries = Series.loc[1:].rolling(window =\
                neighbours * 2 + 1, center=True).mean().round(3)
                
            # Locating best parameters.
            locBestSortino = np.argmax(neigSeries)

            # Finding the nearest limit of the best parameters.
            minLocBestSortino = min([locBestSortino, eventwindow[1] - locBestSortino + 1])
            optNeighbours = minLocBestSortino - 1
            
            if optNeighbours > neighbours:
                neighbours += 1
                if neighbours == maxNeighbours: 
                    neighbours -= 2
                    break
            elif optNeighbours <= neighbours:   
                neighbours -= 1
                break
            try:
                optSortino = Series.loc[locBestSortino]
            except:
                print(self.eventsStats)
                print(locBestSortino)
                print(self.eventsNumber)
                raise TypeError('Hay %i evento(s).'%self.eventsNumber)
                
            # Main formula that merges SQN and robustness.
            RSortino = (optSortino * weights[0]) + \
                   (np.sqrt(neighbours) * weights[1])
                   
            if RSortino < RSortinoPrev: 
                neighbours -= 2
                break
            RSortinoPrev = RSortino
        
        if neighbours < 0: 
            neighbours = 0 # To avoid bugs
            
        neigSeries = Series.loc[1:].rolling(window =\
                neighbours * 2 + 1, center=True).mean().round(3)
        self._optSortinoNeighbours = neighbours
        
        return neigSeries
        

    def plot_all_events(self, mean = True, SQN = True, Sortino = True, commission = False,
                        stats=True, showExamples = 25, 
                        color = '#793471'):
        """
        """
        plt.figure(1, figsize = (11, 6.5))
        plt.title('All events')
        ylim = int(self.eventsStats.Std.loc[0:].max() / 5)
        plt.ylim(-ylim, ylim)
        for e in self._eventsStandard.columns[-showExamples:]:
            plt.plot(self._eventsStandard.loc[:,e] / 10, 
                     #color=color, 
                     alpha=0.3)
        if mean == True:
            plt.plot(self.eventsStats.Mean / 10, 'r', linewidth=2.0)
            plt.figure(2, figsize = (11, 3))
            plt.title('Mean')
            plt.plot(self.eventsStats.Mean.loc[0:] / 10, 'r', linewidth=2.0)
            plt.plot((0,self._eventsStandard.index[-1]),
                     (0,0), 'k', linewidth=1)
            quartile = self.signalsSpan['1st quartile']
            plt.plot((quartile, quartile), plt.ylim(), color = '#8b0000')
            decile = self.signalsSpan['1st decile']
            plt.plot((decile, decile), plt.ylim(), color = '#ff6600')
        if SQN == True:
            plt.figure(3, figsize = (11, 3))
            plt.title('SQN and Neig_SQN')
            if commission == True:
                plt.plot(self.eventsStats.SQN.loc[0:], color = '#793471', 
                         linewidth=2.0)
                plt.plot(self.eventsStats.Neig_SQN.loc[0:], color = 'b', 
                         linewidth=2.0)
                plt.plot((0,self._eventsStandard.index[-1]),
                         (0,0), 'k', linewidth=1)
                plt.scatter(self.eventsRelevantData['Period with best Neig_SQN'],
                            self.eventsRelevantData['Best Neig_SQN'],
                            s=1000, alpha=0.5, color= '#ff6600')
            else:
                plt.plot(abs(self.eventsStats.SQN.loc[0:]), color = '#793471', 
                         linewidth=2.0)
                plt.plot(abs(self.eventsStats.Neig_SQN.loc[0:]), color = 'b', 
                         linewidth=2.0)
        if Sortino == True:
            plt.figure(4, figsize = (11, 3))
            plt.title('Sortino Ratio')
            plt.plot(self.eventsStats.Sortino_positive.loc[0:], 'b', linewidth=2.0)
            plt.plot(self.eventsStats.Sortino_negative.loc[0:], 'r', linewidth=2.0)
            plt.plot(self.eventsStats.Neig_Sortino_Pos.loc[0:], 
                     color = '#793471', linewidth=2.0)
            plt.plot(self.eventsStats.Neig_Sortino_Neg.loc[0:], 
                     color = '#793471', linewidth=2.0)
            
            plt.plot((0,self._eventsStandard.index[-1]),
                     (0,0), 'k', linewidth=1)
            quartile = self.signalsSpan['1st quartile']
            plt.plot((quartile, quartile), plt.ylim(), color = '#8b0000')
            decile = self.signalsSpan['1st decile']
            plt.plot((decile, decile), plt.ylim(), color = '#ff6600')
            plt.legend()
            quartile = self.signalsSpan['1st quartile']
            plt.plot((quartile, quartile), plt.ylim(), color = '#8b0000')
            decile = self.signalsSpan['1st decile']
            plt.plot((decile, decile), plt.ylim(), color = '#ff6600')
        plt.show()
        
        if stats == True: self.print_stats()
        
    def print_stats(self):
        """
        """
        #print strategy.name        
        
        print('Number of events: %i' %self.eventsNumber)
        print('Lookback: %i' %self.signalsSpan['Lookback'])
        """
        print('10% of new events happen before the period: %i')%self.signalsSpan['1st decile']
        print('25% of new events happen before the period: %i' %self.signalsSpan['1st quartile'])
        print('50% of new events happen before the period: %i' %self.signalsSpan['Median'])
        """        
        print('')
        
        print('Mean of positive Sortino: %0.3f' %self.eventsRelevantData['Mean of positive Sortino'])
        print('Mean of negative Sortino: %0.3f' %self.eventsRelevantData['Mean of negative Sortino'])
        print('Sortino Bias: %0.3f' %self.eventsRelevantData['Sortino Bias'])
        print('System Quality Sortino: %0.3f' %self.eventsRelevantData['SQS'])
        print('')
        
        print('Best Sortino Ratio: %0.3f'%self.eventsRelevantData['Best Sortino ratio'])           
        print('Period with best Sortino Ratio: %i'%self.eventsRelevantData['Period with best Sortino ratio'])           
        print('Period with best Neig_Sortino: %i'%self.eventsRelevantData['Period with best Neig_Sortino'])           
        print('')

        print('Mean of SQNs: %0.2f' %self.eventsRelevantData['Mean of SQNs'])
        print('Mean of positive SQNs: %0.2f' %self.eventsRelevantData['Mean of positive SQNs'])
        print('Mean of negative SQNs: %0.2f' %self.eventsRelevantData['Mean of negative SQNs'])
        print('Sharpe of SQNs: %0.3f' %self.eventsRelevantData['Sharpe of SQNs'])
        try:print('Sharpe of positive SQNs: %0.3f' %self.eventsRelevantData['Sharpe of positive SQNs'])
        except: pass        
        try:print('Sharpe of negative SQNs: %0.3f' %self.eventsRelevantData['Sharpe of negative SQNs'])
        except:pass
    
        print('')
        print('Period with best SQN: %i' %self.eventsRelevantData['Period with best SQN'])
        print('Best SQN: %0.3f' %self.eventsRelevantData['Best SQN'])
        print('')
        print('Period with best Neig_SQN: %i' %self.eventsRelevantData['Period with best Neig_SQN'])
        print('SQN of period with best Neig_SQN: %0.3f' %self.eventsRelevantData['Best Neig_SQN'])
        print('Sharpe of period with best Neig_SQN: %0.3f' %self.eventsRelevantData['Sharpe of period with best Neig_SQN'])
        print('RSQN of period with best Neig_SQN: %0.3f' %self.eventsRelevantData['RSQN of period with best Neig_SQN'])
        print('Mean of period with best Neig_SQN: %0.3f' %self.eventsRelevantData['Mean of period with best Neig_SQN'])
        print('')
        #print('Commission costs')        
        print('Order type: %s' %self.orderType)
        print('')
        
   
if __name__ == "__main__":

    """
    """
    import indicators
    import datetime as dt
    
    endDate = dt.date(2016,12,31)
    startDate = endDate - dt.timedelta(365*5)

    RSI = indicators.RSI('EURCAD', 'H4', startDate=startDate, endDate=endDate)
    RSI.break_level(70)
    
    s = StrategyCreator('EURCAD', 'H4', indicatorsList=[RSI], commission=20)
    s.generate_optimized_event_study(occurrenceNumber=1, lookback=20)




   