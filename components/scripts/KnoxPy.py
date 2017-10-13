# -*- coding: utf-8 -*-
"""
Knoxpy

Versión 1.0

Copyright 2015 - Paúl Herrera.

 Versión 1.2 Cambia el período del mínimo o máximo local. Anteriormente 
se usaba un número fijo de 30, ahora depende del período en que se esté 
evaluando la divergencia.

"""

from __future__ import print_function


import numpy as np
import pandas as pd
import time
#import cdiv

from tqdm import tqdm


      
t0 = time.time()



 
#Setting functions -----------------------------------------------------------

#Momentum Indicator as a class  
def MOM(df, MomentumPeriod):
    M = pd.Series((df['Close']/(df['Close'] - 
                    df['Close'].diff(MomentumPeriod)))*100)  
    df["Momentum"] = M.round(4)
    return df
        
#Relative Strength Index  in two functions

def relative_strength(prices, RSIPeriod):

    deltas = np.diff(prices)
    seed = deltas[:RSIPeriod+1]
    up = seed[seed >= 0].sum()/RSIPeriod
    down = -seed[seed < 0].sum()/RSIPeriod
    rs = up/down
    rsi = np.zeros_like(prices, dtype=float)
    rsi[:RSIPeriod] = 100. - 100./(1. + rs)

    for i in range(RSIPeriod, len(prices)):
        delta = deltas[i - 1]  # cause the diff is 1 shorter

        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up*(RSIPeriod - 1) + upval)/RSIPeriod
        down = (down*(RSIPeriod - 1) + downval)/RSIPeriod

        rs = up/down
        rsi[i] = 100. - 100./(1. + rs)
        
    return rsi.round(4)
    
def RSIndex(df,RSIPeriod):
    prices = df["Close"]
    df["RSI"] = relative_strength(prices,RSIPeriod)
    return df
         
#Knoxville Divergence Detection

def pyDiv(df, KDPeriod, MinPeriod):
    Momentum = list(df["Momentum"])
    Close = list(df["Close"])
    RSI = list(df["RSI"])
    High = list(df["High"])
    Low = list(df["Low"])
    Div = np.empty_like(Close, dtype=int)
    Div[:] = 0
    Iteration = range(KDPeriod+1,len(Momentum))
    PeriodIter = range(MinPeriod,KDPeriod+1)
    
    #Iteration for each bar
    for i in Iteration:
        #Another iteration to check for divergence
        for j in PeriodIter:
            if (Momentum[i] < Momentum[i-j]):
                if (Close[i] > Close[i-j]):
                    if High[i] >= max(High[i-j:i]):
                        if len(list(filter(lambda x: x > 70, RSI[i-j:i+1]))) > 0:
                            Div[i] = -1
                            break
            elif (Momentum[i] > Momentum[i-j]):
                if (Close[i] < Close[i-j]):
                    if Low[i] <= min(Low[i-j:i]):
                        if len(list(filter(lambda x: x < 30, RSI[i-j:i+1]))) > 0:
                            Div[i] = 1
                            break
    
    df["Knoxpy"] = Div
    return df          


def Knoxpy(DataFile, lookback=30, verbose=True):
    """
    Wrapper for external calls
    """
    t998324756767 = time.time()    
    
    #Importing csv into a DataFrame
    t1273648195 = time.time()
    histdataheader = ["Date","Time","Open","High","Low","Close","Vol"]
    df = pd.read_csv( DataFile, header=None)
    df.columns = histdataheader
    df.set_index(['Date', 'Time'], inplace=True)
    del df['Vol']
    if verbose == True:
        print ("Csv read in %r seconds" % round((time.time() - t1273648195), 2))
       
    #Calling functions
    
    t1273648195 = time.time()
    MOM(df, MomentumPeriod)
    if verbose == True:
        print("Momentum analized in %r seconds" % round((time.time() - t1273648195), 2))
    
    t1273648195 = time.time()
    RSIndex(df, RSIPeriod)
    if verbose == True:
        print("RSI analized in %r seconds" % round((time.time() - t1273648195), 2))

  
    t1273648195 = time.time()
    pyDiv(df, KDPeriod=lookback, MinPeriod=4)
    cdiv.cDiv(df, KDPeriod=lookback, MinPeriod=4)
    
#    kd_generator(df, KDPeriod=lookback, MinPeriod=4)
    if verbose == True:
        print("Knoxpyed in %r seconds" % round((time.time() - t1273648195), 2))

    #Dropping NANs (Deprecated because caused problems with the backtester)
    #df = df.dropna()
    
    if verbose == True:
        print("Total time was %r seconds" % round((time.time() - t998324756767), 2))

    return df
    
def Knoxpy_df(df, lookback=30, verbose=False):
    t998324756767 = time.time()    
    
    #Calling functions
    
    t1273648195 = time.time()
    MOM(df, MomentumPeriod)
    if verbose == True:
        print ("Momentum analized in %r seconds" % round((
                                            time.time() - t1273648195), 2))
    
    t1273648195 = time.time()
    RSIndex(df, RSIPeriod)
    if verbose == True:
        print ("RSI analized in %r seconds" % round((
                                            time.time() - t1273648195), 2))

  
    t1273648195 = time.time()
    pyDiv(df, lookback, MinPeriod=4)
    cdiv.cDiv(df, lookback, MinPeriod=4)
    if verbose == True:
        print ("Knoxpyed in %r seconds" % round((
                                            time.time() - t1273648195), 2))

    #Dropping NANs
    df = df.dropna()
    
    if verbose == True:
        print ("Total time was %r seconds" % round((
                                        time.time() - t998324756767), 2))

    return df


def Knoxpy_df2(prices, lookback=30, verbose=False):
    """
    Input: a 'prices' DataFrame (OHLC)
    
    Output: a 'df' DataFrame with the column 'Knoxpy' with values 1, 0 or 1 for
    bullish divergence, no divergence and bearish divergence.
    
    Arguments:
        1- lookback.
        2- verbose: if true, prints the time elapsed for each step of the
        function.
    """
    t998324756767 = time.time()    
    
    #Calling functions
    
    t1273648195 = time.time()
    MOM(prices, MomentumPeriod=20)
    if verbose == True:
        print ("Momentum analized in %r seconds" % round((
                                            time.time() - t1273648195), 2))
    
    t1273648195 = time.time()
    RSIndex(prices, RSIPeriod=21)
    if verbose == True:
        print ("RSI analized in %r seconds" % round((
                                            time.time() - t1273648195), 2))
    
    t1273648194 = time.time()
    pyDiv(prices, lookback, MinPeriod=4)
#    cdiv.cDiv(prices, lookback, MinPeriod=4)
    if verbose == True:
        print ("Knoxpyfied in %r seconds" % round((
                                            time.time() - t1273648194), 2))

    #creating df
    df = pd.DataFrame(index=prices.index)
    df['Knoxpy'] = prices['Knoxpy']
    
    #Dropping columns and NANs
    del prices['Momentum']
    del prices['RSI']
    del prices['Knoxpy']
    df = df.dropna()
    
    if verbose == True:
        print ("Total time was %r seconds" % round((
                                        time.time() - t998324756767), 2))

    return df


def kd_generator(df, KDPeriod=30, MinPeriod=4):
    """
    KD calculator using separate loops for RSI and Momentum divergence.
    """
    df['n'] = range(len(df))
#    Iteration = xrange(KDPeriod+1,len(df))
    
#    for i in tqdm(Iteration):        
#        # Make lists of OB/OS periods.
#        iterdf = df[i-KDPeriod:i]
#        overboughts = iterdf.n[(iterdf.RSI > 70) & (iterdf.n < i-MinPeriod)]
#        oversolds = iterdf.n[(iterdf.RSI < 30) & (iterdf.n < i-MinPeriod)]
#        lowerHighs = iterdf.n[iterdf.High < iterdf.High.iloc[-1]]
#        higherLows = iterdf.n[iterdf.Low > iterdf.Low.iloc[-1]]
#        bearCandidates = [x for x in overboughts if x in lowerHighs.values]
#        bullCandidates = [x for x in oversolds if x in higherLows.values]
#        
#        #This horrible one-liner checks divergences.
#        KD = 1 if iterdf.n[iterdf.n.isin(bullCandidates)\
#                           & (iterdf.Close > iterdf.Close.iloc[-1])\
#                           & (iterdf.Momentum < iterdf.Momentum.iloc[-1])].any()\
#             else -1 if iterdf.n[iterdf.n.isin(bearCandidates)\
#                           & (iterdf.Close < iterdf.Close.iloc[-1])\
#                           & (iterdf.Momentum > iterdf.Momentum.iloc[-1])].any()\
#             else 0
#        
#        # Assigning the presence of KD 
#        df.set_value(df.iloc[i].name, 'KD', KD)
#
#    # Deleting unnecessary rows.
#    del df['Momentum']
#    del df['RSI']
#    del df['n']
    
    
    def foo(series):
        i = int(series.n)
        if not i % 100:
            print(i)
        if i >= KDPeriod:
            iterdf = df[i - KDPeriod:i]
            overboughts = iterdf.n[(iterdf.RSI > 70) & (iterdf.n < i-MinPeriod)]
            oversolds = iterdf.n[(iterdf.RSI < 30) & (iterdf.n < i-MinPeriod)]
            lowerHighs = iterdf.n[iterdf.High < iterdf.High.iloc[-1]]
            higherLows = iterdf.n[iterdf.Low > iterdf.Low.iloc[-1]]
            bearCandidates = [x for x in overboughts if x in lowerHighs.values]
            bullCandidates = [x for x in oversolds if x in higherLows.values]
            
            #This horrible one-liner checks divergences.
            KD = 1 if iterdf.n[iterdf.n.isin(bullCandidates)\
                               & (iterdf.Close > iterdf.Close.iloc[-1])\
                               & (iterdf.Momentum < iterdf.Momentum.iloc[-1])].any()\
                 else -1 if iterdf.n[iterdf.n.isin(bearCandidates)\
                               & (iterdf.Close < iterdf.Close.iloc[-1])\
                               & (iterdf.Momentum > iterdf.Momentum.iloc[-1])].any()\
                 else 0
                 
            return KD
        
    
    df['KD'] = df.apply(foo, axis=1)
              

 #Running all functions for the DataFrame ------------------------------------

if __name__ == "__main__":
    #Inputs-----------------------------------------------------------------------
    DataFile = "../../csv/EURUSD_M15_UTC+0_00_noweekends.csv"
    MomentumPeriod = 20
    RSIPeriod = 21
    KDPeriod = 30    #This is the Knoxville Divergence Period
    MinPeriod = 4

#Importing csv into a DataFrame
    t1273648195 = time.time()
    histdataheader = ["Date","Time","Open","High","Low","Close","Vol"]
    df = pd.read_csv( DataFile, header=None)
    df.columns = histdataheader
    df.set_index(['Date', 'Time'], inplace=True)
    del df['Vol']
    
    df = Knoxpy_df2(df, KDPeriod, verbose=True)


                        
