# -*- coding: utf-8 -*-
"""
Created on Fri Jul 01 12:37:23 2016

@author: forex
"""

import numpy as np
cimport cython
from cython.parallel import prange

cdef double cmax(list a):
    cdef double[999] lista
    cdef int i, length = len(a)
    for i in range(length): lista[i] = a[i]    
    cdef double x = lista[0]
    for i in range(length):
            if lista[i] > x: x = lista[i]
    return x
    
cdef double cmin(list a):
    cdef double[999] lista
    cdef int i, length = len(a)
    for i in range(length): lista[i] = a[i]    
    cdef double x = lista[0]
    for i in range(length):
        if lista[i] < x: x = lista[i]
    return x

def cDiv(object df, int KDPeriod, int MinPeriod):
    cdef int i, j, lenM    
    cdef double x    
   
    cdef double[:] cMomentum
    Momentum = np.array(df["Momentum"])
    cMomentum = Momentum
    lenMom = len(Momentum)
    lenM = lenMom
  
    cdef long long[:] cClose
    Close = np.array(df["Close"])
    cClose = Close  
    
    RSI = list(df["RSI"])
    High = list(df["High"])
    Low = list(df["Low"])
    
    cdef int[:] a
    Div = np.empty_like(Close, dtype=int)
    Div[:] = 0
    a = Div        
 
    #Iteration for each bar
    for i in range(KDPeriod+1,lenM):
        #Another iteration to check for divergence
        for j in range(MinPeriod,KDPeriod+1):
            if (cMomentum[i] < cMomentum[i-j]):
                if (cClose[i] > cClose[i-j]):
                    if High[i] >= max(High[i-j:i]):
                        if len(filter(lambda x: x > 70, RSI[i-j:i+1])) > 0:
                            Div[i] = -1
                            break
            elif (cMomentum[i] > cMomentum[i-j]):
                if (cClose[i] < cClose[i-j]):
                    if Low[i] <= min(Low[i-j:i]):
                        if len(filter(lambda x: x < 30, RSI[i-j:i+1])) > 0:
                            Div[i] = 1
                            break
    
    df["Knoxpy"] = Div          
    