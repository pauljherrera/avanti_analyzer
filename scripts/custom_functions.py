# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 07:56:38 2016

@author: forex
"""

from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import datetime as dt
import pandas as pd
import os
import cPickle as pickle
from pygame import mixer


def adapt_to_timeframe(df, time1,time2):
    """
    Adapts a df 'Date', 'Time' and 'Datetime' to a higher timeframe.
    """
    # Creates a copy of the original datetima to preserve this values.
    df['Datetime_original'] = df.Datetime
    
    # Function calls for each case.
    if time2 == 'D1':
        df = any_to_d1(df)
        
    if (time1 == 'H1') and (time2 == 'H4'):
        df = h1_to_h4(df)
                
    if time1 == 'M15':
        if time2 == 'H1':
            df = m15_to_h1(df)
        if time2 == 'H4':
            df = m15_to_h1(df)
            df = h1_to_h4(df)
            
    if time1 == 'M5':
        if time2 == 'M15':
            df = m5_to_m15(df)
        if time2 == 'H1':
            df = m5_to_m15(df)
            df = m15_to_h1(df)
        if time2 == 'H4':
            df = m5_to_m15(df)
            df = m15_to_h1(df)
            df = h1_to_h4(df)
            
    if time1 == 'M1':
        if time2 == 'M5':
            df = m1_to_m5(df)
        if time2 == 'M15':
            df = m1_to_m5(df)
            df = m5_to_m15(df)
        if time2 == 'H1':
            df = m1_to_m5(df)
            df = m5_to_m15(df)
            df = m15_to_h1(df)
        if time2 == 'H4':
            df = m1_to_m5(df)
            df = m5_to_m15(df)
            df = m15_to_h1(df)
            df = h1_to_h4(df)

    # Creates a new datetime column with the adapted Date and Time.
    df['Datetime'] = map(lambda x: pd.Timestamp(str(x[0])+ " " + str(x[1])), 
                                zip(df.Date, df.Time))
    
    return df


def any_to_d1(df):
    df = df[df.Date != df.Date.shift(-1)]
    df.Time = [dt.time(0,0)] * len(df.Time)
    
    return df
    

def change_time(Series, toChangeList, desiredValue):
    for i in toChangeList:
        Series[Series == i] = desiredValue
    
    return Series
    

def check_attr(name, attr, value):
    if not hasattr(name, '%s' %attr):
        setattr(name, '%s' %attr, value)
    else: pass


def clean_panel(panel):
    """
    Cleans a panel of the Nan preserving as much as the 3D data as possible. 
    """
    #
    shapes = []
    for item in panel.items:
        df = panel[item].dropna(axis=1, how='all').dropna()
        shapes.append(df.shape)
        
    # Looking for best shape according to COLUMN numbers
    bestSize = 0
    bestShape1 = (0,0)
    for s in shapes:
        size = s[0] * s[1] *\
               len([x for x in [x[1] for x in shapes] if x >= s[1]])
        if size > bestSize:
            bestSize = size
            bestShape1 = s
    
    # Selecting which shapes to delete.
    forDelete = []
    for i in xrange(len(shapes)):
        if shapes[i][1] < bestShape1[1]:
            forDelete.append(i)
    
    forDelete.reverse()

    # Deleting shapes.
    for i in forDelete:
        del shapes[i]
        
    # Looking for best shape according to ROW numbers
    bestSize = 0
    bestShape2 = (0,0)
    for s in shapes:
        size = s[0] * s[1] *\
               len([x for x in [x[0] for x in shapes] if x >= s[0]])
        if size > bestSize:
            bestSize = size
            bestShape2 = s
            
    # Setting definitive shape.
    bestShape = (min([bestShape1[0], bestShape2[0]]),
                 min([bestShape1[1], bestShape2[1]]))
            
    # Creating the cleansed panel.
    items = {}
    for item in panel.items:
        df = panel[item].dropna(axis=1, how='all').dropna()
        if df.shape[1] >= bestShape[1]:
            items[item] = df[:bestShape[0]]
    
    cleanPanel = pd.Panel.from_dict(items, intersect=True)
    
    return cleanPanel


def h1_to_h4(df):
    
    Series = df.Time.values
    array = np.zeros([len(Series),2])
    for n in xrange(len(Series)):
        array[n][0] = Series[n].hour
        array[n][1] = Series[n].minute

    # Main algorithm in Numpy
    condList = [np.in1d(array[:,0], [0,1,2,3]), np.in1d(array[:,0], [4,5,6,7]),
                np.in1d(array[:,0], [8,9,10,11]), 
                np.in1d(array[:,0], [12,13,14,15]),
                np.in1d(array[:,0], [16,17,18,19]), 
                np.in1d(array[:,0], [20,21,22,23])]
    choiceList = [0,4,8,12,16,20]
    array[:,0] = np.select(condList, choiceList)
        
    # Converting back the 2d array into a dt.time Series.
    df.Time = [dt.time(int(x[0]), int(x[1])) for x in array]

    # Leaving only one copy of each time.
    df = df[df.Time != df.Time.shift(-1)]

    return df


def load_pickles(path='', verbose = True): # Testing if path='' works.
    """
    """
    # Making a list of file names to unpickle.   
    files = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        files.extend(filenames)
        break
    
    #Loading the pickles.
    pickles = []
    for name in files:
        if verbose == True:
            print('Unpickling %s' %name)
        pickles.append(pickle.load(open(os.path.join(path, name))))

    return pickles
    

def m1_to_m5(df):
    array = time_values_to_numpy(df.Time.values)
        
    # Main algorithm in Numpy
    condList = [np.in1d(array[:,1], [0,1,2,3,4]), 
                np.in1d(array[:,1], [5,6,7,8,9]),
                np.in1d(array[:,1], [10,11,12,13,14]), 
                np.in1d(array[:,1], [15,16,17,18,19]),
                np.in1d(array[:,1], [20,21,22,23,24]),
                np.in1d(array[:,1], [25,26,27,28,29]),
                np.in1d(array[:,1], [30,31,32,33,34]),
                np.in1d(array[:,1], [35,36,37,38,39]),
                np.in1d(array[:,1], [40,41,42,43,44]),
                np.in1d(array[:,1], [45,46,47,48,49]),
                np.in1d(array[:,1], [50,51,52,53,54]),
                np.in1d(array[:,1], [55,56,57,58,59])]
    choiceList = [0,5,10,15,20,25,30,35,40,45,50,55]
    array[:,1] = np.select(condList, choiceList)

    # Converting back the 2d array into a dt.time Series.
    df.Time = [dt.time(int(x[0]), int(x[1])) for x in array]
    
    # Leaving only one copy of each time.
    df = df[df.Time != df.Time.shift(-1)]

    return df


def m5_to_m15(df):
    array = time_values_to_numpy(df.Time.values)
        
    # Main algorithm in Numpy
    condList = [np.in1d(array[:,1], [0,5,10]), np.in1d(array[:,1], [15,20,25]),
                np.in1d(array[:,1], [30,35,40]), np.in1d(array[:,1], [45,50,55])]
    choiceList = [0,15,30,45]
    array[:,1] = np.select(condList, choiceList)

    # Converting back the 2d array into a dt.time Series.
    df.Time = [dt.time(int(x[0]), int(x[1])) for x in array]
    
    # Leaving only one copy of each time.
    df = df[df.Time != df.Time.shift(-1)]
 
    return df


def m15_to_h1(df):
    array = time_values_to_numpy(df.Time.values)
    
    # Main function in Numpy.
    array[:,1] = 0

    # Converting back the 2d array into a dt.time Series.
    df.Time = [dt.time(int(x[0]), int(x[1])) for x in array]

    # Leaving only one copy of each time.
    df = df[df.Time != df.Time.shift(-1)]

    return df
    
    
def max_argmax_df(df, ordinal=1):
    """
    Returns the max value of a DataFrame and the location of that value 
    in the DataFrame.
    If the ordinal is 2, it returns the second max, and so on.
    """
    # Adjusting the df for ordinal different than 1.
    i = 1    
    while i < ordinal:
        df = df[df != max(df.max())]
        i += 1
    
    # Main part of the algorithm. Looking max and argmax.        
    maximum = max(df.max())
    col = df.max().idxmax()
    idx = df.idxmax()[col]
    
    return maximum, (idx,col)
    
    
def max_argmax_panel(panel):
    """
    Returns the max value of a Panel and the location of that value 
    in the Panel
    """
    maximum = panel.max().max().max()
    item = panel.max().max().idxmax()
    col = panel[item].max().idxmax()
    idx = panel[item].idxmax()[col]
    
    return maximum, (item,idx,col)
    
    
def min_argmin_df(df):
    """
    Returns the min value of a DataFrame and the location of that value 
    in the DataFrame
    """
    minimum = min(df.min())
    col = df.min().idxmin()
    idx = df.idxmin()[col]
    
    return minimum, (idx,col)
    
    
def min_argmin_panel(panel):
    """
    Returns the max value of a Panel and the location of that value 
    in the Panel
    """
    minimum = panel.min().min().min()
    item = panel.min().min().idxmin()
    col = panel[item].min().idxmin()
    idx = panel[item].idxmin()[col]
    
    return minimum, (item,idx,col)
    
    
def notification(path='sounds', name='sound01.mp3'):
    """
    Plays a notification sound using the pygame mixer.
    """
    mixer.init()
    mixer.music.load(os.path.join(path, name))
    mixer.music.play()    
    
    
def plot_matrix(matrix, method='lines', view=(5,-25)):
    """
    Plots according to various methods.
    
    Methods:
    wireframe -
    lines -
    surface - Cuts the performance matrix according to the Series with 
                less values.
    """
    if method=='wireframe':
        X, Y = np.meshgrid(matrix.index, matrix.columns)
        Z = matrix.values
        
        plot = plt.figure(figsize=(9.5,9.5)).gca(projection='3d')
        plot.plot_wireframe(X.T,Y.T, Z, rstride=0, cstride=1)
        
        plot.set_xlabel('Periods')
        plot.set_ylabel('Sample')
        plot.set_zlabel('SQN')
        
        plot.view_init(view[0], view[1])
        plt.show()
        
    elif method=='lines':
        fig = plt.figure(figsize=(9.5,9.5))
        ax = Axes3D(fig)
        
        x = matrix.index
        a = matrix.index[0]
        b = matrix.index[-1]
        c = matrix.columns[0]
        d = matrix.columns[-1]
        
        for col in matrix.columns:
            y = matrix[col]
            ax.plot(xs=x, ys=[col]*len(x), zs=y, zdir='z', alpha=0.7)
        
        # Plotting a rectangle in the 0,0 plane.
        ax.plot_surface([[a, a],[b, b]], 
                        [[c, d],[c, d]], 
                        0, rstride=1, cstride=1, linewidth=0, alpha=0.3)
                        
        plt.yticks(matrix.columns)
        ax.set_xlabel('Periods')
        ax.set_ylabel('Sample')
        ax.set_zlabel('SQN')
        
        ax.view_init(view[0], view[1])
        plt.show()
        
    elif method=='surface':
        plot_surface(matrix, view=view)


def plot_surface(df, view=(45,45), dropna=True):
    """
    Plots a dataframe as a surface.
    """
    if dropna == True: df = df.dropna()
    X, Y = np.meshgrid(df.index, df.columns)
    Z = df.values
    fig = plt.figure() 
    ax = Axes3D(fig)
    ax.plot_surface(X.T,Y.T,Z, rstride=1, cstride=1, alpha=0.4)
    ax.view_init(view[0], view[1])
    plt.yticks(df.columns)
    

def round_to_multiple(num, multiple):
    rounder_down = num % multiple
    rounder_up = multiple - (num % multiple)
    if rounder_down < rounder_up:
        return num - rounder_down
    else:
        return num + rounder_up
    
def time_values_to_numpy(Series):
    # Converting the time column in a 2dimensional array with hour and minutes that can be processed by Cython
    array = np.zeros([len(Series),2])
    for n in xrange(len(Series)):
        array[n][0] = Series[n].hour
        array[n][1] = Series[n].minute

    return array

    
    
        