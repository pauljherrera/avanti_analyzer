# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 08:20:04 2016

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
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import seaborn



class PerformanceVisitor(object):
    """
    Visits an Optimization instance and creates a performance matrix. 
    """
    
    def visit(self, visited):
        """
        Main method. Visits an Optimization instance.
        """
        raise NotImplementedError('To be implemented')
        

class SortinoPerformanceVisitor(PerformanceVisitor):
    """
    Visits an Optimization instance and creates a performance matrix based on
    the Sortino Ratio
    """
    
    def visit(self, visited):
        """
        Main method. Visits an Optimization instance.
        """
        raise NotImplementedError('To be implemented')


class SortinoTwoVariablesVisitor(SortinoPerformanceVisitor):
    
    def visit(self,visited):

        #Initialize the matrix.
        rows = set([int(x.split(',')[0].split('_')[1]) 
                   if str(x.split(',')[0].split('_')[1]).isdigit()
                   else x.split(',')[0].split('_')[1] 
                   for x in visited.optimization.keys()])
                       
        cols = set([int(x.split(',')[1].split('_')[1]) 
                   if str(x.split(',')[1].split('_')[1]).isdigit()
                   else x.split(',')[1].split('_')[1]
                   for x in visited.optimization.keys()])
                       
        df = pd.DataFrame(index = rows, columns = cols)
        df.sort_index(inplace=True)
        df.sort_index(axis=1, inplace=True)
        
        # Setting the variable names as column and index names.
        df.index.set_names(list(visited.optimization.keys())[0].split(',')[0].split('_')[0], 
                           inplace=True)
        df.columns.set_names(list(visited.optimization.keys())[0].split(',')[1].split('_')[0], 
                             inplace=True)
        
        #Populate the performance matrix.
        for key, value in list(visited.optimization.items()):
            x = int(key.split(',')[0].split('_')[1])\
                if str(key.split(',')[0].split('_')[1]).isdigit()\
                else key.split(',')[0].split('_')[1] 
            y = int(key.split(',')[1].split('_')[1])\
                if str(key.split(',')[1].split('_')[1]).isdigit()\
                else key.split(',')[1].split('_')[1]
            df.set_value(x, y, value.eventsRelevantData['SQS'])
        
        #Assign performance matrix.
        visited.performanceMatrix = df
        
        
class SortinoThreeVariablesVisitor(SortinoPerformanceVisitor):
    
    def visit(self,visited):
        
        #Initialize the performance panel.
        axis = set([x.split(',')[0].split('-')[1] 
                    for x in visited.optimization.keys()])
        axis2 = set([x.split(',')[1].split('-')[1] 
                     for x in visited.optimization.keys()])
        axis3 = set([x.split(',')[2].split('-')[1] 
                     for x in visited.optimization.keys()])
        panel = pd.Panel(items = axis, major_axis = axis2, minor_axis = axis3)
        
        # Setting the variable names as column and index names.
        panel.items.set_names(visited.optimization.keys()[0].split(',')[0].split('-')[0], inplace=True)
        panel.major_axis.set_names(visited.optimization.keys()[0].split(',')[1].split('-')[0], inplace=True)
        panel.minor_axis.set_names(visited.optimization.keys()[0].split(',')[2].split('-')[0], inplace=True)
        
        # Populate the performance panel.
        for key, value in visited.optimization.iteritems():
            panel.set_value(key.split(',')[0].split('-')[1], 
                            key.split(',')[1].split('-')[1], 
                            key.split(',')[2].split('-')[1], 
                            value.eventsRelevantData['SQS'])
        
        #Assign performance matrix.
        visited.performancePanel = panel
        




