# -*- coding: utf-8 -*-
"""
Created on Mon Jan 02 17:47:06 2017

@author: forex
"""

from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn
import datetime as dt
from copy import copy
from pprint import pprint





class WalkforwardVisitor(object):
    
    def __init__(self, reporter):
        self._reporter = reporter
    
    def visit(self, visited):
        self._reporter.show_report(visited)    
        

class LoopVisitor(WalkforwardVisitor):

    def visit(self, visited):
        for i in visited:
            self._reporter.show_report(i)     
    
    
    
    
    
    