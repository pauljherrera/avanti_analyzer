# -*- coding: utf-8 -*-
"""
Created on Mon Jan 09 09:51:15 2017

@author: forex
"""

from __future__ import division
from __future__ import print_function

from copy import copy

from .persistance import Pickler



class WalkforwardPersistanceBuilder(object):
    """
    From a walkforward, builds an object as light as possible to be saved
    or viewed.
    It was impossible that the built object have the same interface
    as a walkforward. 
    """
    def __init__(self, walkforward=None, persistor=Pickler()):
        
        if walkforward:
            self.backtest = copy(walkforward.backtestFinal.backtest)
            self.stats = copy(walkforward.backtestFinal.stats)
            
            self.calendar = copy(walkforward.calendar)
            self.performanceMatrices = [copy(x.performanceMatrix) for x 
                                            in walkforward.optimizations]

        self._persistor = persistor
        
    def save(self, path, name):
        toSave = {k:v for k,v in self.__dict__.items() 
                  if not k.startswith('_')}
        toSave.update({'name': name})
        self._persistor.save(toSave, path, name)
                        
    def load(self, path, name):
        self.__dict__.update(self._persistor.load(path, name))
        
    
if __name__ == '__main__':
    
    name = 'prueba'
    path = 'walkforwards'
    
    lightWf = WalkforwardPersistanceBuilder(wf)
    lightWf.save(path, name)
    lightWf.load(path, name)
    