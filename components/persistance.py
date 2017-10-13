# -*- coding: utf-8 -*-
"""
Created on Mon Jan 09 11:20:47 2017

@author: forex
"""

from __future__ import division
from __future__ import print_function

import _pickle as pk
from os.path import join


class Pickler(object):
    def __init__(self, name=None):
        self._name = name
        
    
    def save(self, obj, path, name=None):
        """
        Main method.
        """
        if name: 
            self._name = name
        
        with open(join(path, self._name), 'wb') as outputFile:
            pk.dump(obj, outputFile)
            
            
    def load(self, path, name=None):
        if name: 
            self._name = name
        
        with open(join(path, self._name), 'rb') as outputFile:
            return pk.load(outputFile)
        
        

if __name__ == '__main__':

    name = 'prueba'
    path = 'walkforwards'
    
    pickler = Pickler()
    pickler.save(lightWf.data, path, name)
    
    prueba = pickler.load(path, name)
    
    
    