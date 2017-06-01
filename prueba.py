# -*- coding: utf-8 -*-
"""
Created on Mon Feb 06 15:22:49 2017

@author: forex
"""

from pandas import read_csv
import gc

#@profile
def memory_check():
    with read_csv('csv/AUDCAD_M1_UTC+0_00_noweekends.csv', header=None) as df:
        df.columns = ['d','t','o','h','l','c','v']

        foo = df.loc[:,'d'].iloc[0]
        print foo
        
    print 'hola'
    
    
if __name__ == '__main__':
    memory_check()
    
    