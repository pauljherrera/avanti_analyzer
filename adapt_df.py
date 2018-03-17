# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 17:13:03 2018

@author: arnaldo
"""
import pandas as pd
import datetime as dt
import os

def merge_adapt_df(a,b):
    """
    a is the oldest
    b is the newest
    """
    c = pd.concat([a,b])
    c['Date'] = [dt.datetime.strptime(d,"%d.%m.%Y %H:%M:%S.%f").date() for d in c['Gmt time']]
    c['Time'] = [dt.datetime.strptime(d,"%d.%m.%Y %H:%M:%S.%f").time() for d in c['Gmt time']]
    del c['Gmt time']
    c = c.rename(columns={"Volume": "Vol"})
    c = c[["Date","Time","Open","High","Low","Close","Vol"]]
    return c

def merge_mixed_dfs(a,b):
    """
    a oldest in dukas format
    b is the newest alredy formated
    """
    a['Date'] = [dt.datetime.strptime(d,"%d.%m.%Y %H:%M:%S.%f").date() for d in a['Gmt time']]
    a['Time'] = [dt.datetime.strptime(d,"%d.%m.%Y %H:%M:%S.%f").time() for d in a['Gmt time']]
    del a['Gmt time']
    a = a.rename(columns={"Volume": "Vol"})
    a = a[["Date","Time","Open","High","Low","Close","Vol"]]
    
    c = pd.concat([a,b])
    return c
    

def differents():
    path = "C:/Users/arnal/Desktop/csvsss/"
    a = "AUDJPY_Candlestick_5_m_BID_04.03.2008-04.03.2013.csv"
    b = "AUDJPY_5_UTC+0_00_noweekends.csv"
    items = a.split("_")
    name = items[0]
    tf = items[2]
    a = pd.read_csv(path+a)
    b = pd.read_csv(path+b)
    c = merge_mixed_dfs(a,b)
    csvname = "listos/{}_{}_UTC+0_00_noweekends.csv".format(name,tf)
    c.to_csv(path+csvname,index=False)
    

def equals():
    path = "C:/Users/arnal/Desktop/csvsss/"
    a = "AUDJPY_Candlestick_5_m_BID_04.03.2008-04.03.2013.csv"
    b = "AUDJPY_Candlestick_5_m_BID_05.03.2013-08.03.2018.csv"
    items = a.split("_")
    name = items[0]
    tf = items[2]
    a = pd.read_csv(path+a)
    b = pd.read_csv(path+b)
    c = merge_adapt_df(a,b)
    csvname = "{}_{}_UTC+0_00_noweekends.csv".format(name,tf)
    c.to_csv(path+csvname,index=False)
    
def round_dfs():
    path = "C:/Users/arnal/Desktop/csvsss/listos2"
    files = os.listdir(path)
    for file in files:
        df = pd.read_csv(path+"/"+file)
        df = df.round(4)
        df.to_csv(path+"/"+file, header=False,index=False)

if __name__ == "__main__":
    round_dfs()