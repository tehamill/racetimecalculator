import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import re
from sklearn.model_selection import GridSearchCV


class MergeData(object):
    """merges gpx files (listed in gpx_dict in config file) with main data frame. 
    Output can be sent into final data cleaning"""
    def __init__(self, config,df):
        self.config = config
        self.df = df
        self.merge_with_gpx()
        #self.racesdf = racesdf
        #self.weathers= weather
    
    
    def merge_with_gpx(self):
        import read_gpx
        listofnames = []
        for key,val in self.config.gpx_dict.items():
            thisreadrace = read_gpx.ReadGPX(val)
            thisraceDF= thisreadrace.racesdf
            metersup = "metersup"+key
            std = "std"+key
            listofnames.append(metersup)
            listofnames.append(std)
            thisraceDF.rename(index=str,columns={"metersup":metersup,"std":std},inplace=True)
            self.df = self.df.join(thisraceDF, on='race_location')

        self.df.dropna(how='all',subset=listofnames,inplace=True)
        self.df['metersup'] = self.df[listofnames[0]]
        self.df['std'] = self.df[listofnames[1]]
        for key,val in self.config.gpx_dict.items():
            metersup = "metersup"+key
            std = "std"+key
            self.df['metersup'].fillna(self.df[metersup],inplace=True)
            self.df['std'].fillna(self.df[std],inplace=True)
        self.df = self.df.drop(listofnames,axis=1)
        return
        
