import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import re
from sklearn.model_selection import GridSearchCV



class RunRaceModel(object):
    """Takes dataframe from CompileData.feed_data and selects relevant features and runs random forest regression
    outputs: RunRaceModel.model (trained random forest model), RunRaceModel.xtrain,RunRaceModel.ytrain,
    RunRaceModel.xtest,RunRaceModel.ytest, RunRaceModel.explainer (LimeTabularExplainer see lime docs for how to run)"""
    def __init__(self, config,feed_data):
        self.config = config
        self.feed_data = feed_data
     
    def select_features(self):
        """Selects features using forward selection. Access model & data selected by using self.model (trained random forest model), self.Xtrain, self.ytrain, self.Xtest, self.ytest
           self.model should then be sent through self.run_model() to run grid search etc"""
        
        features_list = list(self.feed_data.columns.values)
        features_list.remove("min_time")
        thisrace = self.config.race_to_predict

        #if never ran race before, don't include these variables in feature
        #selection, they're just 0's anyway
        if self.config.first_time_running_race == True:
            unuseable_columns = [('min_time', thisrace),('std', thisrace),('num_races', thisrace),
                ('rainfall', thisrace),
                ('temp', thisrace),
                ('wind', thisrace),
                ('metersup', thisrace),    
                             'sex_W']
        else:
            #drop this column...probs should have removed it earlier. 
            unuseable_columns = ['sex_W']
        #print(features_list)
        for element in unuseable_columns:
            features_list.remove(element)
        data_with_all_feats = self.feed_data.drop(unuseable_columns,axis=1)
        colstodrop = features_list
        thiscols = []
        data_with_current_feats = data_with_all_feats.drop(features_list,axis=1)
        checkfit=100.0
        scores = []
        dropped_cols = []
        loopgain =True
        #mymod = RandomForestRegressor(n_estimators=80, oob_score = True, max_depth=10,
        #                                      min_samples_split = 25, criterion='mse')
        thisloopfeatures_list = features_list
        curcols = data_with_current_feats.columns
        countgain=0
        #print("cc",curcols)
        while loopgain == True:
            thisloopscore=100.0
            for fet in thisloopfeatures_list:
                data_with_current_feats[fet] = data_with_all_feats[fet]
                etrain=data_with_current_feats.sample(frac=0.8,random_state=200)
                etest=data_with_current_feats.drop(etrain.index)
                y = etrain.pop('min_time')
                ytest = etest.pop('min_time')
                #print(y)
                model = RandomForestRegressor(n_estimators=80, oob_score = True, max_depth=15,
                                              min_samples_split = 12, criterion='mse')
                model.fit(etrain,y)

                PRED = model.predict(etrain)
                predscore = self.mean_absolute_percentage_error(y,PRED)#= r2_score(y,PRED)
                oobs = self.mean_absolute_percentage_error(y,model.oob_prediction_)
                scores.append(oobs)
                if ((thisloopscore - oobs) > 0.0):
                    thisloopscore = oobs
                    fetwinner = fet
                data_with_current_feats.drop(fet,axis=1,inplace=True)
                etrain.drop(fet,axis=1,inplace=True)

            data_with_current_feats[fetwinner] = data_with_all_feats[fetwinner]
            etrain=data_with_current_feats.sample(frac=0.8,random_state=200)
            etest=data_with_current_feats.drop(etrain.index)
            y = etrain.pop('min_time')
            ytest = etest.pop('min_time')
            #print(y)
            model = RandomForestRegressor(n_estimators=80, oob_score = True, max_depth=30,
                                              min_samples_split = 12,min_samples_leaf =7, criterion='mse')
            model.fit(etrain,y)

            PRED = model.predict(etrain)
            predscore = self.mean_absolute_percentage_error(y,PRED)#= r2_score(y,PRED)
            #print(fetwinner,predscore)
            oobs = self.mean_absolute_percentage_error(y,model.oob_prediction_)
            scores.append(oobs)
            #print(fetwinner,"~",oobs)
            thisloopfeatures_list.remove(fetwinner)
            if ((checkfit-oobs)>0.0001):
                checkfit = oobs
                curcols = data_with_current_feats.columns
                #print(curcols)
            else:
                break


        self.final_df = self.feed_data[data_with_current_feats.columns]
        self.Xtrain=self.final_df.sample(frac=0.8,random_state=200)
        self.Xtest=self.final_df.drop(self.Xtrain.index)#
        self.ytrain = self.Xtrain.pop('min_time')
        self.ytest = self.Xtest.pop('min_time')
        self.model=  RandomForestRegressor(n_estimators=80, oob_score = True, max_depth=30,
                                              min_samples_split = 12,min_samples_leaf =7, criterion='mse')
        self.model.fit(self.Xtrain,self.ytrain)
        #print(y)
        return
    
    def run_model(self):
        """Runs grid search. After running run_model(), self.model with hold best estimator from gridsearch"""
        param_grid = { 
            'n_estimators': [30, 60],
            'max_features': ['sqrt', 'log2'],
            'max_depth':[5,38],
             'min_samples_split' : [12,40],
            'min_samples_leaf' :[12,40], 
        }

        CV_rfc = GridSearchCV(estimator=self.model, param_grid=param_grid, cv= 7)
        CV_rfc.fit(self.Xtrain, self.ytrain)
        #print(CV_rfc.best_params_)
        self.model = CV_rfc.best_estimator_
        return 
    
    def mean_absolute_percentage_error(self,y_true, y_pred): 
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
