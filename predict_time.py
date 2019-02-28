import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import re
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import merge_data
import compile_data
import run_race_model
        

class PredictTime(object):
    """Get model that predicts time. Optionally save plots. highest level class that calls RunRaceModel. 
     member variables: PredictTime.model (trained random forest model), PredictTime.xtrain,PredictTime.ytrain,
    PredictTime.xtest,PredictTime.ytest, PredictTime.explainer (LimeTabularExplainer see lime docs for how to run)
    to make a prediction call PredictTime.model.predict(data)
    data should have same format as PredictTime.model.xtrain"""
    def __init__(self, config):
        self.config = config
                
    def clean_data(self):
        """Given datafile in RunConfig, this merges that dataframe with the elevation data. Then it stacks the dataframe so that each runner is one row of the data frame (previous racetimes, weather, and elevation are averaged so as to occupy one cell in the runner's row for each race type (5k, 10k, HM, Mar))""" 

        #read in data. this is race time finisher lists combined with weather data
        race_results = pd.read_csv(self.config.datafile)

        #include elevation data
        merge_race_data_with_gpx = merge_data.MergeData(self.config,race_results)
        race_results=merge_race_data_with_gpx.df
        
        #compile data holds dataframe in compile_data.feed_data
        #"CompileData" brings finisher list where each row is a single time
        #for one runner and stacks it so that each runner occupies one row
        #which includes their average times (5k,10k,HM,Mar) among other variables 
        compiled_data = compile_data.CompileData(self.config,race_results)
        self.all_data = compiled_data.feed_data 

        
    def train_model(self):
        #class to train a random forest model
        runracemodel = run_race_model.RunRaceModel(self.config,self.all_data)

        #selects features using forward selection algorithm
        runracemodel.select_features()

        #trains random forest model, uses gridsearch to tune hyperparameters 
        runracemodel.run_model()

        #make easily accessible
        self.model=runracemodel.model
        self.Xtrain = runracemodel.Xtrain
        self.Xtest=runracemodel.Xtest
        self.ytrain = runracemodel.ytrain
        self.ytest=runracemodel.ytest


        #initialize lime---need to find categorical indices
        import lime
        import lime.lime_tabular
        cat_index = []
        for catind in range(len(self.Xtrain.columns)):
            if list(self.Xtrain.columns.values)[catind] =="sex_M":
                cat_index.append(catind)
            elif list(self.Xtrain.columns.values)[catind] == "age_group":
                cat_index.append(catind)
        self.explainer = lime.lime_tabular.LimeTabularExplainer(self.Xtrain.values, 
                                                           feature_names=list(self.Xtrain.columns), 
                                                           class_names=['min_time'], 
                                                           categorical_features=[4,6],
                                                           verbose=True, mode='regression',
                                                           discretize_continuous=True)
        

        #print model performance
        self.trainpred = self.model.predict(self.Xtrain)
        self.testpred = self.model.predict(self.Xtest)
        print("MAPE for test set = ",self.mean_absolute_percentage_error(self.ytest,self.testpred))
        print("MAPE for train set = ",self.mean_absolute_percentage_error(self.ytrain,self.trainpred))
    
        
        return

    def make_plots(self):
        fig = plt.figure(figsize=[20, 16.81])
        ax = fig.gca()
        #plt.plot(range(len(ytest))[:100],ytest[:100],label='Race Times')
        plt.scatter(self.ytrain,self.trainpred,label ='Training Set')
        plt.scatter(self.ytest,self.testpred,label ='Validation set')
        #plt.scatter(Eytest,Averages,label ='Predicted Race times')
        #plt.plot(range(150,350),range(150,350))
        ax.set_ylabel('PREDICTED MARATHON TIME (MINUTES)', fontsize=36)
        ax.set_xlabel('ACTUAL MARATHON TIME (MINUTES)', fontsize=36)
        ax.legend(loc='best', fontsize=36)
        plt.setp(ax.get_xticklabels(), fontsize=24)
        plt.setp(ax.get_yticklabels(), fontsize=24)
        #plt.ylim(150,350)
        fig.savefig(self.config.race_to_predict+'_pred_v_actual.png', dpi=fig.dpi)
    
        fig = plt.figure(figsize=[20, 16.81])
        ax = fig.gca()

        feature_importances = pd.Series(self.model.feature_importances_,index=self.Xtrain.columns)
        a = feature_importances.nlargest(11)
        a.plot(kind="barh",figsize=(16,15))

        plt.setp(ax.get_yticklabels(), fontsize=24)
        plt.setp(ax.get_xticklabels(), fontsize=24)
        fig.savefig(self.config.race_to_predict+'_feature_importances.png', dpi=fig.dpi)

    
    def mean_absolute_percentage_error(self,y_true, y_pred): 
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
