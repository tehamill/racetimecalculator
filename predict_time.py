import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import re
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt


class PredictTime(object):
    """Get model that predicts time. Optionally save plots. highest level class that calls RunRaceModel 
     outputs: PredictTime.model (trained random forest model), PredictTime.xtrain,PredictTime.ytrain,
    PredictTime.xtest,PredictTime.ytest, PredictTime.explainer (LimeTabularExplainer see lime docs for how to run)
    to make a prediction call PredictTime.model.predict(data)
    data should have same format as PredictTime.model.xtrain"""
    def __init__(self, config,datafile):
        self.config = config
        self.datafile = datafile
        
        
    def run_model(self):
        import merge_data
        import compile_data
        import run_race_model
        eda = pd.read_csv(self.datafile)

        eda = eda.drop(['Unnamed: 0','Unnamed: 0.1','chip_time','gun_time','position','rb_personal_best',
               'rb_season_best','race_data','uniqueID','chip_time_corr','gun_time_corr','meeting_id','terrain'],axis=1)
        eda.replace('--',0.0,inplace=True)

        md = merge_data.MergeData(self.config,eda)
        eda=md.df
        
        #compile data holds dataframe in compile_data.feed_data
        compiled_data = compile_data.CompileData(self.config,eda)
        runracemodel = run_race_model.RunRaceModel(self.config,compiled_data.feed_data)
        runracemodel.select_features()
        runracemodel.run_model()
        self.model=runracemodel.model
        self.Xtrain = runracemodel.Xtrain
        self.Xtest=runracemodel.Xtest
        self.ytrain = runracemodel.ytrain
        self.ytest=runracemodel.ytest
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
