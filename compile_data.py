import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import re
from sklearn.model_selection import GridSearchCV

class CompileData(object):
    """Takes dataframe of runner info, weather data and elevation data that has one race per row and organizes
    it into one row per runner. Access final data frame as: CompileData.feed_data"""
    def __init__(self, config,df):
        self.config = config
        self.df = df
        self.compile_data()
        #self.racesdf = racesdf
        #self.weathers= weather
    
    
    def compile_data(self):
        #change to numeric
        self.df['rainfall'] = pd.to_numeric(self.df['rainfall'])
        self.df['wind'] = pd.to_numeric(self.df['wind'])
        self.df['temp'] = pd.to_numeric(self.df['temp'])
        self.df['min_time'] = pd.to_numeric(self.df['min_time'])
        self.df['std'] = pd.to_numeric(self.df['std'])
        self.df['metersup'] = pd.to_numeric(self.df['metersup'])
        
        #remove walkers, this helps the fit by removing outliers
        self.df = self.df[~((self.df['race_title'] == '10K') & ( self.df['min_time'] > self.config.max10k))]
        self.df = self.df[~((self.df['race_title'] == 'HM') & ( self.df['min_time'] > self.config.maxhm))]
        self.df = self.df[~((self.df['race_title'] == 'Mar') & ( self.df['min_time'] > self.config.maxmar))]
        self.df = self.df[~((self.df['race_title'] == '5K') & ( self.df['min_time'] > self.config.max5k))]
        
        #label encode age, rename
        self.clean_age(self.config.age_list)
        self.df = self.df.drop(["age_group"],axis=1)
        self.df["age_group"] = self.df['ages']
        self.df = self.df.drop(["ages"],axis=1)
        
        #Keep runners who run >2 races       
        countts = self.df['name'].value_counts()
        self.df = self.df[self.df['name'].isin(countts.index[(countts > self.config.min_num_races )])]
        
        #feed_data is data frame ready to be fed into stepwise feature selection
        #it has the runner name, past race averages (10K average time, 10K average upward climb, 5K average time, etc)
        #as well as the time of the current race you're predicting, which is called "min_time"
        self.feed_data = self.stack_runners()
        #print(self.feed_data)
        return
        
        
        
    def stack_runners(self):
        #final data here holds the race you're predicting the time for, rest of data will be added to it
        final_data = self.df[~self.df.duplicated(['name'],keep='last')]

        #df_previous_races is the previous races' information--these are features and have to be averaged so that
        #each runner has same number of features
        df_previous_races = self.df[self.df.duplicated(['name'],keep='last')]
        df_previous_races= df_previous_races.drop(['age_group'],axis=1)

        #eda first not used right now. to get avg time between races use first race date -last race date/num races
        df_first = self.df[~self.df.duplicated(['name'],keep='first')]

        #the race before the one we're predicting, not used right now
        df_last_race = df_previous_races[~df_previous_races.duplicated(['name'],keep='last')]

        df_first = df_first.drop(['age_group','race_title','sex','race_location','min_time',
                                    'temp','race_city','lat,lon','rainfall','wind','metersup','std'],axis=1)

        df_first.rename(index=str,columns={"name":"name","race_date":"first_date"},inplace=True)


        #get first date into final data to get avg time btwn races
        final_data = final_data.join(df_first.set_index('name'), on='name')

        #get total num races ran
        num_races = df_previous_races.groupby(['name']).count().reset_index()
        num_races = num_races.drop(['sex','race_date','race_location','min_time','temp',
                                   'race_city','lat,lon','rainfall','wind','metersup','std'],axis=1)
        num_races.rename(index=str,columns={"name":"name","race_title":"num_races"},inplace=True)

        #join num races ran to main dataframe
        final_data = final_data.join(num_races.set_index('name'), on='name')

        #next few lines unused
        final_data["race_date"] = pd.to_datetime(final_data["race_date"])
        final_data["first_date"] = pd.to_datetime(final_data["first_date"])

        final_data['time_btwn'] = (final_data["race_date"] - final_data["first_date"]).dt.days#) / final_data[num_races]
        final_data['avg_time_btwn'] = final_data['time_btwn'] / final_data['num_races']
        final_data = final_data.drop(['time_btwn','first_date'],axis=1)


        #avgtimes,std, metersup, temps, wind, and rain for each race type (5k, 10k, HM, Mar)
        avgtimes = df_previous_races.groupby(['name','race_title']).mean().reset_index()

        #STACK INTO ONE RUNNER PER ROW. 
        times = avgtimes.set_index(['name','race_title']).unstack()


        #number of times ran previous races
        racecounts= df_previous_races.groupby(['name','race_title']).count().reset_index()
        racecounts = racecounts.drop(['sex','race_date','min_time'],axis=1)


        racecounts = racecounts.drop(['race_city','lat,lon','rainfall','temp','wind','metersup','std'],axis=1)
        racecounts.rename(index=str,columns={"name":"name","race_title":"race_title","race_location":"num_races"},inplace=True)

        #stack into one runner per row
        rcounts = racecounts.set_index(['name','race_title']).unstack()


        rcounts= rcounts.fillna(0)
        final_data = final_data.join(times, on='name')
        final_data = final_data.join(rcounts, on='name')
        model_data = final_data.drop(['race_location','avg_time_btwn'],axis=1)

        model_data_stacked = self.clean_cat_vars(model_data,['sex'])

        thisrace = self.config.race_to_predict


        feed_data = model_data_stacked[model_data_stacked['race_title']==thisrace]
        if self.config.first_time_running_race == True:
            feed_data = feed_data[feed_data[('num_races', thisrace)] == 0.0]    
            
        #feed_data = feed_data[feed_data[('num_races', '5K')] == 0.0]
        feed_data=feed_data.drop(['race_date','name','race_title','lat,lon','race_city'],axis=1)
        feed_data = feed_data.fillna(0.0)
        #feed_data.to_csv('fd.csv')1
        #self.feed_data = feed_data
        return feed_data
        
    def clean_age(self,age_list):
        """label encode age_group and keep order of age_list--should have it do this given U<SEN<V"""
        self.df['age_group'].fillna("SEN",inplace=True)
        self.df['ages'] = self.df.age_group.apply(lambda x: self.config.age_list.index(x) )
        return

    def clean_cat_vars(self, data, list_cats):
        """takes ;data; dataframe a list of categorical variables in data to be
        one-hot encoded so they can be used in a random forest regression"""
        for var in list_cats:
            data[var].fillna("Missing",inplace=True)
            dummies = pd.get_dummies(data[var],prefix = var)
            data = pd.concat([data,dummies],axis=1)
            data.drop([var],axis=1,inplace=True)
        return data
    
