import numpy as np
import pandas as pd
import re


class ReadGPX(object):
    def __init__(self,gpx_list_file):
        self.gpx_list_df = pd.read_csv(gpx_list_file)
        self.racesdf = self.extract_features()
        self.racesdf = self.racesdf.drop(['lat','lon','metersdown'],axis=1)


        
        
    def read_gpx(self,gpx_file):
        """read gps extract (lat, long, elevation) as tuple of lists of floats"""
        lat = []
        lon = []
        ele = []
        #print('here')
        with open(gpx_file,'r') as file:
            for line in file:
                if "<trkpt lat" in line:
                    thislat, thislon = re.findall(r'[-+]?\d*\.\d+|\d+',line)
                    lat.append(float(thislat))
                    lon.append(float(thislon))
                elif "<ele>" in line:
                    thisele = re.findall(r'[-+]?\d*\.\d+|\d+',line)
                    #print("thisline",line,"=== ",thisele[0])
                    ele.append(float(thisele[0]))


        return (lat,lon,ele)


    def extract_features(self):
        racesdf = pd.DataFrame(columns=['lat','lon','metersup','metersdown','std'], index=self.gpx_list_df['race_location'])
        for race_location in self.gpx_list_df['race_location']:
            gpxfile = '/Users/teh33/letsdothis/gpxfiles/'+self.gpx_list_df[self.gpx_list_df['race_location']==race_location]['gpxfile'].iloc[0]       
            #gpxfile2 = self.gpx_list[self.race_list.index(race_location)]   
            lat,lon,ele = self.read_gpx(gpxfile)
            mean_lat = np.mean(lat)
            mean_lon = np.mean(lon)
            std = np.std(ele)
            ele_diff = [ele[n]-ele[n-1] for n in range(1,len(ele))]
            meters_up = 0.0
            meters_down = 0.0
            for i in ele_diff:
                if i > 0:
                    meters_up += i
                elif i <0:
                    meters_down += i
            racesdf.loc[race_location] = pd.Series({'lat':round(mean_lat),'lon':round(mean_lon),
                                                  'metersup':meters_up,'metersdown':meters_down,'std':std})
        return racesdf


