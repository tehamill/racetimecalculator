class RunConfig:
    RACE_DISTANCE = ['5K','10K','HM','Mar']
    RACE_TYPE = ['Road']
    NUM_RACES_TO_INCLUDE = 3
    RACE_COUNT= 0#1624
    EXCLUDE_ABOVE = 150
    LIST_CATS = ['age_group','race_title']
    DROP_LIST = ['Unnamed: 0','gun_time','terrain','race_data',
                   'meeting_id','uniqueID','position','rb_personal_best','rb_season_best']
    gpx_dict= {'5K':'5k_gpxfiles.csv','10K':'10k_gpxfiles.csv','HM':'HM_gpxfiles.csv','Mar':'Mar_gpxfiles.csv'}
    max10k = 93
    max5k = 37
    maxhm = 195
    maxmar =350
    min_num_races =2
    age_list = ['U9','U11','U13','U14','U15','U17','U18','U20','U23','SEN',
                'V35', 'V40','V45', 'V50','V55','V60','V65','V70','V75','V80','V85','V110','V115']
    first_time_running_race =True
    race_to_predict = 'Mar'

