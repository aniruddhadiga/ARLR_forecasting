__version__='1.0.0'
__processor__='forecast_ARLR_exog'

import sys
import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
#import matplotlib as mpl
#import matplotlib.pyplot as plt
#import matplotlib.cm as cm
import epiweeks as epi
import datetime
import argparse
import re
import shutil
import pdb
import configparser
import argparse
from aw_micro import cdc_data

from pandas import Series
from datetime import date, time, datetime, timedelta

from statsmodels.tsa.ar_model import AR

from sklearn.metrics import mean_squared_error

from scipy.linalg import toeplitz
from scipy.stats.distributions import chi2
from scipy import signal

from statsmodels.tsa.stattools import adfuller

from data_prep import data_read_and_prep, get_season, prepdata, prepdata_flux, prep_aw_data

from ARLR_exog import ARLR_regressor,ARLR_exog_module, get_bin

from output_format import outputdistribution_bst, outputdistribution_Gaussker,accu_output
import pkg_resources
import warnings
warnings.filterwarnings('ignore')

import logging
log = logging.getLogger(__processor__)

def get_targets():
    targets = {"wili": "% WEIGHTED ILI", "ili": "%UNWEIGHTED ILI", "ilitotal": "ILITOTAL", "totalpatients": "TOTAL PATIENTS", "flux_wili": "weighted_ili", "flux_ili": "unweighted_ili", "flux_region_type": "region_type", "ili_region_type": "REGION TYPE", "flux_region": "region", "ili_region": "REGION"} #onset and peak week still need to be added
    return targets


def parse_args():
    ap = argparse.ArgumentParser(description='ARLR forecasting method for'
                                 ' state ILI.')
    ap.add_argument('-b', '--forecast_from', required=False,
                    help='a date EW format indicating first week to predict.')
    ap.add_argument('-w', '--weeks', required=True, type=int,
                    help='number of weeks to predict')
    ap.add_argument('--out_state', required=False,
                    help='CSV format output file of state predictions')
    ap.add_argument('--out_county', required=False,
                    help='CSV format output file of county predictions')
    ap.add_argument('--ground_truth', required=False,
                    help='CSV file ("|") from CDC of state ILI levels')
    ap.add_argument('--region_type', required=False,
                    help='national, 1, 2,...,10, state')
    ap.add_argument('--st_fips', required=False,
                    help='file of state fips and names')
    ap.add_argument('--county_ids', required=False,
                    help='file of all county 5-digit fips')
    ap.add_argument('--end_date', required=False,
                    help='date (yyyymmdd) of last ground truth data point')
    ap.add_argument('--CDC', required=True, default=1,
                    help='CDC=0 means no uncertainty binning, CDC=1')
    ap.add_argument('--test', required=False, default=0,
                    help='test mode, dumps the predictions in folder dump')
    ap.add_argument('-v', '--verbose',
                    help='verbose logging', action='store_true')
    ap.add_argument('-l', '--log', default=None,
                    help='log file, by default logs are written to'
                    ' standard output')
    ap.add_argument('-o','--out_folder', required=True, help='CSV format output file of county predictions')
    ap.add_argument('--accu_data', required=False, help='accuweather data stream')
    ap.add_argument('--ght_data', required=False, help='google health trends data stream')


    return ap.parse_args()



def main():
    args = parse_args()
    
    level = logging.INFO
    if args.verbose:
        level = logging.DEBUG
    log.setLevel(level)

    if args.log is None:
        handler = logging.StreamHandler()
    else:
        handler = logging.FileHandler(args.log)

    log_formatter = logging.Formatter('%(asctime)s:%(levelname)s:'
                                      '%(name)s.%(funcName)s:%(message)s',
                                      datefmt='%Y%m%d-%H%M%S')
    handler.setFormatter(log_formatter)
    log.addHandler(handler)

    log.info('{} v{}'.format(__processor__,__version__))    
    
    
    #if args.region not in regions:
    #    raise TypeError("region is not valid")
    #if args.region_type == "national":
    #    args.region_type = "US National"
    fct_weeks = args.weeks
    # 
        
    
    
    csv_path = args.ground_truth
    
    epiyear = args.forecast_from
    startyear = epiyear[:4] #args.forecast_from[:4]
    startweek = epiyear[4:] #args.forecast_from[6:8]
    #trainweek = startweek
    ews = epi.Week(int(startyear), int(startweek))
    targets = get_targets()
    header_region_type = targets['ili_region_type'] #"REGION TYPE" for retro or old datasets
    header_region = targets['ili_region'] #"REGION" for retro or old datasets
    


    fdf = prepdata(csv_path)
    fdf = fdf[~fdf.REGION.isin(['Puerto Rico','Virgin Islands','New York City'])]
    fdf.index = fdf['DATE']
    fdf.index = fdf.index.rename('DATE')

        
    # weather data
    aw_csv_path = args.accu_data#'../data/data-aw-cumulative_20191018_1620-weekly-state.csv'
    
    aw_target = ['temperature_mean', 'AH_mean', 'cloud_cover_mean']#, 'wind_speed_mean']
    df_wtr = prep_aw_data(aw_csv_path)
    df_wtr = df_wtr[~df_wtr.area_id.isin([72,78])]
    df_st_id = pd.read_csv('../data/fips_to_statename.csv')
    df_wtr['REGION'] = df_wtr.apply(lambda row: df_st_id[df_st_id['area_id']==row['area_id']]['REGION'].values[0], axis=1)
    
   
     
    # GHT data
    ght_csv_path = args.ght_data
    df_ght = pd.read_csv(ght_csv_path)
    df_ght.index = df_ght.date
    df_ght.index = df_ght.index.rename('DATE')
    df_ght = df_ght.rename(columns={'state':'REGION'})
    ght_target = ['flu', 'cough', 'fever', 'influenza', 'cold']
    
    allw_lags_f = np.arange(1,53) # should have atleast "ms_fct" lags as we find "ms_fct" filters separately
    if args.ght_data is None and args_accu_data is None:
        targ_dic = {"target" : [targets['ili'],targets['wili']]}
    if args.ght_data is None:
        targ_dict = {"target" : [targets['ili'],targets['wili']], "ght_target" : [], "aw_target" : ['temperature_max', 'temperature_min','temperature_mean', 'RH_max', 'RH_min', 'RH_mean', 'wind_speed_mean','cloud_cover_mean', 'water_total', 'pressure_max', 'pressure_min','pressure_mean', 'AH_max', 'AH_min', 'AH_mean', 'SH_max', 'SH_min','SH_mean', 'ep_st_date',]}#, 'wind_speed_mean']}
    elif args.accu_data is None:
        targ_dict = {"target" : [targets['ili'], targets['wili']], "ght_target" : ['flu', 'cough', 'fever', 'influenza', 'cold']}
    else:
        targ_dict = {"target" : [targets['ili'],targets['wili']], "ght_target" : ['flu', 'cough', 'fever', 'influenza', 'cold'], "aw_target" : ['temperature_max', 'temperature_min',
    'temperature_mean', 'RH_max', 'RH_min', 'RH_mean', 'wind_speed_mean','cloud_cover_mean', 'water_total', 'pressure_max', 'pressure_min','pressure_mean', 'AH_max', 'AH_min', 'AH_mean', 'SH_max', 'SH_min','SH_mean', 'ep_st_date',]}#, 'wind_speed_mean']}
    
    if int(args.test):
        directory = 'dump/'
    if not os.path.exists(directory):
            os.makedirs(directory) 
            
    directory_bst = args.out_folder + 'ARLR_bst/' + str(args.forecast_from)
    directory_Gaussker = args.out_folder + 'ARLR_Gaussker/' + str(args.forecast_from)
    
    if not os.path.exists(directory_bst):
        os.makedirs(directory_bst)
    if not os.path.exists(directory_Gaussker):
        os.makedirs(directory_Gaussker)
    bin_ed = get_bin()

     

    targ_dict = {"target" : [targets['ili'], targets['wili']], "ght_target" : ['flu', 'cough', 'fever', 'influenza', 'cold'], "aw_target" : ['temperature_max', 'temperature_min',
'temperature_mean', 'RH_max', 'RH_min', 'RH_mean', 'wind_speed_mean','cloud_cover_mean', 'water_total', 'pressure_max', 'pressure_min','pressure_mean', 'AH_max', 'AH_min', 'AH_mean', 'SH_max', 'SH_min','SH_mean', 'ep_st_date',]}#, 'wind_speed_mean']}
    targ_dict
    for region in fdf[header_region].unique():
        if fdf[header_region_type].unique() == 'States':
            for v in targ_dict.values():
                if targets['wili'] in v:
                    v.remove(targets['wili'])
        else:
            for v in targ_dict.values():
                if targets['ili'] in v:
                    v.remove(targets['ili'])
            


        df_m  = ARLR_regressor(fdf, df_wtr, df_ght, region, targ_dict, ews)
        yp_fct, bn_mat_bst, bn_mat_Gaussker, seas, lags_app_f, coeffs_f = ARLR_exog_module(fdf, df_wtr, df_ght, region, targ_dict, ews, fct_weeks, allw_lags_f) 

    
        if int(args.CDC):
                #outputdistribution_bst(predictions[0,0:4], bn_mat_bst[0,:,0:4], bin_ed, region, target, directory_bst, ews)
                #outputdistribution_Gaussker(predictions[0,0:4], bn_mat_Gaussker[:,0:4], bin_ed, region, target, directory_Gaussker, ews)
                outputdistribution_fromtemplate(predictions[0,0:4], bn_mat_Gaussker[:,0:4], bin_ed, region, target, directory_Gaussker, ews)
    
    
        if fdf[header_region_type].unique() == 'States':
            print(region)
            accu_output(predictions.reshape(fct_weeks), region,  args.out_state, ews, args.st_fips)

if __name__ == "__main__":
    main()
   
