__version__='1.1.0'
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

from data_prep import data_read_and_prep, get_season, prepdata, prepdata_flux, prep_aw_data, prep_ght_data, prepdata_append, prepdata_retro
from ARLR_exog import ARLR_regressor,ARLR_exog_module, get_bin

from output_format import outputdistribution_bst, outputdistribution_Gaussker,accu_output, outputdistribution_fromtemplate, outputdistribution_fromtemplate_for_FSN, outputdistribution_fromtemplate_for_FluSight, outputdistribution_state_fromtemplate

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
    
    ap.add_argument('--accu_data_nat', required=False, help='accuweather data stream')
    ap.add_argument('--accu_data_hhs', required=False, help='accuweather data stream')
    ap.add_argument('--accu_data_state', required=False, help='accuweather data stream')

    ap.add_argument('--ght_data_nat', required=False, help='google health trends data stream')
    ap.add_argument('--ght_data_hhs', required=False, help='google health trends data stream')
    ap.add_argument('--ght_data_state', required=False, help='google health trends data stream')
    ap.add_argument('--sub_date', required=False, help='Submission date for FluSight output file, if not mentioned automatically computed to Monday date')
    ap.add_argument('--eval', required=False, help='evaluation mode for determing accuracy of forecasts, expects the input data frame to contain data for full season') 

    return ap.parse_args()



def main():
    config = configparser.ConfigParser()
    config_file = pkg_resources.resource_filename(__name__, 'config.ini')
    config.read(config_file)
    
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
    st_id_path = args.st_fips
    
    epiyear = args.forecast_from
    startyear = epiyear[:4] #args.forecast_from[:4]
    startweek = epiyear[4:] #args.forecast_from[6:8]
    #trainweek = startweek
    ews = epi.Week(int(startyear), int(startweek))
    targets = get_targets()
    header_region_type = targets['flux_region_type'] #"REGION TYPE" for retro or old datasets
    header_region = targets['flux_region'] #"REGION" for retro or old datasets
    

    end_date = args.end_date
    fdf = prepdata_flux(csv_path, ews)
    fdf = fdf.rename(columns={'REGION TYPE': 'region_type', 'REGION': 'region', '% WEIGHTED ILI': 'weighted_ili', '%UNWEIGHTED ILI': 'unweighted_ili', 'DATE':'date'})
    if end_date is None:
        end_date = fdf['date'].max().date() + timedelta(days=3)
    else:
        dt = datetime.strptime(end_date,'%Y%m%d').date()
        end_date = dt + timedelta(days = (3 - dt.isoweekday()%7))
    if args.end_date is not None:
        fdf = fdf[fdf['date'] <= pd.Timestamp(end_date)] 
    fdf = fdf[~fdf.region.isin(['Puerto Rico','Virgin Islands','New York City'])]
    fdf.index = fdf['date']
    fdf.index = fdf.index.rename('date')
   
    # DataFrame preparation part, integrating accuweather, ght time series with ILI
    kwargs_wtr = {"National": args.accu_data_nat, "HHS": args.accu_data_hhs, "States": args.accu_data_state}
    accu_data_fl = None
    for _,value in kwargs_wtr.items():
        accu_data_fl = accu_data_fl or value 
    kwargs_ght = {"National": args.ght_data_nat, "HHS": args.ght_data_hhs, "States": args.ght_data_state}
    ght_data_fl = None
    for _,value in kwargs_ght.items():
        ght_data_fl = ght_data_fl or value
    
    if ght_data_fl is None and accu_data_fl is None:
        df_ght = pd.DataFrame()
        df_wtr = pd.DataFrame()
        targ_dict = {"target" : [targets['flux_ili'],targets['flux_wili']],"ght_target" : [], "aw_target" : []}
    elif ght_data_fl is None and accu_data_fl is not None:
        df_ght = pd.DataFrame()
        targ_dict = {"target" : [targets['ili'],targets['wili']], "ght_target" : [], "aw_target" : ['temperature_max', 'temperature_min','temperature_mean', 'RH_max', 'RH_min', 'RH_mean', 'wind_speed_mean','cloud_cover_mean', 'water_total', 'pressure_max', 'pressure_min','pressure_mean', 'AH_max', 'AH_min', 'AH_mean', 'SH_max', 'SH_min']}#, 'wind_speed_mean']}
        #aw_csv_path = args.accu_data#'../data/data-aw-cumulative_20191018_1620-weekly-state.csv'
     
        df_wtr = prep_aw_data(st_id_path, **kwargs_wtr)
        
    elif accu_data_fl is None and ght_data_fl is not None:
        df_wtr = pd.DataFrame()
        targ_dict = {"target" : [targets['ili'], targets['wili']], "ght_target" : ['flu', 'cough', 'fever', 'influenza', 'cold'], "aw_target" : []}
        #ght_csv_path = args.ght_data
        df_ght = prep_ght_data(**kwargs_ght)
        #df_ght.index = df_ght.date
        #df_ght.index = df_ght.index.rename('DATE')
        #df_ght = df_ght.rename(columns={'state':'REGION'})
        ght_target = ['flu', 'cough', 'fever', 'influenza', 'cold']

    else:
        targ_dict = {"target" : [targets['flux_ili'],targets['flux_wili']], "ght_target" : ['flu', 'cough', 'fever', 'influenza', 'cold'], "aw_target" : ['temperature_max', 'temperature_min','temperature_mean', 'RH_max', 'RH_min', 'RH_mean', 'wind_speed_mean','cloud_cover_mean', 'water_total', 'pressure_max', 'pressure_min','pressure_mean', 'AH_max', 'AH_min', 'AH_mean', 'SH_max', 'SH_min']}#, 'wind_speed_mean']}
        # weather data
        #aw_csv_path = args.accu_data
        df_wtr = prep_aw_data(st_id_path, **kwargs_wtr)
        
        # GHT data
        #ght_csv_path = args.ght_data
        
        df_ght = prep_ght_data(**kwargs_ght)
        #df_ght.index = df_ght.date
        #df_ght.index = df_ght.index.rename('DATE')
        #df_ght = df_ght.rename(columns={'state':'REGION'})
        ght_target = ['flu', 'cough', 'fever', 'influenza', 'cold']
    
           
    directory_bst = args.out_folder + 'ARLR_bst/'# + str(args.forecast_from[:4])
    directory_Gaussker = args.out_folder + 'ARLR_Gaussker/'# + str(args.forecast_from[:4])
    
    if not os.path.exists(directory_bst):
        os.makedirs(directory_bst)
    if not os.path.exists(directory_Gaussker):
        os.makedirs(directory_Gaussker)
    bin_ed = get_bin()

    allw_lags_f = np.arange(1,55) # should have atleast "ms_fct" lags as we find "ms_fct" filters separately

    #targ_dict = {"target" : [targets['ili'], targets['wili']], "ght_target" : ['flu', 'cough', 'fever', 'influenza', 'cold'], "aw_target" : ['temperature_max', 'temperature_min','temperature_mean', 'RH_max', 'RH_min', 'RH_mean', 'wind_speed_mean','cloud_cover_mean', 'water_total', 'pressure_max', 'pressure_min','pressure_mean', 'AH_max', 'AH_min', 'AH_mean', 'SH_max', 'SH_min']}#, 'wind_speed_mean']}
    if args.sub_date is not None:
        sub_date = args.sub_date
    else:
        sub_date = ((ews+1).enddate()+timedelta(days=2)).isoformat() #submission for epiweek N is (epiweek N+1).enddate() + timedelta(days=2)
    df_full_res = pd.DataFrame(columns=['DATE','location', '1 week ahead', '2 week ahead', '3 week ahead', '4 week ahead', targ_dict['target'][0]])
    df_full_res = df_full_res.set_index('DATE')
    df_full_seas = pd.DataFrame(columns=['season', 'location'])
    idx = [(ews+i).startdate() for i in range(1, len(range((ews.week)-40,35)))]
    df_full_seas = pd.DataFrame(columns=['season', 'location'])
    df_full_seas['DATE'] = idx

    for region in fdf[header_region].unique():
        df_res = pd.DataFrame(columns=['DATE','location', '1 week ahead', '2 week ahead', '3 week ahead', '4 week ahead', targ_dict['target'][0]])
        idx = [(ews+i).startdate() for i in range((ews.week-40+1),35)]
        df_res['DATE'] = idx
        df_res = df_res.set_index('DATE')

        targ_dict['target'] = [targets['flux_ili'], targets['flux_wili']]
        if fdf[header_region_type][fdf[header_region]==region].unique() == 'States':
            print(region)
            for v in targ_dict.values():
                if targets['flux_wili'] in v:
                    v.remove(targets['flux_wili'])
        else:
            for v in targ_dict.values():
                if targets['flux_ili'] in v:
                    v.remove(targets['flux_ili'])
        
        win = int(config['Forecasting']['win']) # training window
        max_lag = np.max(allw_lags_f) # maximum lag considered in the model
        # Check if datastream has no missing information for all lagged regressors of length equal to training length window  
        nan_chk_mask = (fdf[header_region]==region)&(fdf.index<=pd.to_datetime(ews.startdate()))&(fdf.index>=pd.to_datetime((ews-int(win+max_lag)).startdate())) 
        if fdf[nan_chk_mask][targ_dict['target']].isna().values.any():
            print('Missing values in ILI data, cannot produce forecasts')
            continue    
        df_m  = ARLR_regressor(fdf, df_wtr, df_ght, region, targ_dict, ews)
        predictions, bn_mat_bst, bn_mat_Gaussker, seas, lags_app_f, coeffs_f = ARLR_exog_module(df_m, targ_dict, ews, fct_weeks, allw_lags_f) 
        for i in range(1,len(predictions[0,:])+1):
            print('Week: {}, Fct: {}'.format(i,(predictions[0,i-1])))
            df_res.loc[(ews+i).startdate(), 'location'] = region
            df_res.loc[(ews+i).startdate(), '{} week ahead'.format(i)] = predictions[0,i-1]
            if args.eval is not None:
                df_res.loc[(ews+i).startdate(), targ_dict['target']] = fdf[(fdf.index==pd.to_datetime((ews+i).startdate())) &(fdf[header_region]==region)][targ_dict['target']].values[0]
        idx = [(ews+i).startdate() for i in range(1, len(range((ews.week)-40,35)))]
        df_seas = pd.DataFrame(columns=['season', 'location'])
        df_seas['DATE'] = idx
        df_seas['location'] = df_seas.apply(lambda x: region, axis=1)
        df_seas.loc[:,'season'] = seas
        df_seas = df_seas.set_index('DATE')
        #df_res = df_res.merge(df_seas, how='outer', left_index=True, right_index=True)
        
        df_full_res = df_full_res.append(df_res)    
        df_full_seas = df_full_seas.append(df_seas)
        if int(args.CDC) and fdf[header_region_type][fdf[header_region]==region].unique() != 'States':
            target = targets['flux_wili'] 
                #outputdistribution_bst(predictions[0,0:4], bn_mat_bst[0,:,0:4], bin_ed, region, target, directory_bst, ews)
                #outputdistribution_Gaussker(predictions[0,0:4], bn_mat_Gaussker[:,0:4], bin_ed, region, target, directory_Gaussker, ews)
            outputdistribution_fromtemplate_for_FSN(predictions[0,0:4], bn_mat_Gaussker[0,:,0:4], bin_ed, region, target, directory_Gaussker, ews)
            #outputdistribution_fromtemplate_for_FluSight(predictions[0,0:4], bn_mat_Gaussker[0,:,0:4], bin_ed, region, target, directory_Gaussker, ews, sub_date)

 
   
        if fdf[header_region_type][fdf[header_region]==region].unique() == 'States':
            target = targets['flux_ili'] 
            accu_output(predictions.reshape(fct_weeks), region,  args.out_state, ews, args.st_fips)
            outputdistribution_state_fromtemplate(predictions[0,0:4], bn_mat_Gaussker[0,:,0:4], bin_ed, region, target, directory_Gaussker, ews, sub_date)
    df_full_res.to_csv('result_'+str(ews.year) + 'EW' + str(ews.week))
    df_full_seas.to_csv('result_seas_'+str(ews.year) + 'EW' + str(ews.week))
if __name__ == "__main__":
    main()
   
