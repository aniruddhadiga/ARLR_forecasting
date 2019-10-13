__version__='1.0.0'
__processor__='forecast_ARLR'

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

from pandas import Series
from datetime import date, time, datetime, timedelta

from statsmodels.tsa.ar_model import AR

from sklearn.metrics import mean_squared_error

from scipy.linalg import toeplitz
from scipy.stats.distributions import chi2
from scipy import signal

from statsmodels.tsa.stattools import adfuller
from data_prep import data_read_and_prep, get_season, prepdata, prepdata_retro
from ARLR import ARLR_aug_phase, ARLR_red_phase, ARLR_fct, ARLR_err_met, fct_uncert, uncer_scr,multi_step_fct, ARLR_model, outputdistribution_bst, outputdistribution_Gaussker,accu_output
import pkg_resources
import warnings
warnings.filterwarnings('ignore')

import logging
log = logging.getLogger(__processor__)

def national():
    cdcdf = pd.read_csv('data/national/ILINet.csv', header=1)
    df = cdcdf.drop(["REGION", "REGION TYPE", "AGE 0-4", "AGE 25-49", "AGE 25-64", "AGE 5-24", "AGE 50-64", "AGE 65", "NUM. OF PROVIDERS"], axis=1)
    
    df['DATE'] = pd.to_datetime(df.apply(lambda row : epi.Week(int(row["YEAR"]), int(row["WEEK"])).startdate() ,axis=1, result_type='reduce'))
    
    return df

def region(number):
    cdcdf = pd.read_csv('data/regional/ILINet.csv', header=1)
    cdcdf.drop(["REGION TYPE", "AGE 0-4", "AGE 25-49", "AGE 25-64", "AGE 5-24", "AGE 50-64", "AGE 65", "NUM. OF PROVIDERS"], axis=1, inplace = True)
    dfs = {}
    for region in cdcdf["REGION"].unique():
        dfs[region] = pd.DataFrame(cdcdf.loc[cdcdf['REGION'] == region])
        
    for df in dfs.values():
        df.drop(["REGION"], axis=1, inplace=True)
        df['DATE'] = pd.to_datetime(df.apply(lambda row : epi.Week(int(row["YEAR"]), int(row["WEEK"])).startdate() ,axis=1, result_type='reduce'))
        #df.drop(["YEAR", "WEEK"], axis = 1, inplace = True)
    return dfs["Region " + number]

def state(name):
    cdcdf = pd.read_csv('data/state/ILINet.csv', header=1)
    cdcdf = cdcdf.drop(["REGION TYPE", "AGE 0-4", "AGE 25-49", "AGE 25-64", "AGE 5-24", "AGE 50-64", "AGE 65", "NUM. OF PROVIDERS"], axis=1)
    dfs = {}

    for state in cdcdf["REGION"].unique():
        dfs[state] = pd.DataFrame(cdcdf.loc[cdcdf["REGION"] == state])


    for df in dfs.values():
        df.drop(["REGION"], axis=1, inplace=True)
        df['DATE'] = pd.to_datetime(df.apply(lambda row : epi.Week(int(row["YEAR"]), int(row["WEEK"])).startdate() ,axis=1, result_type='reduce'))
        #df.drop(["YEAR", "WEEK"], axis = 1, inplace = True)
    
    return dfs[name]

def get_regions():
    regions = {"national": national, "1": region, "2": region, "3": region, "4": region, "5": region, "6": region, "7": region, "8": region, "9": region, "10": region, "Alabama": state, "Alaska": state, "Arizona": state, "Arkansas": state, "California": state, "Colorado": state, "Connecticut": state, "Delaware": state, "Georgia": state, "Hawaii": state, "Idaho": state, "Illinois": state, "Indiana": state, "Iowa": state, "Kansas": state, "Kentucky": state, "Louisiana": state, "Maine": state, "Maryland": state, "Massachusetts": state, "Michigan": state, "Minnesota": state, "Mississippi": state, "Missouri": state, "Montana": state, "Nebraska": state, "Nevada": state, "New Hampshire": state, "New Jersey": state, "New Mexico": state, "New York": state, "North Carolina": state, "North Dakota": state, "Ohio": state, "Oklahoma": state, "Oregon": state, "Pennsylvania": state, "Rhode Island": state, "South Carolina": state, "South Dakota": state, "Tennessee": state, "Texas": state, "Utah": state, "Vermont": state, "Virginia": state, "Washington": state, "West Virginia": state, "Wisconsin": state, "Wyoming": state}
    return regions

def get_targets():
    targets = {"wili": "% WEIGHTED ILI", "ili": "%UNWEIGHTED ILI", "ilitotal": "ILITOTAL", "totalpatients": "TOTAL PATIENTS"} #onset and peak week still need to be added
    return targets

def get_bin():    
    bin_ed= [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 7.0, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 8.0, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 9.0, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9, 10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9, 11.0, 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 11.8, 11.9, 12.0, 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7, 12.8, 12.9, 13.0, 100]
    return bin_ed
 

def ARLR_module(df, region, target, epi_week, fct_weeks):
    config = configparser.ConfigParser()
    config_file = pkg_resources.resource_filename(__name__, 'config.ini')
    config.read(config_file) 
    ews_train = epi_week-1
    ews_test = epi_week   
    df_train = df[(df['DATE']<=pd.to_datetime(ews_train.startdate()))]
    df_train[target] = np.array(df_train[target],float)
    df_train[target] = df_train[target].replace(0,1e-2) # check if zeros are there in ILI data as we take log
    df_test = df[(df['DATE']>=pd.to_datetime(ews_train.startdate()))]
    df_test[target] = np.array(df_test[target],float)
    df_test[target] = df_test[target].replace(0,1e-2)
    #targetdf = Series(df[target])
    #target_series = targetdf[:starttraining_date]
    #df_train = target_series[:-1]
    #df_test = target_series[-1:]
    train = np.log(np.array(df_train[target],'float').astype(float))
    test = np.log(np.array(df_test[target],'float').astype(float))
    train = pd.Series(train)
    train.index = df_train['DATE']
    test = pd.Series(test)
    test.index = df_test['DATE']
    # Multi-step forecast
    
    
    win = int(config['Forecasting']['win']) # Length of the historial training data to be considered
    
    fut_wks = int(config['Forecasting']['fut_wks']) # Number of weeks ahead to forecast from training data 
    ms_fct = fct_weeks # For every forecast week, give additional ms_fct weeks forecast
    
    test_win = fut_wks+ms_fct # Number of true value to be fetched (testing accuracy)
    exp_max_lags =  int(config['Forecasting']['exp_max_lags'])# expected maximum lags to be considered in the model
    llr_tol=1e-2 # log-likelihood tolerance
    
    # Uncertainty analysis
    uncer_anl = int(config['CDC']['uncer_anl'])
    Nb = int(config['CDC']['Nb'])
    # create bins
    n_bins=int(config['CDC']['n_bins'])
    
    #bin_ed = np.arange(0,n_bins,.1)
    #bin_ed = np.append(bin_ed,20)
    bin_ed = get_bin()
    
    # Read csv file and create train and test data
    # dates = pd.DatetimeIndex(df_train["DATE"])
    #plt.figure(figsize=(12,7))
    #plt.subplot(2,1,1);plt.plot(train.index,(train));plt.title('Full training data from specified epiweek {}, {}'.format(epwk,yr))
    # plt.subplot(2,1,2);plt.plot((hist_win(train,win)).index,(hist_win(train,win)));plt.title('Training data: 4 year period')
    
    # Check data for stationarity in the training data with padding
    train_win = train[-1:(-win-exp_max_lags-1):-1] # training samples in the window period + buffer
    
    
    #result = adfuller(train_win)
    #print(result)
    #if result[1] < 0.05:
    #    print('p-val of ADF test %e' %result[1])
    #    print('Stationary signal')
    # plt.plot(train_win)
    # Check seasonality
    #season_ind = get_season(train_win,fft_len=1024,figs=False)
    # train the model
    max_lags = 108
    coeffs=np.zeros([ms_fct,max_lags])
    train_pred_err=np.zeros([ms_fct, win])
    yp_train=np.zeros([ms_fct, win]) 
    lags_app=np.zeros([ms_fct,max_lags])
    # Train to obtain ARLR coeffs for all specified multi-step forecast:
    # Ex: For 1-step forecast, consider data from t-1 to t-p for training: ms_fct = 1
    # for 4-step forecast, consider data for t-4 to t-p for training: ms_fct = 4 
    # similarly for 1 season, ms_fct = 52
    for wks in range(1,ms_fct+1):
        allw_lags = (np.arange(wks,max_lags))
        coeffs_temp, yp_train[wks-1,:], tr_tp1, llr1, train_pred_err[wks-1,:], lags_temp = ARLR_model(train,allw_lags,win,llr_tol)
        lags_app[wks-1,lags_temp] = lags_temp
        coeffs[wks-1,:] = coeffs_temp
    yp_fct=np.zeros([fut_wks,ms_fct])
    yb_fct=np.zeros([fut_wks,ms_fct,Nb])
    log_scr = np.zeros([fut_wks, ms_fct])
    bn_mat_bst = np.zeros([fut_wks, len(bin_ed)-1, ms_fct])
    bn_mat_Gaussker = np.zeros([fut_wks, len(bin_ed)-1, ms_fct])
    # Once trained, use the coeffs to forecast multi-steps given data frame
    
    # For obtaining uncertainty in forecast estimates (using Boot strapping), choose uncer_anl = True,  
    data_frame = train
    data_test = []#test
    for new_wks in np.arange(0,fut_wks):
        data_frame = data_frame.append(test[new_wks:(new_wks+1)])
        data_test = data_test[1:]
        yp_fct[new_wks,:], yb_fct[new_wks,:,:], log_scr[new_wks,:], bn_mat_bst[new_wks, :,:], bn_mat_Gaussker, train_pred_err = multi_step_fct(data_frame, coeffs, lags_app, train_pred_err, ms_fct, win, Nb, bin_ed, uncer_anl)
    
    return np.exp(yp_fct), bn_mat_bst, bn_mat_Gaussker

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
    
    
    regions = get_regions()
    targets = get_targets()
    #if args.region not in regions:
    #    raise TypeError("region is not valid")
    #if args.region_type == "national":
    #    args.region_type = "US National"
    fct_weeks = args.weeks
    # 
    
    
    
    csv_path = args.ground_truth 
    year = args.forecast_from
    if int(args.test):
        directory = 'dump/'
    if not os.path.exists(directory):
        os.makedirs(directory) 
        
    
    directory_bst = 'output/' + 'ARLR_bst/' + str(args.forecast_from)
    directory_Gaussker = 'output/' + 'ARLR_Gaussker/' + str(args.forecast_from)

    if not os.path.exists(directory_bst):
        os.makedirs(directory_bst)
    if not os.path.exists(directory_Gaussker):
        os.makedirs(directory_Gaussker)
    bin_ed = get_bin()
    EWs = []
    for y in range(int(year),2020):
        for week in epi.Year(y).iterweeks():
            w = int(str(week))
            if (w<int(year+'40'))|(w>201920):
                continue
    #         print(w)
            EWs.append(str(w)) 
    for wks in EWs:#epi.Year(int(args.forecast_from)).iterweeks():
        startyear = wks[:4] #args.forecast_from[:4]
        startweek = wks[4:] #args.forecast_from[6:8]
        
        
        #trainweek = startweek
        fdf = prepdata_retro(csv_path,wks)
        fdf['REGION'] = fdf['REGION'].fillna('National')
        fdf.dropna(subset=['%UNWEIGHTED ILI'],inplace=True)
        fdf = fdf.drop(fdf[(fdf['REGION'] == 'Puerto Rico')|(fdf['REGION'] == 'Virgin Islands')|(fdf['REGION'] == 'New York City')].index)
    

        
        ews = epi.Week(int(startyear), int(startweek))
        for region in fdf['REGION'].unique():
            #for i in range(0, 1):
            #if region=='National' or 'HHS Regions':
            #    target = targets["wili"]
            #else:
            #    target = targets["ili"]
            target = targets['wili']
            df = fdf[fdf['REGION']==region]       
            predictions, bn_mat_bst, bn_mat_Gaussker = ARLR_module(df, region, target, ews, fct_weeks)
            if int(args.CDC):
                outputdistribution_bst(predictions[0,0:4], bn_mat_bst[0,:,0:4], bin_ed, region, target, directory_bst, ews)
                outputdistribution_Gaussker(predictions[0,0:4], bn_mat_Gaussker[:,0:4], bin_ed, region, target, directory_Gaussker, ews)

            if df['REGION TYPE'].unique() == 'States':
                print(region)
                accu_output(predictions.reshape(fct_weeks), region,  args.out_state, ews, args.st_fips)
if __name__ == "__main__":
    main()
# Multi-step forecast
#    for key in regions.keys():
#        if key=='Florida':
#            continue
#        else:
#            main({"REGION": key, "TARGET": "ili", "STARTDATE": "2017EW40", "ENDDATE": "2016EW39"})
