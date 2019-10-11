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
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima_model import ARIMA

from sklearn.metrics import mean_squared_error

from scipy.linalg import toeplitz
from scipy.stats.distributions import chi2
from scipy import signal

from statsmodels.tsa.stattools import adfuller
from data_prep import data_read_and_prep, get_season, prepdata
from ARLR import ARLR_aug_phase, ARLR_red_phase, ARLR_fct, ARLR_err_met, fct_uncert, uncer_scr,multi_step_fct, ARLR_model, outputdistribution

def national():
    cdcdf = pd.read_csv('../data/national/ILINet.csv', header=1)
    df = cdcdf.drop(["REGION", "REGION TYPE", "AGE 0-4", "AGE 25-49", "AGE 25-64", "AGE 5-24", "AGE 50-64", "AGE 65", "NUM. OF PROVIDERS"], axis=1)
    
    df['DATE'] = pd.to_datetime(df.apply(lambda row : epi.Week(int(row["YEAR"]), int(row["WEEK"])).startdate() ,axis=1, result_type='reduce'))
    
    return df

def region(number):
    cdcdf = pd.read_csv('../data/regional/ILINet.csv', header=1)
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
    cdcdf = pd.read_csv('../data/state/ILINet.csv', header=1)
    cdcdf = cdcdf.drop(["REGION TYPE", "AGE 0-4", "AGE 25-49", "AGE 25-64", "AGE 5-24", "AGE 50-64", "AGE 65", "NUM. OF PROVIDERS"], axis=1)
    dfs = {}

    for state in cdcdf["REGION"].unique():
        dfs[state] = pd.DataFrame(cdcdf.loc[cdcdf["REGION"] == state])


    for df in dfs.values():
        df.drop(["REGION"], axis=1, inplace=True)
        df['DATE'] = pd.to_datetime(df.apply(lambda row : epi.Week(int(row["YEAR"]), int(row["WEEK"])).startdate() ,axis=1, result_type='reduce'))
        #df.drop(["YEAR", "WEEK"], axis = 1, inplace = True)
    
    return dfs[name]
regions = {"national": national, "1": region, "2": region, "3": region, "4": region, "5": region, "6": region, "7": region, "8": region, "9": region, "10": region, "Alabama": state, "Alaska": state, "Arizona": state, "Arkansas": state,"California": state, "Colorado": state, "Connecticut": state, "Delaware": state, "Florida": state, "Georgia": state, "Hawaii": state, "Idaho": state, "Illinois": state, "Indiana": state, "Iowa": state, "Kansas": state, "Kentucky": state,"Louisiana": state, "Maine": state, "Maryland": state, "Massachusetts": state, "Michigan": state, "Minnesota": state, "Mississippi": state, "Missouri": state, "Montana": state, "Nebraska": state, "Nevada": state, "New Hampshire": state, "New Jersey": state, "New Mexico": state, "New York": state, "North Carolina": state, "North Dakota": state, "Ohio": state, "Oklahoma": state, "Oregon": state, "Pennsylvania": state, "Rhode Island": state, "South Carolina": state, "South Dakota": state, "Tennessee": state, "Texas": state, "Utah": state, "Vermont": state, "Virginia": state, "Washington": state, "West Virginia": state, "Wisconsin": state, "Wyoming": state}

targets = {"wili": "% WEIGHTED ILI", "ili": "%UNWEIGHTED ILI", "ilitotal": "ILITOTAL", "totalpatients": "TOTAL PATIENTS"} #onset and peak week still need to be added



def ARLR_module(df, region, target, epi_week):
    config = configparser.ConfigParser()
    config_file = 'config.ini'
    config.read(config_file) 
    ww_train = epi_week-1
    ww_test = epi_week   
    cdcdf = df
    starttraining_date = pd.to_datetime(ww_train.startdate())
    testing_date = pd.to_datetime(ww_test.startdate())
    #endtraining = pd.to_datetime(enddate.startdate())
    #startpredict = pd.to_datetime((enddate+1).startdate())
    #endpredict = pd.to_datetime((enddate+4).startdate())
    
    if region == 'US National':
        df = cdcdf[cdcdf['REGION TYPE']=='National']
        df['DATE'] = pd.to_datetime(df.apply(lambda row : epi.Week(int(row["YEAR"]), int(row["WEEK"])).startdate() ,axis=1, result_type='reduce'))
        #df.set_index(['DATE'], inplace=True)

    elif region.isdigit():
        df = cdcdf[cdcdf['REGION']== "Region " + str(region)]
        df['DATE'] = pd.to_datetime(df.apply(lambda row : epi.Week(int(row["YEAR"]), int(row["WEEK"])).startdate() ,axis=1, result_type='reduce'))
        df.set_index(['DATE'], inplace=True)

        #When I set the date row as the index, I can no longer access it using df['DATE]
    else:
        df = cdcdf[cdcdf['REGION']==region]
        df['DATE'] = pd.to_datetime(df.apply(lambda row : epi.Week(int(row["YEAR"]), int(row["WEEK"])).startdate() ,axis=1, result_type='reduce'))
        df.set_index(['DATE'], inplace=True)
    
    df_train = df[(df['DATE']<pd.to_datetime(ww_train.startdate()))]
    df_test = df[(df['DATE']>=pd.to_datetime(ww_train.startdate()))]
    
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
    config = configparser.ConfigParser()
    config_file = 'config.ini'  
    config.read(config_file)
    # Multi-step forecast
    
    
    win = int(config['Forecasting']['win']) # Length of the historial training data to be considered
    
    fut_wks = int(config['Forecasting']['fut_wks']) # Number of weeks ahead to forecast from training data 
    ms_fct = int(config['Forecasting']['ms_fct']) # For every forecast week, give additional ms_fct weeks forecast
    
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
    bin_ed= [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 7.0, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 8.0, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 9.0, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9, 10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9, 11.0, 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 11.8, 11.9, 12.0, 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7, 12.8, 12.9, 13.0, 100]
    # Read csv file and create train and test data
    # dates = pd.DatetimeIndex(df_train["DATE"])
    #plt.figure(figsize=(12,7))
    #plt.subplot(2,1,1);plt.plot(train.index,(train));plt.title('Full training data from specified epiweek {}, {}'.format(epwk,yr))
    # plt.subplot(2,1,2);plt.plot((hist_win(train,win)).index,(hist_win(train,win)));plt.title('Training data: 4 year period')
    
    # Check data for stationarity in the training data with padding
    train_win = train[-1:(-win-exp_max_lags-1):-1] # training samples in the window period + buffer
    
    
    result = adfuller(train_win)
    #print(result)
    #if result[1] < 0.05:
    #    print('p-val of ADF test %e' %result[1])
    #    print('Stationary signal')
    # plt.plot(train_win)
    # Check seasonality
    season_ind = get_season(train_win,fft_len=1024,figs=False)
    # train the model
    max_lags = 55
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
    bn_mat = np.zeros([fut_wks, len(bin_ed)-1, ms_fct])
    # Once trained, use the coeffs to forecast multi-steps given data frame
    
    # For obtaining uncertainty in forecast estimates (using Boot strapping), choose uncer_anl = True,  
    data_frame = train
    data_test = []#test
    for new_wks in np.arange(0,fut_wks):
        data_frame = data_frame.append(test[new_wks:(new_wks+1)])
        data_test = data_test[1:]
        yp_fct[new_wks,:], yb_fct[new_wks,:,:], log_scr[new_wks,:], bn_mat[new_wks, :,:], train_pred_err = multi_step_fct(data_frame, coeffs, lags_app, train_pred_err, ms_fct, win, Nb, bin_ed, uncer_anl)
    
    return np.exp(yp_fct), bn_mat

def main(args):
    #parser = argparse.ArgumentParser(description='Script that runs an autoregressive forecasting model upon available CDC flu data.')
    #parser.add_argument('REGION', help='Region selector. Valid regions are "national", regions "1" - "10", or any state, e.g. "Alabama", "Michigan"')
    #parser.add_argument('TARGET', help='Target to forecast upon. Valid targets are Weighted ILI ("wili"), Unweighted ILI ("ili"), ILI Total ("ilitotal"), or Total Patients ("totalpatients")')
    #parser.add_argument('STARTDATE', help='Year in which the model will start training, formatted as "2018EW05".')
    #parser.add_argument('ENDDATE', help='Date at which the model will stop training, formatted as "2018EW05".')
    #args = parser.parse_args()
    args = vars(args) 
    if args["REGION"] not in regions:
        raise TypeError("REGION is not valid")

    if args["REGION"] == "national":
        args["REGION"] = "US National"
    

    if args["TARGET"] not in targets:
        raise TypeError("TARGET is not valid")

    if re.fullmatch('\d{4}(EW)\d{2}', args["STARTDATE"]) is None:
        raise TypeError("STARTDATE is formatted incorrectly")
    
    if re.fullmatch('\d{4}(EW)\d{2}', args["ENDDATE"]) is None:
        raise TypeError("ENDDATE is formatted incorrectly")
    
    bin_ed= [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 7.0, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 8.0, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 9.0, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9, 10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9, 11.0, 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 11.8, 11.9, 12.0, 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7, 12.8, 12.9, 13.0, 100]
   
 
    startyear = args["STARTDATE"][:4]
    startweek = args["STARTDATE"][6:8]
    trainweek = startweek
    ww = epi.Week(int(startyear), int(startweek))
    region = args["REGION"]
    target = targets[args["TARGET"]]
    df = prepdata()
    directory = 'output/' + str(ww.year) + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    for i in range(0, 40):
        predictions, bn_mat = ARLR_module(df, region, target, ww+i)
        #pdb.set_trace()
        outputdistribution(predictions.reshape(4), bn_mat.reshape([131,4]), bin_ed, region, target, directory, ww+i)
        pdb.set_trace()

#if __name__ == "__main__":
#    
#    parser = argparse.ArgumentParser(description='Script that runs an autoregressive forecasting model upon available CDC flu data.')
#    parser.add_argument('REGION', help='Region selector. Valid regions are "national", regions "1" - "10", or any state, e.g. "Alabama", "Michigan"')
#    parser.add_argument('TARGET', help='Target to forecast upon. Valid targets are Weighted ILI ("wili"), Unweighted ILI ("ili"), ILI Total ("ilitotal"), or Total Patients ("totalpatients")')
#    parser.add_argument('STARTDATE', help='Year in which the model will start training, formatted as "2018EW05".')
#    parser.add_argument('ENDDATE', help='Date at which the model will stop training, formatted as "2018EW05".')
#    parser.add_argument('--sigma', default=.5, help="Standard deviation with which to apply normal distribution", type=float)
#    parser.add_argument('--distribution', action='store_true', default=False, 
#    help='Specifies whether to return the predictions as a normal distribution')
#
#    args = parser.parse_args()
#    main(vars(args))

# Multi-step forecast
for key in regions.keys():
    forecast_ARLR.main({"REGION": key, "TARGET": "ili", "STARTDATE": "2017EW40", "ENDDATE": "201    6EW39"})
