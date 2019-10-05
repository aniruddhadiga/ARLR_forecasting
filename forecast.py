# %load MS_ahd_fct.py
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
from data_prep import data_read_and_prep, get_season
from ARLR import ARLR_aug_phase, ARLR_red_phase, ARLR_fct, ARLR_err_met, fct_uncert, uncer_scr,multi_step_fct, ARLR_model

config = configparser.ConfigParser()
config_file = sys.argv[1]
config.read(config_file)

# Multi-step forecast
csv_path=config['Forecasting']['csv_path'] #'data/national/ILINet.csv'

epwk = int(config['Forecasting']['epwk']) # date from which to start forecasting
yr = int(config['Forecasting']['yr']) # Model trained on data upto (epiweek, year)

win = int(config['Forecasting']['win']) # Length of the historial training data to be considered

fut_wks = int(config['Forecasting']['fut_wks']) # Number of weeks ahead to forecast from training data 
ms_fct = int(config['Forecasting']['ms_fct']) # For every forecast week, give additional ms_fct weeks forecast

test_win = fut_wks+ms_fct # Number of true value to be fetched (testing accuracy)
exp_max_lags =  int(config['Forecasting']['exp_max_lags'])# expected maximum lags to be considered in the model
llr_tol=1e-2 # log-likelihood tolerance

# Uncertainty analysis
uncer_anl = (config['CDC']['uncer_anl'])
Nb = int(config['CDC']['Nb'])
# create bins
n_bins=int(config['CDC']['n_bins'])

#bin_ed = np.arange(0,n_bins,.1)
#bin_ed = np.append(bin_ed,20)
bin_ed= [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4,
        1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6,
        3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8,
        5.9, 6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 7.0, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 8.0,
        8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 9.0, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9, 10.0, 10.1,
        10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9, 11.0, 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 11.8, 11.9,
        12.0, 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7, 12.8, 12.9, 13.0, 100]
# Read csv file and create train and test data
train, test, df, df_train, df_test = data_read_and_prep(csv_path, epwk, yr, test_win, wght=True, log_tr=True)
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
#lags_season = [1, season_ind, 2*season_ind]
# lags_s, lags_app_s, res_s, yp_s, y_obs_s, tr_tp_s, llr_s, err_old_s,ind_s = ARLR_aug_phase(train,lags_season,260,llr_tol=3e-3) # augmentation phase
#coeffs_seas, yp_seas, tr_tp1_seas, llr1_seas, train_pred_err_seas, lags_seas = ARLR_model(train,lags_season,win,llr_tol)
# plt.fiigure(figsize=(12,7))
# plt.plot(ind_s,yp_s)
# plt.plot(ind_s,y_obs_s.values)
# plt.figure(figsize=(12,7))
# plt.plot(yp_s-y_obs_s)

# train the model
max_lags = 2*season_ind+2
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
data_test = test
for new_wks in np.arange(0,fut_wks):
    data_frame = data_frame.append(test[new_wks:(new_wks+1)])
    data_test = data_test[1:]
    yp_fct[new_wks,:], yb_fct[new_wks,:,:], log_scr[new_wks,:], bn_mat[new_wks, :,:], train_pred_err = multi_step_fct(data_frame, data_test, coeffs, lags_app, train_pred_err, ms_fct, win, Nb, bin_ed, uncer_anl)

## Chained forecasting
#allw_lags_ch = [1,2,3,4,5,6,7,8,9,season_ind,season_ind+2, 2*season_ind]
#coeffs_ch, yp_train_ch, tr_tp1_ch, llr1_ch, train_pred_err_ch, lags_app_ch = ARLR_model(train,allw_lags_ch,win,llr_tol)
#yp_fct_ch, ind_ch, pred_err_ch = ARLR_fct(coeffs_ch,train,test,lags_app_ch, fut_wks, 1)
