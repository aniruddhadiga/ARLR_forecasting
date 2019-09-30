# %load MS_ahd_fct.py
import sys
import os
import pandas as pd
from cycler import cycler
import numpy as np
import statsmodels.api as sm
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import epiweeks as epi
import datetime
import argparse
import re
import shutil
import pdb
sns.set()

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
from ARLR import ARLR_aug_phase, ARLR_red_phase, ARLR_fct, ARLR_err_met, fct_uncert, uncer_scr

# Multi-step forecast
csv_path='data/national/ILINet.csv'

epwk = 40
yr = 2016
win = 208
ms_fct = 4
fct_win = 108

llr_tol=1e-2
allw_lags = np.arange(1,52)
uncer_anl = False
Nb = 100
# Read csv file and create train and test data
train, test, df, df_train, df_test = data_read_and_prep(csv_path, epwk, yr, fct_win, wght=False, log_tr=True)
# dates = pd.DatetimeIndex(df_train["DATE"])
#plt.figure(figsize=(12,7))
#plt.subplot(2,1,1);plt.plot(train.index,(train));plt.title('Full training data from specified epiweek {}, {}'.format(epwk,yr))
# plt.subplot(2,1,2);plt.plot((hist_win(train,win)).index,(hist_win(train,win)));plt.title('Training data: 4 year period')

# Check data for stationarity in the training data with padding
train_win = train[-1:(-win-max(allw_lags)-1):-1] # training samples in the window period + buffer


result = adfuller(train_win)
print(result)
if result[1] < 0.05:
    print('p-val of ADF test %e' %result[1])
    print('Stationary signal')
plt.plot(train_win)

# Check seasonality
season_ind = get_season(train_win,fft_len=1024,figs=True)
# lags_season = [1, season_ind, 2*season_ind]
# lags_s, lags_app_s, res_s, yp_s, y_obs_s, tr_tp_s, llr_s, err_old_s,ind_s = ARLR_aug_phase(train,lags_season,260,llr_tol=3e-3) # augmentation phase
# plt.figure(figsize=(12,7))
# plt.plot(ind_s,yp_s)
# plt.plot(ind_s,y_obs_s.values)
# plt.figure(figsize=(12,7))
# plt.plot(yp_s-y_obs_s)

# train the model

res, yp_aug, y_obs, tr_tp, llr, err_old, lags_app, ind = ARLR_aug_phase(train,allw_lags,win,llr_tol)
# augmentation phase
resf, yp_red, tr_tp, llr, train_pred_err, res1, lags_app, ind = ARLR_red_phase(y_obs,tr_tp, err_old, allw_lags, res,lags_app,ind,llr_tol) # reduction phase
# create series
yp_train = pd.Series(yp_red)
yp_train.index = ind[:]


# lags to be tested
obs = train
yp_fct = np.zeros([ms_fct, fct_win+ms_fct])

for fct_wks in range(0, fct_win):
    
    cffs_arlr = np.zeros([ms_fct,win])
    lags_app_fct = np.zeros([ms_fct,win])
    pred_err = np.zeros([ms_fct,win])
    pred_err= train_pred_err
    fct_prdtrs = obs
    ind_fct = []
    yb_fct = np.zeros([ms_fct, Nb])
#     ind_fct = []
    for mwks in range(0,ms_fct):
        #range(mwks+1,60)
        yp_fct[mwks,fct_wks+mwks], ind_fct= ARLR_fct(resf,fct_prdtrs,test[(mwks):],lags_app,1)
#         ind_fct.append(ind_temp)
#         pdb.set_trace()
        pred_err = np.append(test[mwks]-yp_fct[mwks],pred_err)
        cffs_arlr[mwks,lags_app-1] = resf.params
        lags_app_fct[mwks,lags_app] = lags_app
        if uncer_anl:    
            yb_fct[mwks,:] = fct_uncert(fct_prdtrs, test, pred_err[:win],resf,lags_app, win, Nb)
        #pdb.set_trace()
        fct_prdtrs = fct_prdtrs.append(pd.Series(yp_fct[mwks,fct_wks],index=ind_fct))
        
        #plt.figure(mwks,figsize=(14,5));plt.plot(ind,yp1);plt.plot(y_obs)
    #     pdb.set_trace()
    if uncer_anl:
        log_scr, bn_mat = uncer_scr(yb_fct, test, yp_fct, ms_fct, Nb, 13)
    #rmse, mae, mape = ARLR_err_met(yp_fct[:,fct_wks], test[:ms_fct])
    
    #pdb.set_trace()
    #print(rmse, mae, mape)
    obs = obs.append(pd.Series(test.values[fct_wks], index=test.index[fct_wks:fct_wks+1]))

