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

epwk = 20
yr = 2016
test_wks = 52
win = 208
ms_fct = 4

llr_tol=1e-2
lags =[1,2,3,4,52, 53, 54]
Nb = 10
# Read csv file and create train and test data
train, test, df, df_train, df_test = data_read_and_prep(csv_path, epwk, yr, test_wks, wght=False, log_tr=True)
# dates = pd.DatetimeIndex(df_train["DATE"])
#plt.figure(figsize=(12,7))
#plt.subplot(2,1,1);plt.plot(train.index,(train));plt.title('Full training data from specified epiweek {}, {}'.format(epwk,yr))
# plt.subplot(2,1,2);plt.plot((hist_win(train,win)).index,(hist_win(train,win)));plt.title('Training data: 4 year period')

# Check data for stationarity in the training data with padding
train_win = train[-1:(-win-max(lags)-1):-1] # training samples in the window period


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

 # lags to be tested
cffs_arlr = np.zeros([ms_fct,win])
lags_app_fct = np.zeros([ms_fct,win])
pred_err = np.zeros([ms_fct,win])
train_fct = train
yp_fct = np.zeros(ms_fct)
yb_fct = np.zeros([ms_fct,Nb])
for mwks in range(0,ms_fct):
    #range(mwks+1,60)
    allw_lags = np.append(np.arange(1,5),52)
    res, yp, y_obs, tr_tp, llr, err_old, lags_app, ind = ARLR_aug_phase(train_fct,allw_lags,win,llr_tol)
    # augmentation phase
    resf, yp1, tr_tp, llr, pred_err[mwks,:], res1, lags_app, ind = ARLR_red_phase(y_obs,tr_tp, err_old, lags, res,lags_app,llr_tol) # reduction phase
    yp_fct[mwks], ind_fct= ARLR_fct(resf,train_fct,test[(mwks):],lags_app,1)
    cffs_arlr[mwks,lags_app-1] = resf.params
    lags_app_fct[mwks,lags_app] = lags_app
    rmse, mae, mape = ARLR_err_met(yp1, y_obs)
    yb_fct[mwks,:] = fct_uncert(train_fct, test, pred_err[mwks,:],resf,lags_app, win, Nb)
    #pdb.set_trace()
    train_fct = train_fct.append(pd.Series(yp_fct[mwks],index=ind_fct))
    #plt.figure(mwks,figsize=(14,5));plt.plot(ind,yp1);plt.plot(y_obs)
#     pdb.set_trace()
uncer_scr(yb_fct, test, yp_fct, ms_fct, Nb, 13)
