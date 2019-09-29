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
import time
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
from ARLR import ARLR_aug_phase, ARLR_red_phase, ARLR_fct

csv_path='data/national/ILINet.csv'

epwk = 20
yr = 2018
test_wks = 52

train, test, df, df_train, df_test = data_read_and_prep(csv_path, epwk, yr, test_wks, wght=False, log_tr=True)
# dates = pd.DatetimeIndex(df_train["DATE"])
plt.figure(figsize=(12,7))
plt.subplot(2,1,1);plt.plot(train.index,(train));plt.title('Full training data from specified epiweek {}, {}'.format(epwk,yr))
# plt.subplot(2,1,2);plt.plot((hist_win(train,win)).index,(hist_win(train,win)));plt.title('Training data: 4 year period')

# Check data for stationarity in the training data with padding
win = 208 # 4-year window
lags = range(1,win);
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

# Run ARLR to find model parameters
llr_tol=1e-2
 # lags to be tested
res, yp, y_obs, tr_tp, llr, err_old, lags_app, ind = ARLR_aug_phase(train,lags,win,llr_tol) # augmentation phase
resf, yp1, tr_tp, llr, pred_err, res1, lags_app, ind = ARLR_red_phase(y_obs,tr_tp, err_old, lags, res,lags_app,ind,llr_tol) # reduction phase

coeffs_arlr = np.zeros(len(lags))
coeffs_arlr[lags_app-1] = res.params
print(lags_app)

yp_fct, ind_fct= ARLR_fct(res,train,test,lags_app,test_wks)

plt.figure(figsize=(12,7))
# h5, = plt.plot(train)
h1, = plt.plot(y_obs)
h2, = plt.plot(ind,(yp1))
h3, = plt.plot(ind_fct, yp_fct, '--')
h4, = plt.plot(test[0:test_wks])

plt.legend([h1,h2,h4,h3],['True', 'Pred', 'True Test', 'Forecast'])

pred_err = y_obs.values-yp1
plt.figure(figsize=(12,7))
h1 = plt.plot(ind, pred_err)
plt.title('Error')
plt.figure(figsize=(12,7))
pred_acf = np.correlate(pred_err,pred_err,"full")
plt.figure(figsize=(12,7))
plt.plot(coeffs_arlr)
plt.figure()
h1 = plt.plot(pred_acf)
