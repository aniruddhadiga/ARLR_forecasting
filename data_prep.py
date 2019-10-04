# %load data_prep.py
import sys
import os
import pandas as pd
from cycler import cycler
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import epiweeks as epi
import datetime
import argparse
import time
import re
import shutil
import pdb

from pandas import Series
from datetime import date, time, datetime, timedelta
from scipy import signal

def data_read_and_prep(csv_path, epwk, yr, test_wks=4, wght=False, log_tr=False):
    # Read in the historical ILI data from startdate given by epwk and year from the csv_path and 
    #   create train and test set 
    cdcdf = pd.read_csv(csv_path, header=1)
    df = cdcdf.drop(["REGION", "REGION TYPE", "AGE 0-4", "AGE 25-49", "AGE 25-64", "AGE 5-24", "AGE 50-64", "AGE 65", "NUM. OF PROVIDERS"], axis=1)


    df['DATE'] = pd.to_datetime(df.apply(lambda row : epi.Week(int(row["YEAR"]), int(row["WEEK"])).startdate() ,axis=1, result_type='reduce'))

    
    week=epi.Week(yr, epwk)

    df_train = df[(df['DATE']<=pd.to_datetime(week.startdate()))]
    df_test = df[(df['DATE']>pd.to_datetime(week.startdate()))&((df['DATE']<=pd.to_datetime(week.startdate())+timedelta(weeks=test_wks)))]
    if wght:
        train = df_train['% WEIGHTED ILI']
        test = df_test['% WEIGHTED ILI']
    else:
        train = df_train['%UNWEIGHTED ILI']
        test = df_test['%UNWEIGHTED ILI']
    if log_tr:
        train = np.log(train)
        test = np.log(test)
    train.index = df_train['DATE']
    test.index = df_test['DATE']
    return train, test, df, df_train, df_test

def get_season(y,fft_len=1024,figs=False):
    f, Pxx_den = signal.periodogram(y, nfft=fft_len)
    season_ind = round(fft_len/Pxx_den.argmax())
    print('Season index {}'.format(season_ind))
    if figs:
        plt.figure();plt.semilogy(f, Pxx_den)
#         plt.plot(Pxx_den.argmax(),1,'r--')
    return int(season_ind)

def hist_win(y,win):
    y_hist = y[(-win-1):-1]
    return y_hist



