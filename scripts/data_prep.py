# %load data_prep.py
import sys
import os
import pandas as pd
import numpy as np
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
    Pxx_den = np.abs(Pxx_den)
    season_ind = round(fft_len/Pxx_den.argmax())
    print('Season index {}'.format(season_ind))
    return int(season_ind)

def hist_win(y,win):
    y_hist = y[(-win-1):-1]
    return y_hist

def prepdata_append():
    national = pd.read_csv('data/national/ILINet.csv', header=1)
    regional = national.append(pd.read_csv('data/regional/ILINet.csv', header=1))
    df = regional.append(pd.read_csv('data/state/ILINet.csv', header=1))
    return df

def prepdata(csv_path):
    
    df = pd.read_csv(csv_path, na_values='X')
    df['DATE'] = pd.to_datetime(df.apply(lambda row : epi.Week(int(row["YEAR"]), int(row["WEEK"])).startdate() ,axis=1, result_type='reduce'))

def prepdata_retro(csv_path,epwk):
    nat_csv_file = csv_path + '/' +'national/'+'ILINet_National_' + epwk + '.csv'
    df = pd.read_csv(nat_csv_file, na_values='X')
    
    hhs_csv_file = csv_path +'/'+'hhs/'+'ILINet_HHS_' + epwk + '.csv'
    df = df.append(pd.read_csv(hhs_csv_file,na_values='X'))
    df['REGION'] = df['REGION'].fillna('National')
    df['DATE'] = pd.to_datetime(df.apply(lambda row : epi.Week(int(row["YEAR"]), int(row["WEEK"])).startdate() ,axis=1, result_type='reduce'))

def prepdata_flux(csv_path,epwk):
    nat_csv_file = csv_path + '/'+'ILINet_national_' + str(epwk.year) +'EW'+ str(epwk.week) + '.csv'
    df = pd.read_csv(nat_csv_file, na_values='X')
    
    hhs_csv_file = csv_path +'/'+ 'ILINet_hhs_' + str(epwk.year) +'EW'+ str(epwk.week) + '.csv'
    df = df.append(pd.read_csv(hhs_csv_file,na_values='X'))
    df['region'] = df['region'].fillna('National')
    df['DATE'] = pd.to_datetime(df.apply(lambda row : epi.Week(int(row["year"]), int(row["week"])).startdate() ,axis=1, result_type='reduce'))


        #df.set_index(['DATE'], inplace=True)

    #if region == 'US National':
    #    df = cdcdf[cdcdf['REGION TYPE']=='National']
    #    df['DATE'] = pd.to_datetime(df.apply(lambda row : epi.Week(int(row["YEAR"]), int(row["WEEK"])).startdate() ,axis=1, result_type='reduce'))
    #    #df.set_index(['DATE'], inplace=True)

    #elif region.isdigit():
    #    df = cdcdf[cdcdf['REGION']== "Region " + str(region)]
    #    df['DATE'] = pd.to_datetime(df.apply(lambda row : epi.Week(int(row["YEAR"]), int(row["WEEK"])).startdate() ,axis=1, result_type='reduce'))
    #    #df.set_index(['DATE'], inplace=True)

    #    #When I set the date row as the index, I can no longer access it using df['DATE]
    #else:
    #    df = cdcdf[cdcdf['REGION']==region]
    #    
    #    df['DATE'] = pd.to_datetime(df.apply(lambda row : epi.Week(int(row["YEAR"]), int(row["WEEK"])).startdate() ,axis=1, result_type='reduce'))
        #df.set_index(['DATE'], inplace=True)

    return df
