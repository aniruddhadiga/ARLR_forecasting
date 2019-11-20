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
from aw_micro import cdc_data

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

def prepdata_append(csv_path):
    
    national = pd.read_csv(csv_path+'national/ILINet.csv', na_values='X',header=1)
    national['REGION'] = national['REGION'].fillna('National')
    regional = national.append(pd.read_csv(csv_path+'regional/ILINet.csv',na_values='X', header=1))
    df = regional.append(pd.read_csv(csv_path+'state/ILINet.csv', na_values='X', header=1))
    df['DATE'] = pd.to_datetime(df.apply(lambda row : epi.Week(int(row["YEAR"]), int(row["WEEK"])).startdate() ,axis=1, result_type='reduce'))

    return df

def prepdata(csv_path):    
    df = pd.read_csv(csv_path, na_values='X', header=1)
    df['REGION'] = df['REGION'].fillna('National')
    df['DATE'] = pd.to_datetime(df.apply(lambda row : epi.Week(int(row["YEAR"]), int(row["WEEK"])).startdate() ,axis=1, result_type='reduce'))
    return df
    
def prepdata_retro(csv_path,epwk):
    nat_csv_file = csv_path + '/' +'national/'+'ILINet_National_' + str(epwk) + '.csv'
    df = pd.read_csv(nat_csv_file, na_values='X')
    df['REGION'] = df['REGION'].fillna('National')
    hhs_csv_file = csv_path +'/'+'hhs/'+'ILINet_HHS_' + str(epwk) + '.csv'
    df = df.append(pd.read_csv(hhs_csv_file,na_values='X'))
    df['REGION'] = df['REGION'].fillna('National')
    df['DATE'] = pd.to_datetime(df.apply(lambda row : epi.Week(int(row["YEAR"]), int(row["WEEK"])).startdate() ,axis=1, result_type='reduce'))
    return df

def prepdata_flux(csv_path,epwk):
    nat_csv_file = csv_path + '/'+'ILINet_national_' + str(epwk.year) +'EW'+ str(epwk.week) + '.csv'
    df = pd.read_csv(nat_csv_file, na_values='X')
    df['region'] = df['region'].fillna('National')
    hhs_csv_file = csv_path +'/'+ 'ILINet_hhs_' + str(epwk.year) +'EW'+ str(epwk.week) + '.csv'
    df = df.append(pd.read_csv(hhs_csv_file,na_values='X'))
    state_csv_file = csv_path +'/'+ 'ILINet_state_' + str(epwk.year) +'EW'+ str(epwk.week) + '.csv'
    df = df.append(pd.read_csv(state_csv_file,na_values='X'))

    df['DATE'] = pd.to_datetime(df.apply(lambda row : epi.Week(int(row["year"]), int(row["week"])).startdate() ,axis=1, result_type='reduce'))
    
    return df

def prep_aw_data(st_id_path, **kwargs):
    '''Prepares weather and return the corresponding dataframe. kwargs is a dictionary woth key as "national", "HHS", and/or "States" and values are the paths. Prepare this dictionary before calling this functions.'''
    df_wtr = pd.DataFrame()
    for key,value in kwargs.items():
        if key == "National":
            df_wtr_temp = pd.read_csv(value)
            df_wtr_temp['region'] = df_wtr_temp.apply(lambda x: "National", axis=1)
            df_wtr_temp['region_type'] = df_wtr_temp.apply(lambda x: "National", axis=1)
        elif key == "HHS":
            df_wtr_temp = pd.read_csv(value)
            df_wtr_temp['region'] = df_wtr_temp.apply(lambda x: "Region {}".format(x['area_id']),axis=1)
            df_wtr_temp['region_type'] = df_wtr_temp.apply(lambda x: "HHS Regions", axis=1)
        elif key == "States":
            df_wtr_temp = pd.read_csv(value)
            df_wtr_temp = df_wtr_temp[~df_wtr_temp.area_id.isin([72,78])]
            df_st_id = pd.read_csv(st_id_path)
            df_wtr_temp['region'] = df_wtr_temp.apply(lambda row: df_st_id[df_st_id['state']==row['area_id']]['state_name'].values[0], axis=1)
            df_wtr_temp['region_type'] = df_wtr_temp.apply(lambda x: "States", axis=1)
        df_wtr = df_wtr.append(df_wtr_temp)
    pp = pd.to_datetime([epi.Week(int(cdc_data.date2ew(d.date())[0]),int(cdc_data.date2ew(d.date())[1])).startdate() for d in pd.to_datetime(df_wtr.date)])
    df_wtr.index = pp  
    df_wtr.index = df_wtr.index.rename('DATE')
    return df_wtr

def prep_ght_data(**kwargs):
    '''Prepares ght and return the corresponding dataframe. kwargs is a dictionary woth key as "national", "HHS", and/or "States" and values are the paths. Prepare this dictionary before calling this functions.'''
    df_ght = pd.DataFrame()
    for key,value in kwargs.items():
#         pdb.set_trace()
        if key == "National":
            df_ght_temp = pd.read_csv(value)                                  
            df_ght_temp['region'] = df_ght_temp.apply(lambda x: "National", axis=1)
            df_ght_temp['region_type'] = df_ght_temp.apply(lambda x: "National", axis=1)   
        elif key == "HHS":
            df_ght_temp = pd.read_csv(value)
            df_ght_temp['region'] = df_ght_temp.apply(lambda x: "Region {}".format(x['hhs']),axis=1)
            df_ght_temp['region_type'] = df_ght_temp.apply(lambda x: "HHS Regions", axis=1)
        elif key == "States":
            df_ght_temp = pd.read_csv(value)
            df_ght_temp['region'] = df_ght_temp.apply(lambda x: x['state'], axis=1)
            df_ght_temp['region_type'] = df_ght_temp.apply(lambda x: "States", axis=1)
        df_ght = df_ght.append(df_ght_temp)
#     pp = pd.to_datetime([epi.Week(int(cdc_data.date2ew(d.date())[0]),int(cdc_data.date2ew(d.date())[1])).startdate() for d in pd.to_datetime(df_ght.date)])
#     df_ght.index = pp  
    df_ght = df_ght.set_index('date')
    df_ght.index = df_ght.index.rename('DATE')
    return df_ght

#def prep_aw_data(aw_csv_path, st_id_path):
#    df_ex = pd.read_csv(aw_csv_path)
#    pp = pd.to_datetime([epi.Week(int(cdc_data.date2ew(d.date())[0]),int(cdc_data.date2ew(d.date())[1])).startdate() for d in pd.to_datetime(df_ex.date)])
#    df_ex['ep_st_date'] = pp
#    df_ex.index = pp
#    df_ex.index = df_ex.index.rename('DATE')
#    df_ex = df_ex[~df_ex.area_id.isin([72,78])]
#    df_st_id = pd.read_csv(st_id_path)
#    df_ex['REGION'] = df_ex.apply(lambda row: df_st_id[df_st_id['state']==row['area_id']]['state_name'].values[0], axis=1)
#
#    return(df_ex)
#        #df.set_index(['DATE'], inplace=True)
#
#    #if region == 'US National':
#    #    df = cdcdf[cdcdf['REGION TYPE']=='National']
#    #    df['DATE'] = pd.to_datetime(df.apply(lambda row : epi.Week(int(row["YEAR"]), int(row["WEEK"])).startdate() ,axis=1, result_type='reduce'))
#    #    #df.set_index(['DATE'], inplace=True)
#
#    #elif region.isdigit():
#    #    df = cdcdf[cdcdf['REGION']== "Region " + str(region)]
#    #    df['DATE'] = pd.to_datetime(df.apply(lambda row : epi.Week(int(row["YEAR"]), int(row["WEEK"])).startdate() ,axis=1, result_type='reduce'))
#    #    #df.set_index(['DATE'], inplace=True)
#
#    #    #When I set the date row as the index, I can no longer access it using df['DATE]
#    #else:
#    #    df = cdcdf[cdcdf['REGION']==region]
#    #    
#    #    df['DATE'] = pd.to_datetime(df.apply(lambda row : epi.Week(int(row["YEAR"]), int(row["WEEK"])).startdate() ,axis=1, result_type='reduce'))
#        #df.set_index(['DATE'], inplace=True)
#
#    return df
