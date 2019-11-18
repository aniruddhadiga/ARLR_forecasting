'''Contains functions that correspond to the ARLR method with exogenous regressors '''
from statsmodels.tsa.ar_model import AR
import numpy as np
import statsmodels.api as sm
#import matplotlib.pyplot as plt
import pandas as pd
import pdb, os
from aw_micro import cdc_data
import datetime
import configparser
import pkg_resources
import warnings
warnings.filterwarnings('ignore')
# ARLR functions
def get_bin():    
    bin_ed= [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 7.0, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 8.0, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 9.0, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9, 10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9, 11.0, 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 11.8, 11.9, 12.0, 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7, 12.8, 12.9, 13.0, 100]
    return bin_ed
 
def get_exog_reg(targ_dict):
    all_exog_rgsr = []
    for key in targ_dict:
        if key != 'target':
            all_exog_rgsr = all_exog_rgsr+targ_dict[key]
    return all_exog_rgsr

def rgsr_to_ind(exog_rgsr, allw_lags, all_exog_rgsr):
    rgsr_ind=max(allw_lags)+1+all_exog_rgsr.index(exog_rgsr)
    return rgsr_ind

def rgsrs_to_indices(lags_app, allw_lags, all_exog_rgsr):
    rgsrs_indices = []
    for ind in lags_app:
        if str(ind).isdigit():
            rgsrs_indices.append(ind)
        else:
            rgsrs_indices.append(rgsr_to_ind(ind,allw_lags,all_exog_rgsr))
    return rgsrs_indices

def ind_to_rgsr(exog_rgsr_ind, allw_lags, all_exog_rgsr):
    rgsr_name= all_exog_rgsr[exog_rgsr_ind-(max(allw_lags)+1)]
    return rgsr_name


def indices_to_rgsrs(lags_ind, allw_lags, all_exog_rgsr):
    rgsrs_names = []
    for ind in lags_ind:
        if ind <= max(allw_lags):
            rgsrs_names.append(ind)
        else:
            rgsrs_names.append(ind_to_rgsr(ind,allw_lags,all_exog_rgsr))
    return rgsrs_names
                            

def ARLR_regressor(df, df_wtr, df_ght, region, mask_targ_dict, ews):
    '''If we have other regressors, need a for loop'''
    ews_1 = ews+1 # we need ght and weather data for forecst week, hence +1
    df_reg = df[df['region']==region]
    df_m = df_reg
    if df_wtr.empty and not df_ght.empty:
        df_m = pd.merge(df_m,df_wtr,how='outer', left_index=True, right_index=True)
        df_ght_reg = df_ght[df_ght['region']==region] 
        df_m = pd.merge(df_m,df_ght_reg,how='outer', left_index=True, right_index=True)
    if df_ght.empty and not df_wtr.empty:
        df_m = pd.merge(df_m,df_ght,how='outer', left_index=True, right_index=True)
        df_wtr_reg = df_wtr[df_wtr['region']==region]
        df_m = pd.merge(df_reg,df_wtr_reg,how='outer', left_index=True, right_index=True)
    elif not df_ght.empty and not df_wtr.empty:
        df_wtr_reg = df_wtr[df_wtr['region']==region]
        df_ght_reg = df_ght[df_ght['region']==region]
        df_m = pd.merge(df_m,df_wtr_reg,how='outer', left_index=True, right_index=True)
        df_m = pd.merge(df_m,df_ght_reg,how='outer', left_index=True, right_index=True)
    
    mask_targ = []
    for k in list(mask_targ_dict):
        mask_targ = mask_targ+(mask_targ_dict[k])
    df_m = df_m[mask_targ]
    df_m = df_m[df_m.index<=pd.to_datetime((ews_1).startdate())].fillna(1e-2)
    df_m[mask_targ_dict['target']] = np.log(df_m[mask_targ_dict['target']])
    df_m = df_m.replace(-np.inf, np.nan)
    df_m = df_m.interpolate()
    df_m[mask_targ_dict['ght_target']] = (df_m[mask_targ_dict['ght_target']])
    df_m[mask_targ_dict['aw_target']] = (df_m[mask_targ_dict['aw_target']])
    return df_m
    

def ARLR_aug_phase_exog(df_m, lags, targ_dict, win,llr_tol):
    '''df_m: DataFrame that contains data upto forecast date, hence, for "ili" no values are present but it is present for ght and weather, exogenous variables provide a prior to forecast as they contain forecast week information''' 
    all_exog_rgsr = get_exog_reg(targ_dict)
    y = ((df_m[targ_dict['target']]))
    y = y[-1::-1]
    ind = y.index
    try:
        y = np.array(y).reshape(len(y))
    except:
        pdb.set_trace()
    y_obs = y[1:(win+1)]#-y[2:(win+2)] # 1 shift as we will have current week data for ght and weather for which we provide forecast
    ind = ind[0:win]
    y_obs = np.array(y_obs)
    if np.linalg.norm(y_obs) == 0:
        pdb.set_trace()
    lags_chk = list(np.array(lags).astype(int)) # lags pertaining to AR coeffs
    lags_chk_all = lags_chk+all_exog_rgsr 
    all_rgsr = lags_chk+all_exog_rgsr # Need to keep a copy of lags_chk_all as it is getting updated
    lags_app = []
    err_old = np.linalg.norm(y_obs)#np.random.randn(win,1))
    tr_tp = np.zeros([win,len(lags_chk_all)])
    
    init_lags_len = len(lags_chk_all)
    
    for k in range(0,init_lags_len):
        err_m = np.zeros([len(lags_chk_all)])
        llr = np.zeros([len(lags_chk_all)])
        jj = 0
        for i in lags_chk_all:            
            if str(i).isdigit(): # check if it is a lag or exog column
                tr_tp[:,k] = y[(i+1):(win+i+1)]
            else:
                #print(i)
                exog_reg = (df_m[i]-df_m[i].shift()).values # reads the column name in the dataframe specified by name="i"
                exog_reg = np.flip(exog_reg)
                exog_reg = exog_reg[min(lags_chk):(min(lags_chk)+win)] # Most recent date -1 week's data used for exog. variable for training 
                tr_tp[:,k] = exog_reg[0:win]
                
            tr_tp_mul = np.matmul(tr_tp[:,0:(k+1)].T, tr_tp[:,0:(k+1)])
            try:
                res = sm.OLS(y_obs,tr_tp[:,0:(k+1)]).fit()
            except:
                pdb.set_trace()
            #res = sm.OLS(y_obs,tr_tp[:,0:(k+1)]).fit()   
            yp = res.predict()
            err_m[jj] = np.linalg.norm(y_obs-yp)
            llr[jj] = 2*np.log(err_old/err_m[jj])
            jj+=1
             
        imax = np.argmax(llr)
    #     p_val = chi2.sf(llr[imax],1)
    #     print(p_val)
        if llr[imax] > llr_tol:
#             pdb.set_trace()
            if str(lags_chk_all[imax]).isdigit():
                tr_tp[:,k] =y[(lags_chk[imax]+1):(win+lags_chk[imax]+1)]
            else:
                exog_reg = np.flip(exog_reg)
                exog_reg = df_m[lags_chk_all[imax]].values
                tr_tp[:,k] = exog_reg[min(lags_chk):(min(lags_chk)+win)]
                
            lags_app.append(lags_chk_all[imax])
            lags_chk_all.remove(lags_chk_all[imax])
            err_old = err_m[imax]
        else:
            break
    if k:
        res = sm.OLS(y_obs,tr_tp[:,0:(k)]).fit()
    else:
        res = sm.OLS(y_obs,tr_tp[:,0]).fit()
    
    yp = res.predict()
    pred_err = y_obs-yp
    return res, yp, y, tr_tp, llr, pred_err, lags_app, ind, all_rgsr,win


def ARLR_red_phase_exog(y,tr_tp,err_old, lags, res1, lags_app,ind, win, llr_tol):
    y_obs = y[1:(win+1)]#-y[2:(win+2)] # 1 shift as we will have current week data for ght and weather for which we provide forecast
    tot_col = len(lags_app)
    if tot_col == 0:
        yp_lag1 = y_obs
        temp_tr_tp = tr_tp[:,0:tot_col]
    else:
        temp_tr_tp = tr_tp[:,0:tot_col]
        for i in range(0,tot_col+1):
            init_lag_len_new = temp_tr_tp.shape[1]
            jj = 0 
            err_m = np.zeros([len(lags_app)])
            llr = np.zeros([len(lags_app)])
            for k in range(0,tot_col):
#                 pdb.set_trace()
                if tot_col <=1:
                    break
        
                res = sm.OLS(y_obs,np.delete(temp_tr_tp,k,1)).fit()
                yp = res.predict()
                err_m[jj] = np.linalg.norm(y_obs-yp)
                llr[jj] = 2*np.log(err_old/err_m[jj])
                jj+=1
            try:
                imin = np.argmin(llr)
            except:
                pdb.set_trace()
#             if (llr[imin]>llr_tol):
#                 temp_tr_tp = np.delete(temp_tr_tp,imin,1) 
#                 lags_app.remove(lags_app[imin])
#                 err_old = err_m[imin]
# #                 pdb.set_trace()
#             else:
# #                 pdb.set_trace()
#                 break

    res = sm.OLS(y_obs,temp_tr_tp).fit()
    yp = res.predict()
    pred_err = y_obs-yp
    print(lags_app)
    yp_lag1 = yp 
    return res, yp_lag1, temp_tr_tp, llr, pred_err, res1, lags_app, ind

def ARLR_model_exog(df_m, allw_lags, targ_dict, win,llr_tol=1e-3):
    res, yp_aug, y_obs, tr_tp, llr, train_pred_err, lags_app, ind, all_rgsr, win = ARLR_aug_phase_exog(df_m, allw_lags, targ_dict, win,llr_tol)
    resf, yp_red, tr_tp, llr, train_pred_err, res1, lags_app, ind = ARLR_red_phase_exog(y_obs,tr_tp, np.linalg.norm(train_pred_err), allw_lags, res,lags_app,ind,win,llr_tol) # reduction phase
    yp_train = pd.Series(yp_red)
    yp_train.index = ind[:]
    all_exog_rgsr = get_exog_reg(targ_dict)
    num_coeffs = np.max(allw_lags)+len(all_exog_rgsr)
    coeffs=np.zeros(num_coeffs+1)
#     all_rgsr = 
    rgsrs_indices = rgsrs_to_indices(lags_app, allw_lags, all_exog_rgsr)
    coeffs[rgsrs_indices] = resf.params
    return coeffs, yp_train, tr_tp, train_pred_err, lags_app

def ARLR_fct_exog(coeffs,df_m,lags_app,fct_win, wks, allw_lags, targ_dict):

    all_exog_rgsr = get_exog_reg(targ_dict)
    lags_app = lags_app[lags_app!=0].astype(int)
    lags_chk_all = indices_to_rgsrs(lags_app, allw_lags, all_exog_rgsr)
    yp_fct = np.zeros([fct_win])
    pred_err = np.zeros([fct_win])
    
    y = (df_m[targ_dict['target']])
    y = y[::-1] # flip the vector
    y_obs = y
    y_obs = y_obs.interpolate()
    y_obs = np.array(y_obs)
    fct_var = np.zeros(len(lags_app))
    pred_win = len(coeffs) # coeffs len
#     y_obs = y[lags_app-1]
    j = 0
    for i in lags_chk_all:
            
        if str(i).isdigit(): # check if it is a lag or exog column
            fct_var[j]= y_obs[i]
        else:
            fct_var[j] = df_m[i].iloc[min(allw_lags)-1] # reads the column name in the dataframe specified by name="i"
        j+=1
#     fct_var = y
    for i in range(0,fct_win):
#         pdb.set_trace()
        yp_fct[i] = np.dot(fct_var,coeffs[lags_app])
        fct_var = np.append(yp_fct[i],fct_var[0:(len(fct_var)-1)])
    
    return yp_fct, pred_err

def multi_step_fct_exog(df_m, coeffs, lags_app, train_pred_err, allw_lags,targ_dict, ms_fct, win, Nb, bin_ed, uncer_anl=False):
    '''Using the data_frame, returns 1, 2,... ms_fct-week ahead forecast and also provides the uncertainty in estimation using bootstrap method if uncer_anl=True'''
    yp_fct=np.zeros(ms_fct)
    yb_fct=np.zeros([ms_fct,Nb])
    log_scr = np.zeros(ms_fct)
    bn_mat_bst = np.zeros([len(bin_ed)-1, ms_fct])
    bn_mat_Gaussker = np.zeros([len(bin_ed)-1, ms_fct])
    for wks in range(1,ms_fct+1):
#         pdb.set_trace()
        yp_fct[wks-1],  err = ARLR_fct_exog(coeffs[wks-1,:],df_m,lags_app[wks-1,:],1, wks, allw_lags, targ_dict)
        #train_pred_err[wks-1,:] = np.roll(train_pred_err[wks-1,:],1)# update error vector for uncertainty analy.
        #train_pred_err[wks-1,0] = data_test[wks-1]-yp_fct[wks-1]
#         pdb.set_trace()
        if uncer_anl:
#             pdb.set_trace()
#            yb_fct[wks-1,:] = fct_uncert(df_m, train_pred_err[wks-1,:],coeffs[wks-1,:],lags_app[wks-1,:], win, Nb)
#            log_scr[wks-1], bn_mat_bst[:, wks-1] = uncer_scr(yb_fct[wks-1,:], yp_fct[wks-1], ms_fct, Nb, bin_ed,1e-5)
            bn_mat_Gaussker[:, wks-1] = uncer_Gaussker(yp_fct[wks-1], ms_fct,train_pred_err[wks-1,:], bin_ed, 1e-5)
        print('Week: {}, Fct: {}, Bs: {}, log_scr: {}'.format(wks,np.exp(yp_fct[wks-1]), np.mean(np.exp(yb_fct[wks-1,:])), log_scr[wks-1]))
    return np.exp(yp_fct), yb_fct, log_scr, bn_mat_bst.reshape([131,ms_fct]), bn_mat_Gaussker.reshape([131,ms_fct]), train_pred_err

def rgsrs_ARLR(coeffs, lags, targ_dict, ews):    
    all_exog_rgsr = get_exog_reg(targ_dict)
    lags_chk = list(np.array(lags).astype(int)) # lags pertaining to AR coeffs
    lags_chk_all = lags_chk+all_exog_rgsr
    all_rgsr = lags_chk+all_exog_rgsr
    lags_app = []
    
    
    
    init_lags_len = len(lags_chk_all)
    df = pd.DataFrame(columns=all_rgsr)
    ms_fct_len = coeffs.shape[0]
    for p in range(0,4):
        for k in range(0,init_lags_len):
            err_m = np.zeros([len(lags_chk_all)])
            stnry = np.zeros([len(lags_chk_all)])
            jj = 0
            for i in lags_chk_all:            
                if str(i).isdigit(): # check if it is a lag or exog column
                    df.loc[p, i] = coeffs[p,i]
                else:
                    df.loc[p, i] = coeffs[p,rgsr_to_ind(i,lags,all_exog_rgsr)]

    #             pdb.set_trace()
                jj+=1
        df['DATE'] = pd.to_datetime(ews.startdate())
    return df

def ARLR_exog_module(df_m, targ_dict, ews, fct_weeks, allw_lags_f):
    config = configparser.ConfigParser()
    config_file = pkg_resources.resource_filename(__name__, 'config.ini')
    config.read(config_file) 
    
    ews_train = ews
    ews_test = ews   
    #df_train = df[(df['DATE']<=pd.to_datetime(ews_train.startdate()))]
    #df_train[target] = np.array(df_train[target],float)
    #df_train[target] = df_train[target].replace(0,1e-2) # check if zeros are there in ILI data as we take log
    #df_test = df[(df['DATE']>=pd.to_datetime(ews_train.startdate()))]
    #df_test[target] = np.array(df_test[target],float)
    #df_test[target] = df_test[target].replace(0,1e-2)
    #
    #train = np.log(np.array(df_train[target],'float').astype(float))
    #test = np.log(np.array(df_test[target],'float').astype(float))
    #train = pd.Series(train)
    #train.index = df_train['DATE']
    #test = pd.Series(test)
    #test.index = df_test['DATE']
    # Multi-step forecast
    
    
#     df_m  = ARLR_regressor(df, df_wtr, df_ght, region,targ_dict, ews)
#     if ar_diff:
#         a,b = get_stat_comp(df_m[targ_dict['target']], targ_dict, 208)
#         df_m.loc[:,'%UNWEIGHTED ILI'] = a.loc[:,]
#     # # adf['lag_comp'] = 0
#         df_m.loc[:,'lag_comp'] = b.loc[:,]
#     else:
#         df_m.loc[:,'%UNWEIGHTED ILI'] = a.loc[:,] = df
#         df_m.loc[:,'lag_comp'] = 0
    
    win = int(config['Forecasting']['win']) # Length of the historial training data to be considered
    
    fut_wks = int(config['Forecasting']['fut_wks']) # Number of weeks ahead to forecast from training data 
    ms_fct = fct_weeks # For every forecast week, give additional ms_fct weeks forecast
    
    test_win = fut_wks+ms_fct # Number of true value to be fetched (testing accuracy)
    exp_max_lags =  int(config['Forecasting']['exp_max_lags'])# expected maximum lags to be considered in the model
    llr_tol=1e-4 # log-likelihood tolerance
    
    # Uncertainty analysis
    uncer_anl = 1#int(config['CDC']['uncer_anl'])
    Nb = int(config['CDC']['Nb'])
    # create bins
    n_bins=int(config['CDC']['n_bins'])
    

    bin_ed = get_bin()
    
    # Check data for stationarity in the training data with padding
    #train_win = train[-1:(-win-exp_max_lags-1):-1] # training samples in the window period + buffer
    
    
    #result = adfuller(train_win)
    #print(result)
    #if result[1] < 0.05:
    #    print('p-val of ADF test %e' %result[1])
    #    print('Stationary signal')
    # plt.plot(train_win)
    # Check seasonality
    #season_ind = get_season(train_win,fft_len=1024,figs=False)
    # train the model
    
    max_lags = np.max(allw_lags_f)
    max_lags_with_rgsr = max_lags+len(get_exog_reg(targ_dict)) # No. of AR lags + number of exog. vr 
    coeffs=np.zeros([ms_fct,max_lags_with_rgsr+1])
    train_pred_err=np.zeros([ms_fct, win])
    yp_train=np.zeros([ms_fct, win]) 
    lags_app=np.zeros([ms_fct,max_lags_with_rgsr+1])
    # Train to obtain ARLR coeffs for all specified multi-step forecast:
    # Ex: For 1-step forecast, consider data from t-1 to t-p for training: ms_fct = 1
    # for 4-step forecast, consider data for t-4 to t-p for training: ms_fct = 4 
    # similarly for 1 season, ms_fct = 52
    for wks in range(1,ms_fct+1):
        allw_lags = allw_lags_f[(wks-1):]#np.arange(wks,max_lags+1)#np.append(np.arange(wks,5),52)#
        coeffs_temp, yp_train_temp, tr_tp1, train_pred_err_temp, lags_temp = ARLR_model_exog(df_m, allw_lags, targ_dict, win, 1e-3) #ARLR_model(train,allw_lags,win,llr_tol)
        yp_train[wks-1,:] = yp_train_temp
        train_pred_err[wks-1,:] = train_pred_err_temp
        all_exog_rgsr = get_exog_reg(targ_dict) 
        rgsrs_indices = rgsrs_to_indices(lags_temp, allw_lags, all_exog_rgsr)
        lags_app[wks-1,rgsrs_indices] = rgsrs_indices
        coeffs[wks-1,:] = coeffs_temp
    
    
    yp_fct=np.zeros([fut_wks,ms_fct])
    yp_fct_ch=np.zeros([fut_wks,ms_fct])
    yb_fct=np.zeros([fut_wks,ms_fct,Nb])
    log_scr = np.zeros([fut_wks, ms_fct])
    bn_mat_bst = np.zeros([fut_wks, len(bin_ed)-1, ms_fct])
    bn_mat_Gaussker = np.zeros([fut_wks, len(bin_ed)-1, ms_fct])
#     # Once trained, use the coeffs to forecast multi-steps given data frame
    
    # For obtaining uncertainty in forecast estimates (using Boot strapping), choose uncer_anl = True,  
#     data_test = []#test
    for new_wks in np.arange(0,fut_wks):
#         data_frame = data_frame.append(test[new_wks:(new_wks+1)])
#         data_test = data_test[1:]
        yp_fct[new_wks,:], yb_fct[new_wks,:,:], log_scr[new_wks,:], bn_mat_bst[new_wks, :,:], bn_mat_Gaussker[new_wks, :,:], train_pred_err = multi_step_fct_exog(df_m, coeffs, lags_app, train_pred_err, allw_lags,targ_dict, ms_fct, win, Nb, bin_ed, uncer_anl)
    
    seas, err_p = ARLR_fct_exog(coeffs[0,:],df_m,lags_app[0,:],30,0, allw_lags, targ_dict)
#     seas = 0
    return (yp_fct), bn_mat_bst, bn_mat_Gaussker, seas, lags_app,coeffs



#def ARLR_exog_module(df, df_wtr, df_ght, region, targ_dict, ews, fct_weeks, allw_lags_f):
#    config = configparser.ConfigParser()
#    config_file = pkg_resources.resource_filename(__name__, 'config.ini')
#    config.read(config_file) 
#    
#    ews_train = ews
#    ews_test = ews   
#    #df_train = df[(df['DATE']<=pd.to_datetime(ews_train.startdate()))]
#    #df_train[target] = np.array(df_train[target],float)
#    #df_train[target] = df_train[target].replace(0,1e-2) # check if zeros are there in ILI data as we take log
#    #df_test = df[(df['DATE']>=pd.to_datetime(ews_train.startdate()))]
#    #df_test[target] = np.array(df_test[target],float)
#    #df_test[target] = df_test[target].replace(0,1e-2)
#    #
#    #train = np.log(np.array(df_train[target],'float').astype(float))
#    #test = np.log(np.array(df_test[target],'float').astype(float))
#    #train = pd.Series(train)
#    #train.index = df_train['DATE']
#    #test = pd.Series(test)
#    #test.index = df_test['DATE']
#    # Multi-step forecast
#    df_m  = ARLR_regressor(df, df_wtr, df_ght, region,targ_dict, ews)
#    
#    win = int(config['Forecasting']['win']) # Length of the historial training data to be considered
#    
#    fut_wks = int(config['Forecasting']['fut_wks']) # Number of weeks ahead to forecast from training data 
#    ms_fct = fct_weeks # For every forecast week, give additional ms_fct weeks forecast
#    
#    test_win = fut_wks+ms_fct # Number of true value to be fetched (testing accuracy)
#    exp_max_lags =  int(config['Forecasting']['exp_max_lags'])# expected maximum lags to be considered in the model
#    llr_tol=1e-4 # log-likelihood tolerance
#    
#    # Uncertainty analysis
#    uncer_anl = 0#int(config['CDC']['uncer_anl'])
#    Nb = int(config['CDC']['Nb'])
#    # create bins
#    n_bins=int(config['CDC']['n_bins'])
#    
#
#    bin_ed = get_bin()
#    
#    # Check data for stationarity in the training data with padding
#    #train_win = train[-1:(-win-exp_max_lags-1):-1] # training samples in the window period + buffer
#    
#    
#    #result = adfuller(train_win)
#    #print(result)
#    #if result[1] < 0.05:
#    #    print('p-val of ADF test %e' %result[1])
#    #    print('Stationary signal')
#    # plt.plot(train_win)
#    # Check seasonality
#    #season_ind = get_season(train_win,fft_len=1024,figs=False)
#    # train the model
#    
#    max_lags = np.max(allw_lags_f)
#    max_lags_with_rgsr = max_lags+len(get_exog_reg(targ_dict)) # No. of AR lags + number of exog. vr 
#    coeffs=np.zeros([ms_fct,max_lags_with_rgsr+1])
#    train_pred_err=np.zeros([ms_fct, win])
#    yp_train=np.zeros([ms_fct, win]) 
#    lags_app=np.zeros([ms_fct,max_lags_with_rgsr+1])
#    # Train to obtain ARLR coeffs for all specified multi-step forecast:
#    # Ex: For 1-step forecast, consider data from t-1 to t-p for training: ms_fct = 1
#    # for 4-step forecast, consider data for t-4 to t-p for training: ms_fct = 4 
#    # similarly for 1 season, ms_fct = 52
#    for wks in range(1,ms_fct+1):
#        allw_lags = allw_lags_f[(wks-1):]#np.arange(wks,max_lags+1)#np.append(np.arange(wks,5),52)#
#        coeffs_temp, yp_train_temp, tr_tp1, train_pred_err_temp, lags_temp = ARLR_model_exog(df_m, allw_lags, targ_dict, win, 1e-3) #ARLR_model(train,allw_lags,win,llr_tol)
#        yp_train[wks-1,:] = yp_train_temp
#        train_pred_err[wks-1,:] = train_pred_err_temp
#        all_exog_rgsr = get_exog_reg(targ_dict) 
#        rgsrs_indices = rgsrs_to_indices(lags_temp, allw_lags, all_exog_rgsr)
#        lags_app[wks-1,rgsrs_indices] = rgsrs_indices
#        coeffs[wks-1,:] = coeffs_temp
#    
#    
#    yp_fct=np.zeros([fut_wks,ms_fct])
#    yp_fct_ch=np.zeros([fut_wks,ms_fct])
#    yb_fct=np.zeros([fut_wks,ms_fct,Nb])
#    log_scr = np.zeros([fut_wks, ms_fct])
#    bn_mat_bst = np.zeros([fut_wks, len(bin_ed)-1, ms_fct])
#    bn_mat_Gaussker = np.zeros([fut_wks, len(bin_ed)-1, ms_fct])
##     # Once trained, use the coeffs to forecast multi-steps given data frame
#    
#    # For obtaining uncertainty in forecast estimates (using Boot strapping), choose uncer_anl = True,  
#    
##     data_test = []#test
#    for new_wks in np.arange(0,fut_wks):
##         data_frame = data_frame.append(test[new_wks:(new_wks+1)])
##         data_test = data_test[1:]
#        yp_fct[new_wks,:], yb_fct[new_wks,:,:], log_scr[new_wks,:], bn_mat_bst[new_wks, :,:], bn_mat_Gaussker, train_pred_err = multi_step_fct_exog(df_m, coeffs, lags_app, train_pred_err, allw_lags,targ_dict, ms_fct, win, Nb, bin_ed, uncer_anl)
#    
#    #seas, err_p = ARLR_fct_exog(coeffs[0,:],train,lags_app[0,:],20, 0)
#    seas = 0
#    return (yp_fct), bn_mat_bst, bn_mat_Gaussker, seas, lags_app,coeffs

def fct_uncert(train, pred_err,coeffs, lags_app, win, Nb=1000):
    '''Provides Nb forecasts based on the bootstrap samples generated by create_bootstrap fucntion'''
#     pdb.set_trace()
    lags_app = lags_app[lags_app!=0].astype(int)
    yb_mat = np.zeros([Nb,win])
    yb_fct = np.zeros(Nb)
    i=0 # discard outlier sample values
    t_o = 0 # time out variable
    while i < Nb:
#         pdb.set_trace()
        yb_mat[i,:],ind = create_bootstrap(train, pred_err, coeffs, lags_app, win)
#         pdb.set_trace()
        yb_fct[i], pred = ARLR_fct(coeffs,yb_mat[i,:],lags_app,1,1)
        if np.abs(yb_fct[i]) > 100 and t_o<10000:
            t_o+=1
            i+=1
            #pdb.set_trace()
            continue
        else:
            i+=1
#     pdb.set_trace()
    return yb_fct

def uncer_scr(yb_fct, yp_fct, ms_fct, N_b, bin_ed,alp=1e-5):
    '''Puts the bootstrap samples provided by fct_uncert function into bins with bin edges given by bin_ed'''
#     be = np.arange(0,n_bins,.1)
#     be = np.append(be,20)
    bn_mat = np.zeros([len(bin_ed)-1, ms_fct])
    log_scr = 0#np.zeros(ms_fct)
#     pdb.set_trace()
#         plt.subplot(ms_fct,1,i+1)
    bn = np.histogram(np.exp(yb_fct[:]),bins=bin_ed)# plt.plot(y_obs)
    probs = dict(zip(np.round(bn[1],1),bn[0]/N_b))
    #log_scr = np.log(probs[np.floor(np.exp(test)*10)/10.])
    bn_mat = (1-alp)*bn[0]/N_b+alp
    return log_scr, bn_mat

def uncer_Gaussker(yp_fct, ms_fct, pred_err, bin_ed, alp=1e-5):
    '''Puts the bootstrap samples provided by fct_uncert function into bins with bin edges given by bin_ed'''
    std_err = np.std(pred_err)
    bn_mat = np.exp(-((bin_ed[:-1]-np.exp(yp_fct))/(2*3*std_err))**2)
#         plt.subplot(ms_fct,1,i+1)
    #log_scr = np.log(probs[np.floor(np.exp(test)*10)/10.])
    bn_mat = (1-alp)*bn_mat/np.sum(bn_mat)+(alp)
    return bn_mat

def create_bootstrap(train, pred_err, coeffs, lags_app, win):
    '''Create bootstrap samples given the prediction error and the AR model'''
#     pdb.set_trace()
    lags_app = lags_app[lags_app!=0].astype(int)
    y = np.flip(train)
    ind = y.index
    err_b = np.random.choice(pred_err,win)
    max_lag = np.max(lags_app).astype('int')    
    yb_temp = np.zeros((win+max_lag))
    yb_temp[win:] = y[win:(win+max_lag)]
    for i in range(win):
        yb_temp[(win-i-1)] = np.dot(yb_temp[win-(i+1)+lags_app]+err_b[i], coeffs[lags_app])
#     pdb.set_trace()
    yb = yb_temp[0:win]
    ind = ind[0:win]
#     yp = yb[-1::-1]
#     ind = ind[-1::-1]
    return yb, ind


