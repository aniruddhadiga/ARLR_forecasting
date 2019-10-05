'''Contains functions that correspond to the ARLR method'''
from statsmodels.tsa.ar_model import AR
import numpy as np
import statsmodels.api as sm
#import matplotlib.pyplot as plt
import pandas as pd
import pdb
# ARLR functions
def ARLR_aug_phase(train,lags,win,llr_tol=1e-2):
    y = np.flip(train)
    y_obs = y[0:win]
    ind = y_obs.index
    lags = list(lags)
    lags_chk = np.array(lags).astype(int)
    lags_app = np.array([]).astype(int)
    err_old = np.linalg.norm(y_obs)#np.random.randn(win,1))
    tr_tp = np.zeros([win,len(lags)])
    init_lags_len = len(lags)
    for k in range(0,init_lags_len):
        err_m = np.zeros([len(lags_chk)])
        llr = np.zeros([len(lags_chk)])
        jj = 0
        for i in lags_chk:
#             pdb.set_trace()
            tr_tp[:,k] = y[i:(win+i)]
            tr_tp_mul = np.matmul(tr_tp[:,0:(k+1)].T, tr_tp[:,0:(k+1)])
            res = sm.OLS(y_obs,tr_tp[:,0:(k+1)]).fit()
            yp = res.predict()
            err_m[jj] = np.linalg.norm(y_obs-yp)
            llr[jj] = 2*np.log(err_old/err_m[jj])
#             pdb.set_trace()
            jj+=1            
        imax = np.argmax(llr)
    #     p_val = chi2.sf(llr[imax],1)
    #     print(p_val)
        if llr[imax] >= llr_tol:
#             pdb.set_trace()
            tr_tp[:,k] =y[lags_chk[imax]:(win+lags_chk[imax])]
            lags_app = np.append(lags_app, lags_chk[imax])
            lags_chk = np.delete(lags_chk, imax)
            err_old = err_m[imax]
#             pdb.set_trace()
        else:
#             pdb.set_trace()
            break
    res = sm.OLS(y_obs,tr_tp[:,0:(k)]).fit()
    yp = res.predict()
    return res, yp, y_obs, tr_tp, llr, err_old, lags_app, ind

def ARLR_red_phase(y,tr_tp,err_old, lags, res1, lags_app,ind, llr_tol=1e-2):
    tot_col = len(lags_app)
    temp_tr_tp = tr_tp[:,0:tot_col]
    for i in range(0,tot_col+1):
        init_lag_len_new = temp_tr_tp.shape[1]
        jj = 0 
        err_m = np.zeros([len(lags)])
        llr = np.zeros([len(lags)])
        for k in range(0,init_lag_len_new):
#             pdb.set_trace()
            res = sm.OLS(y,np.delete(temp_tr_tp,k,1)).fit()
            yp = res.predict()
            err_m[jj] = np.linalg.norm(y-yp)
            llr[jj] = 2*np.log(err_old/err_m[jj])
            jj+=1
        imin = np.argmin(llr)
        if (llr[imin]>llr_tol):
            temp_tr_tp = np.delete(temp_tr_tp,imin,1)     
            err_old = err_m[imin]
#             pdb.set_trace()
        else:
#             pdb.set_trace()
            break
    res = sm.OLS(y,temp_tr_tp).fit()
    yp = res.predict()
    pred_err = y-yp
    return res, yp, tr_tp, llr, pred_err, res1, lags_app, ind

def ARLR_model(train,allw_lags,win,llr_tol=1e-2):
    res, yp_aug, y_obs, tr_tp, llr, err_old, lags_app, ind = ARLR_aug_phase(train,allw_lags,win,llr_tol)
    resf, yp_red, tr_tp, llr, train_pred_err, res1, lags_app, ind = ARLR_red_phase(y_obs,tr_tp, err_old, allw_lags, res,lags_app,ind,llr_tol) # reduction phase
    yp_train = pd.Series(yp_red)
    yp_train.index = ind[:]
    coeffs=np.zeros(np.max(allw_lags)+1)
    coeffs[lags_app] = resf.params
    return coeffs, yp_train, tr_tp, llr, train_pred_err, lags_app
# forecast

def ARLR_fct(coeffs,train,test,lags_app,fct_win, wks):
#     pdb.set_trace()
    lags_app = lags_app[lags_app!=0].astype(int)
    yp_fct = np.zeros([fct_win,1])
    pred_err = np.zeros([fct_win,1])
    y = np.flip(train)
    # test = np.array(test).reshape(fct_win,1)
    pred_win = len(coeffs) # coeffs len
#     y_obs = y[lags_app-1]
    fct_var = y
    fct_var = fct_var[lags_app-1]
    for i in range(0,fct_win):
#         pdb.set_trace()
        yp_fct[i] = np.dot(fct_var,coeffs[lags_app])
        fct_var = np.append(yp_fct[i],fct_var[0:(len(fct_var)-1)])
        pred_err[i] = test[i]-yp_fct[i]
#         pdb.set_trace()
    #error metrics
#     pdb.set_trace()
    ind = test[(wks-1):(wks-1+fct_win)].index
    return yp_fct, ind, pred_err

def ARLR_err_met(yp_fct, test):
    fct_win = len(yp_fct)
    e = test[0:fct_win]-(yp_fct)
    RMSE = np.linalg.norm(e)/np.sqrt(fct_win)
    MAE = np.linalg.norm(e,1)/fct_win
    MAPE = np.sum(np.divide(np.abs(e),test))/fct_win
    return RMSE, MAE, MAPE


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

# Bootstrapping 
def fct_uncert(train,test, pred_err,coeffs, lags_app, win, Nb=1000):
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
        yb_fct[i], ind_fct, pred = ARLR_fct(coeffs,yb_mat[i,:],test,lags_app,1,1)
        if np.abs(yb_fct[i]) > 10 & t_o<10000:
            t_o+=1
            continue
        else:
            i+=1
#     pdb.set_trace()
    return yb_fct

def uncer_scr(yb_fct, test, yp_fct, ms_fct, N_b, bin_ed):
    '''Puts the bootstrap samples provided by fct_uncert function into bins with bin edges given by bin_ed'''
#     be = np.arange(0,n_bins,.1)
#     be = np.append(be,20)
    bn_mat = np.zeros([len(bin_ed)-1, ms_fct])
    log_scr = np.zeros(ms_fct)
#     pdb.set_trace()
#         plt.subplot(ms_fct,1,i+1)
    bn = np.histogram(np.exp(yb_fct[:]),bins=bin_ed)# plt.plot(y_obs)
    probs = dict(zip(np.round(bn[1],1),bn[0]/N_b))
    log_scr = np.log(probs[np.floor(np.exp(test)*10)/10.])
    bn_mat = bn[0]/N_b
#         pdb.set_trace()
    return log_scr, bn_mat

def multi_step_fct(data_frame, data_test, coeffs, lags_app, train_pred_err, ms_fct, win, Nb, bin_ed, uncer_anl=False):
    '''Using the data_frame, returns 1, 2,... ms_fct-week ahead forecast and also provides the uncertainty in estimation using bootstrap method if uncer_anl=True'''
    yp_fct=np.zeros(ms_fct)
    yb_fct=np.zeros([ms_fct,Nb])
    log_scr = np.zeros(ms_fct)
    bn_mat = np.zeros([len(bin_ed)-1, ms_fct])
    for wks in range(1,ms_fct+1):
#         pdb.set_trace()
        yp_fct[wks-1], ind1, err = ARLR_fct(coeffs[wks-1,:],data_frame,data_test,lags_app[wks-1,:],1, wks)
        train_pred_err[wks-1,:] = np.roll(train_pred_err[wks-1,:],1)# update error vector for uncertainty analy.
        train_pred_err[wks-1,0] = data_test[wks-1]-yp_fct[wks-1]
#         pdb.set_trace()
        if uncer_anl:
#             pdb.set_trace()
            yb_fct[wks-1,:] = fct_uncert(data_frame, data_test, train_pred_err[wks-1,:],coeffs[wks-1,:],lags_app[wks-1,:], win, Nb)
            log_scr[wks-1], bn_mat[:, wks-1] = uncer_scr(yb_fct[wks-1,:], data_test[wks-1], yp_fct[wks-1], ms_fct, Nb, bin_ed)
        print('Week: {}, Fct: {}, True: {}, Bs: {}, log_scr: {}'.format(wks, np.exp(yp_fct[wks-1]), np.exp(data_test[wks-1]), np.mean(np.exp(yb_fct[wks-1,:])), log_scr[wks-1]))
    return yp_fct, yb_fct, log_scr, bn_mat, train_pred_err

def outputdistribution(predictions,bn_mat, bin_ed, region, target, stdev, directory, week):
    output = pd.DataFrame(columns=['Location', 'Target', 'Type', 'Unit', 'Bin_start_incl', 'Bin_end_notincl', 'Value'])
    for i, prediction in enumerate(predictions):
        df = pd.DataFrame(columns=['Location', 'Target', 'Type', 'Unit', 'Bin_start_incl', 'Bin_end_notincl', 'Value'])
        hist = bn_mat[:,i]
        if region.isdigit():
            df.loc[0] = ["HHS Region " + region, str(i+1) + " wk ahead", "Point", "percent", "NA", "NA", prediction]
        else:
            df.loc[0] = [region, str(i+1) + " wk ahead", "Point", "percent", "NA", "NA", prediction]

        
        for j, element in enumerate(hist):
            if region.isdigit():
                df.loc[j+1] = ["HHS Region " + region, str(i+1) + " wk ahead", "Bin", "percent", bins[j], bins[j+1], hist[j]/10]
            else:
                df.loc[j+1] = [region, str(i+1) + " wk ahead", "Bin", "percent", bins[j], bins[j+1], hist[j]/10]
        
        
        output = output.append(df)
    
    df2 = pd.DataFrame(columns=['Location', 'Target', 'Type', 'Unit', 'Bin_start_incl', 'Bin_end_notincl', 'Value'])
    df3 = pd.DataFrame(columns=['Location', 'Target', 'Type', 'Unit', 'Bin_start_incl', 'Bin_end_notincl', 'Value'])
    df4 = pd.DataFrame(columns=['Location', 'Target', 'Type', 'Unit', 'Bin_start_incl', 'Bin_end_notincl', 'Value'])
    
    if region.isdigit():
        df2.loc[0] = ["HHS Region " + region, "Season onset", "Bin", "week", "none", "none", 0.029411765]
    else:
        df2.loc[0] = [region,"Season onset", "Bin", "week", "none", "none", 0.029411765]

    for i in range(40,53):
        if region.isdigit():
            df2.loc[i-39] = ["HHS Region " + region, "Season onset", "Bin", "week", i, i+1, 0.029411765]
        else:
            df2.loc[i-39] = [region, "Season onset", "Bin", "week", i, i+1, 0.029411765]
    for i in range(1, 21):
        if region.isdigit():
            df2.loc[i+14] = ["HHS Region " + region, "Season onset", "Bin", "week", i, i+1, 0.029411765]
        else:
            df2.loc[i+14] = [region, "Season onset", "Bin", "week", i, i+1, 0.029411765]

    for i in range(40,53):
        if region.isdigit():
            df3.loc[i-40] = ["HHS Region " + region, "Season peak week", "Bin", "week", i, i+1, 0.03030303]
        else:
            df3.loc[i-40] = [region, "Season peak week", "Bin", "week", i, i+1, 0.03030303]
    for i in range(1, 21):
        if region.isdigit():
            df3.loc[i+13] = ["HHS Region " + region, "Season peak week", "Bin", "week", i, i+1, 0.03030303]
        else:
            df3.loc[i+13] = [region, "Season peak week", "Bin", "week", i, i+1, 0.03030303]

    for i, element in enumerate(hist):
        if region.isdigit():
            df4.loc[i+1] = ["HHS Region " + region, "Season peak percentage", "Bin", "percent", bins[i], bins[i+1], 0.007633588]
        else:
            df4.loc[i+1] = [region, "Season peak percentage", "Bin", "percent", bins[i], bins[i+1], 0.007633588]

    output = output.append(df2)
    output = output.append(df3)
    output = output.append(df4)

    
    #Location Target Type Unit Bin_start_incl Bin_end_notincl Value
    
    output.to_csv(directory + "/EW" + str(week).zfill(2) + ".csv", index=False)    
    return
