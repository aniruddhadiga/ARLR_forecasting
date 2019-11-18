'''Contains functions that correspond to formatting the output of ARLR method with exogenous regressors for both accuweather and CDC'''
from statsmodels.tsa.ar_model import AR
import numpy as np
import statsmodels.api as sm
#import matplotlib.pyplot as plt
import pandas as pd
import pdb, os
from aw_micro import cdc_data
import datetime
#
def outputdistribution_bst(predictions,bn_mat, bins, region, target, directory, epi_wk):
    output = pd.DataFrame(columns=['Location', 'Target', 'Type', 'Unit', 'Bin_start_incl', 'Bin_end_notincl', 'Value'])
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
            df2.loc[i+14] = [region, "Season onset", "Bin", "week", i, i+1, .0765]

    for i in range(40, 53):
        if region.isdigit():
            df3.loc[i-40] = ["HHS Region " + region, "Season peak week", "Bin", "week", i, i+1, 0.03030303]
        else:
            df3.loc[i-40] = [region, "Season peak week", "Bin", "week", i, i+1, 0.03030303]
    for i in range(1, 21):
        if region.isdigit():
            df3.loc[i+13] = ["HHS Region " + region, "Season peak week", "Bin", "week", i, i+1, 0.03030303]
        else:
            df3.loc[i+13] = [region, "Season peak week", "Bin", "week", i, i+1, 0.03030303]
    for i in range(bn_mat.shape[0]):
        if region.isdigit():
            df4.loc[i+1] = ["HHS Region " + region, "Season peak percentage", "Bin", "percent", bins[i], bins[i+1], 0.007633588]
        else:
            df4.loc[i+1] = [region, "Season peak percentage", "Bin", "percent", bins[i], bins[i+1], 0.007633588]

    output = output.append(df2)
    output = output.append(df3)
    output = output.append(df4)

    for i in range(predictions.shape[0]):
            df = pd.DataFrame(columns=['Location', 'Target', 'Type', 'Unit', 'Bin_start_incl', 'Bin_end_notincl', 'Value'])
            hist = bn_mat[:,i]
            if region.isdigit():
                df.loc[0] = ["HHS Region " + region, str(i+1) + " wk ahead", "Point", "percent","","", predictions[i]]
            else:
                df.loc[0] = [region, str(i+1) + " wk ahead", "Point", "percent","","", predictions[i]]
    
            
            for j in range(hist.shape[0]):
                if region.isdigit():
                    df.loc[j+1] = ["HHS Region " + region, str(i+1) + " wk ahead", "Bin", "percent", bins[j], bins[j+1], hist[j]]
                else:
                    df.loc[j+1] = [region, str(i+1) + " wk ahead", "Bin", "percent", bins[j], bins[j+1], hist[j]/10]
            
            
            output = output.append(df)
        #pdb.set_trace()
    #Location Target Type Unit Bin_start_incl Bin_end_notincl Value
    filename = 'EW' +'{:02}'.format(epi_wk.week) + '-' + str(epi_wk.year)+ '-FluX_ARLR' +'.csv'
    filepath = os.path.join(directory,filename)
    if not os.path.isfile(filepath):
        output.to_csv(filepath, index=False) 
    else:
        output.to_csv(filepath, mode='a', index=False, header=False)  
    return

def outputdistribution_Gaussker(predictions,bn_mat, bins, region, directory, epi_wk):
    if region=='National':
        region = 'US National'
    if (region[len(region)-1]).isdigit():
        region = 'HHS Region ' + region[len(region)-1]
    output = pd.DataFrame(columns=['Location', 'Target', 'Type', 'Unit', 'Bin_start_incl', 'Bin_end_notincl', 'Value'])
    df2 = pd.DataFrame(columns=['Location', 'Target', 'Type', 'Unit', 'Bin_start_incl', 'Bin_end_notincl', 'Value'])
    df3 = pd.DataFrame(columns=['Location', 'Target', 'Type', 'Unit', 'Bin_start_incl', 'Bin_end_notincl', 'Value'])
    df4 = pd.DataFrame(columns=['Location', 'Target', 'Type', 'Unit', 'Bin_start_incl', 'Bin_end_notincl', 'Value'])
    
    df2.loc[0] = [region,"Season onset", "Bin", "week", "none", "none", 0.029411765]

    for i in range(40,53):
        df2.loc[i-39] = [region, "Season onset", "Bin", "week", i, i+1, 0.029411765]
    
    for i in range(1, 21):
        df2.loc[i+14] = [region, "Season onset", "Bin", "week", i, i+1, 0.029411765]

    for i in range(40, 53):
        df3.loc[i-40] = [region, "Season peak week", "Bin", "week", i, i+1, 0.03030303]
    for i in range(1, 21):
        df3.loc[i+13] = [region, "Season peak week", "Bin", "week", i, i+1, 0.03030303]

    for i in range(bn_mat.shape[0]):
        df4.loc[i+1] = [region, "Season peak percentage", "Bin", "percent", bins[i], bins[i+1], 0.007633588]

    output = output.append(df2)
    output = output.append(df3)
    output = output.append(df4)

    for i in range(predictions.shape[0]):
            df = pd.DataFrame(columns=['Location', 'Target', 'Type', 'Unit', 'Bin_start_incl', 'Bin_end_notincl', 'Value'])
            hist = bn_mat[:,i]
            df.loc[0] = [region, str(i+1) + " wk ahead", "Point", "percent","","", predictions[i]]
    
            
            for j in range(hist.shape[0]):
                df.loc[j+1] = [region, str(i+1) + " wk ahead", "Bin", "percent", bins[j], bins[j+1], hist[j]]
            
            
            output = output.append(df)
    #Location Target Type Unit Bin_start_incl Bin_end_notincl Value
    filename = 'EW' +'{:02}'.format(epi_wk.week) + '-' + str(epi_wk.year)+ '-FluX_ARLR'+ '.csv'

    filepath = os.path.join(directory,filename)
    if not os.path.isfile(filepath):
        output.to_csv(filepath, index=False) 
    else:
        output.to_csv(filepath, mode='a', index=False, header=False)  
    return
    
def accu_output(predictions, region, accu_file, ews, st_fips_path):
    df_s = pd.read_csv(st_fips_path)
    st_fips_val = df_s[df_s['state_name']==region]['state']
    pred_state = []
    output = pd.DataFrame(columns=['date', 'area_id', 'ili'])
    for i in range(predictions.shape[0]):
        prediction_date = cdc_data.ew2date(ews.year,ews.week+i)
        pred_state.append({'date':prediction_date.isoformat(),'area_id':'{:02}'.format(st_fips_val.values[0]),'ili':predictions[i]})
        df_state = pd.DataFrame(pred_state)
    output=output.append(df_state)
    if not os.path.isfile(accu_file):
        output.to_csv(accu_file, index=False) 
    else:
        output.to_csv(accu_file, mode='a', index=False, header=False)  
    return

def outputdistribution_fromtemplate(predictions,bn_mat, bins, region, target, directory, epi_wk):
    if region=='National':
        region = 'US National'
    if (region[len(region)-1]).isdigit():
        region = 'HHS ' + region

    filename = 'EW' +'{:02}'.format(epi_wk.week) + '-' + str(epi_wk.year)+ '-FluX_ARLR'+ '.csv'

    filepath = os.path.join(directory,filename)
    if not os.path.isfile(filepath):
        tpl_path = 'data/Delphi_Uniform'
        if str(epi_wk.year) == '2014':
            tpl_file = 'EW01-2014-Delphi_Uniform.csv'#'EW' +'{:02}'.format(epi_wk.week) + '-' + str(epi_wk.year)+ '-Delphi_Uniform'+ '.csv'
        else:
            tpl_file = 'EW01-2011-Delphi_Uniform.csv'
    
        tpl_df = pd.read_csv(os.path.join(tpl_path,tpl_file))
        tpl_df.to_csv(filepath, index=False)
        df = pd.read_csv(filepath)
    else:
        df = pd.read_csv(filepath)
    
    for i in range(predictions.shape[0]):
        targ_week = '{} wk ahead'.format(i+1)
        mask1 = (df.Location==region)&(df.Target==targ_week)&(df.Type=='Point')
        mask2 = (df.Location==region)&(df.Target==targ_week)&(df.Type=='Bin')
        df.loc[mask1,'Value'] = predictions[i]
        df.loc[mask2,'Value'] = bn_mat[:,i]
    
    df.to_csv(filepath, mode='w', index=False)
    return

def outputdistribution_fromtemplate_for_FSN(predictions,bn_mat, bins, region, target, directory, epi_wk):
    if region=='National':
        region = 'US National'
    if (region[len(region)-1]).isdigit():
        region = 'HHS ' + region

    filename_FSN = 'EW' +'{:02}'.format(epi_wk.week) + '-' + str(epi_wk.year)+ '-FluX_ARLR'+ '.csv'

    filepath = os.path.join(directory,filename_FSN)
    if not os.path.isfile(filepath):
        tpl_path = 'data/Delphi_Uniform'
        if str(epi_wk.year) == '2014':
            tpl_file = 'EW01-2014-Delphi_Uniform.csv'#'EW' +'{:02}'.format(epi_wk.week) + '-' + str(epi_wk.year)+ '-Delphi_Uniform'+ '.csv'
        else:
            tpl_file = 'EW01-2011-Delphi_Uniform.csv'
    
        tpl_df = pd.read_csv(os.path.join(tpl_path,tpl_file))
        tpl_df.to_csv(filepath, index=False)
        df = pd.read_csv(filepath)
    else:
        df = pd.read_csv(filepath)
    
    for i in range(predictions.shape[0]):
        targ_week = '{} wk ahead'.format(i+1)
        mask1 = (df.Location==region)&(df.Target==targ_week)&(df.Type=='Point')
        mask2 = (df.Location==region)&(df.Target==targ_week)&(df.Type=='Bin')
        df.loc[mask1,'Value'] = predictions[i]
        df.loc[mask2,'Value'] = bn_mat[:,i]
    
    df.to_csv(filepath, mode='w', index=False)
    return
    
def outputdistribution_fromtemplate_for_FluSight(predictions,bn_mat, bins, region, target, directory, epi_wk, sub_date):
    if region=='National':
        region = 'US National'
    if (region[len(region)-1]).isdigit():
        region = 'HHS ' + region

    filename_FluSight = 'EW' +'{:02}'.format(epi_wk.week) + '-FluX_ARLR-'+ sub_date + '.csv'

    filepath = os.path.join(directory,filename_FluSight)
    if not os.path.isfile(filepath):
        tpl_path = 'data/Delphi_Uniform'
        if str(epi_wk.year) == '2014':
            tpl_file = 'EW01-2014-Delphi_Uniform.csv'#'EW' +'{:02}'.format(epi_wk.week) + '-' + str(epi_wk.year)+ '-Delphi_Uniform'+ '.csv'
        else:
            tpl_file = 'EW01-2011-Delphi_Uniform.csv'
    
        tpl_df = pd.read_csv(os.path.join(tpl_path,tpl_file))
        tpl_df.to_csv(filepath, index=False)
        df = pd.read_csv(filepath)
    else:
        df = pd.read_csv(filepath)
    
    for i in range(predictions.shape[0]):
        targ_week = '{} wk ahead'.format(i+1)
        mask1 = (df.Location==region)&(df.Target==targ_week)&(df.Type=='Point')
        mask2 = (df.Location==region)&(df.Target==targ_week)&(df.Type=='Bin')
        df.loc[mask1,'Value'] = predictions[i]
        df.loc[mask2,'Value'] = bn_mat[:,i]
    
    df.to_csv(filepath, mode='w', index=False)
    return

def outputdistribution_state_fromtemplate(predictions,bn_mat, bins, region, target, directory, epi_wk, sub_date):
    filename_FluSight = 'EW' +'{:02}'.format(epi_wk.week) + '-FluX_ARLR-'+'StateILI-' + sub_date + '.csv'

    filepath = os.path.join(directory,filename_FluSight)
    if not os.path.isfile(filepath):
        tpl_path = 'data/'
        if str(epi_wk.year) == '2014':
            tpl_file = 'stateili_submission_template_2019_2020.csv'#'EW' +'{:02}'.format(epi_wk.week) + '-' + str(epi_wk.year)+ '-Delphi_Uniform'+ '.csv'
        else:
            tpl_file = 'stateili_submission_template_2019_2020.csv'
    
        tpl_df = pd.read_csv(os.path.join(tpl_path,tpl_file))
        tpl_df.to_csv(filepath, index=False)
        df = pd.read_csv(filepath)
    else:
        df = pd.read_csv(filepath)
    
    for i in range(predictions.shape[0]):
        targ_week = '{} wk ahead'.format(i+1)
        mask1 = (df.Location==region)&(df.Target==targ_week)&(df.Type=='Point')
        mask2 = (df.Location==region)&(df.Target==targ_week)&(df.Type=='Bin')
        df.loc[mask1,'Value'] = predictions[i]
        df.loc[mask2,'Value'] = bn_mat[:,i]
    
    df.to_csv(filepath, mode='w', index=False)
    return

