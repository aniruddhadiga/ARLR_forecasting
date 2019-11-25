import pytrends
import pandas as pd
from pandas.io.json import normalize
from pytrends.request import TrendReq
from datetime import date
from datetime import date
import epiweeks as epi
import sys
import pdb
filepath = sys.argv[1]
kw_list=['flu', 'cough', 'fever', 'influenza', 'cold']

def download_ght_by_country_today(kw_list):
    pytrend = TrendReq()
    pytrend.build_payload(kw_list, geo='US',timeframe='today 5-y')
    
    df = pytrend.interest_over_time()
    return df

def download_ght_by_states_today(kw_list):
    pytrend = TrendReq()

    states = ["AL","AK","AZ","AR","CA","CO","CT","DE","DC","FL","GA","HI","ID","IL","IN","IA","KS","KY","LA","ME","MT",
              "NE","NV","NH","NJ","NM","NY","NC","ND","OH","OK","OR","MD","MA","MI","MN","MS","MO","PA","RI","SC","SD",
              "TN", "TX","UT","VT","VA","WA","WV","WI","WY"]

    def state_ind_to_name():
        state_dict = {'US-AL': 'Alabama',
         'US-AK': 'Alaska',
         'US-AZ': 'Arizona',
         'US-AR': 'Arkansas',
         'US-CA': 'California',
         'US-CO': 'Colorado',
         'US-CT': 'Connecticut',
         'US-DE': 'Delaware',
         'US-DC': 'District of Columbia',
         'US-FL': 'Florida',
         'US-GA': 'Georgia',
         'US-HI': 'Hawaii',
         'US-ID': 'Idaho',
         'US-IL': 'Illinois',
         'US-IN': 'Indiana',
         'US-IA': 'Iowa',
         'US-KS': 'Kansas',
         'US-KY': 'Kentucky',
         'US-LA': 'Louisiana',
         'US-ME': 'Maine',
         'US-MD': 'Maryland',
         'US-MA': 'Massachusetts',
         'US-MI': 'Michigan',
         'US-MN': 'Minnesota',
         'US-MS': 'Mississippi',
         'US-MO': 'Missouri',
         'US-MT': 'Montana',
         'US-NE': 'Nebraska',
         'US-NV': 'Nevada',
         'US-NH': 'New Hampshire',
         'US-NJ': 'New Jersey',
         'US-NM': 'New Mexico',
         'US-NY': 'New York',
         'US-NC': 'North Carolina',
         'US-ND': 'North Dakota',
         'US-OH': 'Ohio',
         'US-OK': 'Oklahoma',
         'US-OR': 'Oregon',
         'US-PA': 'Pennsylvania',
         'US-RI': 'Rhode Island',
         'US-SC': 'South Carolina',
         'US-SD': 'South Dakota',
         'US-TN': 'Tennessee',
         'US-TX': 'Texas',
         'US-UT': 'Utah',
         'US-VT': 'Vermont',
         'US-VA': 'Virginia',
         'US-WA': 'Washington',
         'US-WV': 'West Virginia',
         'US-WI': 'Wisconsin',
         'US-WY': 'Wyoming'}
        return state_dict

    i=0
    df=pd.DataFrame()
    st_name = state_ind_to_name()
    
    for st_ind in states:
        pytrend.build_payload(kw_list, geo='US-'+st_ind,timeframe='today 5-y')
        print('region processed US-'+st_ind)
        if i==0:
            df1 = pytrend.interest_over_time()
            df1 = df1.assign(state=st_name['US-'+st_ind])
            df = df1
        else:
            df1 = pytrend.interest_over_time()
            df1 = df1.assign(state=st_name['US-'+st_ind])
            df = df.append(df1)
        i+=1
#             df = pytrend.interest_over_time()

#             df = df.append(df)
        
        filename = filepath + 'ght_state-'+ str(epi.Week.thisweek().year) + str(epi.Week.thisweek().week)+'.csv'
        df.to_csv(filename)
    return df

state_ght_csv = filepath+'ght_state-201947.csv'
cty_pops_csv = filepath+'countypops_2013.csv'
st_hhs_csv = filepath+'state_hhs_map.csv'
pdb.set_trace()
def download_ght_state_to_hhs_nat(state_ght_csv, cty_pops_csv, st_hhs_csv):
    df_ght = pd.read_csv(state_ght_csv)
    df_ct_pops = pd.read_csv(cty_pops_csv, names=['cty_fips','pops'], header=-1, delim_whitespace=True)
    df_ct_pops['st_fips'] = df_ct_pops.apply(lambda x: int(round(x['cty_fips']/1000)),axis=1)
    df_ct_pops.groupby('st_fips', as_index=False).sum()
    df_st_to_hhs = pd.read_csv(st_hhs_csv, names=['st_fips','hhs', 'st_abrv', 'state'], header=-1)
    df_st = df_ct_pops.groupby('st_fips',as_index=False).sum()
    df_st = df_st.merge(df_st_to_hhs)
    df_hhs_pop = df_st.groupby(['hhs'],as_index=False).sum()
    df_hhs_pop = df_hhs_pop.rename(columns={'pops':'hhs_pops'})
    df_hhs_pop = df_hhs_pop.drop(columns=['st_fips','cty_fips'])
    df_st = df_st.merge(df_hhs_pop)
    df_st['hhs_pops_wt'] = df_st.apply(lambda x: x['pops']/x['hhs_pops'],axis=1)
    df_ght = df_ght.merge(df_st)
    # # df_ght_hhs = df_ght.groupby(['date', 'hhs'], as_index=False).sum()
    df_ght_hhs = df_ght
    df_ght_hhs[['flu','cough','fever','influenza','cold','hhs_pops']] = df_ght_hhs.apply(lambda x:x[['flu','cough','fever','influenza','cold','hhs_pops']]*x['hhs_pops_wt'],axis=1)
    df_ght_hhs = df_ght_hhs.groupby(['date', 'hhs'], as_index=False).sum()
#     df_ght_hhs = df_ght_hhs.set_index('date')
    filename = filepath + 'ght_hhs-'+ str(epi.Week.thisweek().year) + str(epi.Week.thisweek().week)+'.csv'
    df_ght_hhs.to_csv(filename)
    df_ght_hhs['nat_pops_wt'] = df_ght_hhs.apply(lambda x: x['hhs_pops']/df_ght_hhs['hhs_pops'].unique().sum(),axis=1)
    df_ght_nat = df_ght_hhs
    df_ght_nat[['flu','cough','fever','influenza','cold','nat_pops']] = df_ght_hhs.apply(lambda x: x[['flu','cough','fever','influenza','cold','hhs_pops']]*x['nat_pops_wt'],axis=1)    
    df_ght_nat = df_ght_nat.groupby('date').sum()
#     df_ght_nat = df_ght_nat.set_index('date')
    filename = filepath + 'ght_national-'+ str(epi.Week.thisweek().year) + str(epi.Week.thisweek().week)+'.csv'
    df_ght_nat.to_csv(filename)
    return

download_ght_by_states_today(kw_list)
download_ght_state_to_hhs_nat(state_ght_csv, cty_pops_csv, st_hhs_csv)


