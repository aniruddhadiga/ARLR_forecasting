# ARLR_forecasting
python scripts/forecast_ARLR_exog.py -b 201840 -w 4 --out_folder dump/ --ground_truth data/state/ILINet.csv -v --CDC 0 --st_fips data/state_fips.csv --out_state output/predictions.txt --accu_data data/data-aw-cumulative_20191018_1620-weekly-state.csv

Following are the various options for argument:
For Accuweather, --out_state and --st_fips are mandatory. If exogenous regressors are required, --accu_data must be included with path to the weather data csv file. 
usage: forecast_ARLR_exog.py [-h] [-b FORECAST_FROM] -w WEEKS
                             [--out_state OUT_STATE] [--out_county OUT_COUNTY]
                             [--ground_truth GROUND_TRUTH]
                             [--region_type REGION_TYPE] [--st_fips ST_FIPS]
                             [--county_ids COUNTY_IDS] [--end_date END_DATE]
                             --CDC CDC [--test TEST] [-v] [-l LOG] -o
                             OUT_FOLDER [--accu_data ACCU_DATA]
                             [--ght_data GHT_DATA]
