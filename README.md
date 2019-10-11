# ARLR_forecasting
cd scripts
python forecast_ARLR.py -b 2018EW40 -w 4 --out_state dump/ --ground_truth ../data/state/ILINet.csv -v --CDC 1 --test 1 --st_fips ../data/state_fips.csv

optional arguments:
  -h, --help            show this help message and exit
  -b FORECAST_FROM, --forecast_from FORECAST_FROM
                        a date EW format indicating first week to predict.
  -w WEEKS, --weeks WEEKS
                        number of weeks to predict
  --out_state OUT_STATE
                        CSV format output file of state predictions
  --out_county OUT_COUNTY
                        CSV format output file of county predictions
  --ground_truth GROUND_TRUTH
                        CSV file ("|") from CDC of state ILI levels
  --region_type REGION_TYPE
                        national, 1, 2,...,10, state
  --st_fips ST_FIPS     file of state fips and names
  --county_ids COUNTY_IDS
                        file of all county 5-digit fips
  --end_date END_DATE   date (yyyymmdd) of last ground truth data point
  --CDC CDC             CDC=0 means no uncertainty binning, CDC=1
  --test TEST           test mode, dumps the predictions in folder dump
  -v, --verbose         verbose logging
  -l LOG, --log LOG     log file, by default logs are written to standard
                        output
