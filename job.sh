#!/bin/bash
python scripts/forecast_ARLR_exog.py -b $1 -w 4 --out_folder $2 --ground_truth /project/biocomplexity/forecast/cdcfluview/ -v --CDC 1 --st_fips data/state_fips.csv --out_state output/predictions.txt --mode retro --accu_data_state data/data-aw-whole_20191108_1605-weekly-state.csv --accu_data_hhs data/data-aw-whole_20191108_1605-weekly-hhs.csv --accu_data_nat data/data-aw-whole_20191108_1605-weekly-nation.csv --mode retro --ght_data_state data/ght_state-201947.csv --ght_data_hhs data/ght_hhs-201947.csv --ght_data_nat data/ght_national-201947.csv --state_exog 1
#
