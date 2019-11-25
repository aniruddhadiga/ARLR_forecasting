#!/bin/bash

#SBATCH --job-name=NAS
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=bii
#SBATCH --account=biocomplexity
#SBATCH --output=log
#SBATCH --time=8:00:00
#SBATCH --exclude=udc-aj37-15c1,udc-aj37-16c1,udc-aj40-16c0,udc-aj40-17c0,udc-aj40-17c1,udc-aj40-18c0,udc-aj37-17c0,udc-aj38-3c0,udc-aj38-9c0
# Below here enter the commands to start your job

#module add Python/2.7.14-foss-2018a
#hostname
export CONDA_ROOT="/home/$USER/miniconda3"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate fct

python scripts/forecast_ARLR_exog.py -b $1 -w 4 --out_folder dump/all_lag_exog/2017/ --ground_truth /project/biocomplexity/forecast/cdcfluview/ -v --CDC 1 --st_fips data/state_fips.csv --out_state output/predictions.txt --mode retro --ght_data_state data/ght_state-201947.csv --ght_data_hhs data/ght_hhs-201947.csv --ght_data_nat data/ght_national-201947.csv --accu_data_state data/data-aw-whole_20191108_1605-weekly-state.csv --accu_data_hhs data/data-aw-whole_20191108_1605-weekly-hhs.csv --accu_data_nat data/data-aw-whole_20191108_1605-weekly-nation.csv
#python scripts/retro_forecast_ARLR.py -b $1 -w 4 --out_folder int_retro_output/ --ground_truth /project/biocomplexity/forecast/cdcfluview/ -v --CDC 1 --test 1


#--accu_data_state data/data-aw-whole_20191108_1605-weekly-state.csv --accu_data_hhs data/data-aw-whole_20191108_1605-weekly-hhs.csv --accu_data_nat data/data-aw-whole_20191108_1605-weekly-nation.csv --ght_data_state data/ght_state-201947.csv --ght_data_hhs data/ght_hhs-201947.csv --ght_data_nat data/ght_national-201947.csv  
