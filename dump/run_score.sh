#!/bin/bash/
year=2018
sh ../R_load.sh
varibs=state
input_filename=${varibs}_exog/${year}/
score_filename=scores_$((year))_$varibs.csv 
Rscript score_submission.R $year $input_filename $score_filename
