#import FluSight library
library(FluSight)

#set the working directory to be the top level of all CSV files or something

setwd("/project/biocomplexity/aniadiga/Forecasting/ARLR_forecasting/output/ARLR_GaussKernel/")

#downloads the correct scores from the FluSight API

truth <- create_truth(fluview = T, year = 2018)
exp_truth <- expand_truth(truth, week_expand = 1, percent_expand = 5)

#generates a list of all CSV files, including all subdirectories

files <- list.files(path="2018/", pattern="*.csv", full.names=FALSE, recursive=TRUE)

#Creates the column headings

write.table(matrix(c("location", "target", "score", "forecast_week", "competition_week", "skill", "model"), nrow=1, ncol=7, byrow=TRUE ), 'scores_2018_exp_v2.csv', append=FALSE, sep=",", row.names=FALSE, col.names=FALSE)

#Scores each CSV file and appends the scores to the scoresheet

lapply(files, function(x) {
  entry <- read_entry(paste("2018/", x, sep="")) # Read file
  verify_entry(entry)
  print(entry)
  model <- dirname(x)

  exact_scores <- score_entry(entry, truth)
  expand_scores <- score_entry(entry, exp_truth)

  expand_scores$competition_week <- if(expand_scores$forecast_week < 42) expand_scores$forecast_week+10 else expand_scores$forecast_week-42

  expand_scores$skill <- exp(expand_scores$score)

  expand_scores$model <- model


  write.table(expand_scores,'scores_2018_exp_v2.csv', append=TRUE, sep=",", row.names=FALSE, col.names=FALSE)
})
