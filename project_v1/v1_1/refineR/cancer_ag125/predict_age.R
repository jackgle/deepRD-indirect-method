library('refineR')

# path to a directory holding CSVs with CA-125 values for each age group
dir <- paste0('../../neural_net/experiment_baseline/cancer_ag125/data/age_binned/')
file_list <- list.files(path = dir)
file_list <- file.path(dir, file_list)

# check file list
print(file_list)

# for holding predicted CSV files, lower limits, and upper limits
files = c()
llims = c()
ulims = c()

execution_times <- numeric()

for (file in sort(file_list)) { # loop file list
  
  data <- read.csv(file) # read simulated values
  data <- data[,2]

  if (length(data) <= 10) { # if there are less than 10 samples, print the filename and use NaN for predictions
    cat(file, '\n')
    cat(length(data), " values\n")
    files <- c(files, file)
    llims <- c(llims, 'nan')
    ulims <- c(ulims, 'nan')
  }

  else {
    
    start_time <- Sys.time()
    
    fit <- findRI(data, model='modBoxCox') # fit the refineR model
    result <- getRI(fit, Scale='original') # get RI estimate from model
  
    end_time <- Sys.time()
    iteration_time <- end_time - start_time # check processing time

    cat(file,'\n') # print data stats
    cat(length(data), " values\n")
    cat(result$PointEst,'\n') # print estimated RI
    cat('max value: ', max(data), '\n\n') # print max data value
    cat('iteration time: ',iteration_time) # print processing time
  
    # append to results
    files <- c(files, file)
    llims <- c(llims, result$PointEst[1])
    ulims <- c(ulims, result$PointEst[2])
  }
  
}

# save results
results <- data.frame(file = files, ll = llims, ul = ulims)
write.csv(results, file = paste0('refineR_results_2param.csv'), row.names = FALSE)


