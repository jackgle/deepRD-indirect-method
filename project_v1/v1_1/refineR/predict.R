library(refineR)

dr = '../data/RIbench/'

overwrite = 0

# set.seed(123) # random seed for shuffling

cat('Preparing output folder\n')
# create folders for refineR predictions
analytes = list.files(paste0(dr,'Data/'))
if (overwrite) { unlink('./refineR_predictions/', recursive = TRUE) }
if (!dir.exists('./refineR_predictions/')) { dir.create('./refineR_predictions/') }
for (an in analytes) {
    if (!dir.exists(paste0('./refineR_predictions/', an))){
        dir.create(paste0('./refineR_predictions/', an))
    }
}

cat('Reading file list\n')
files_test <- read.csv('./files_list.csv')$files
files_test <- sample(files_test) # randomize list to support estimating performance midway
cat('Number of files:', length(files_test), '\n')

cat('Predicting\n')
# generate refineR predictions
for (file in files_test) {
    print(file)
    
    # format file name
    split_string <- strsplit(file, "/")[[1]]
    last_two_elements <- tail(split_string, 2)
    outfile <- paste(last_two_elements, collapse = "/")
    
    # if the file doesn't already exist
    if (!file.exists(paste0('./refineR_predictions/', outfile))) {
    
        data <- read.csv(file, header=FALSE)$V1 # read simulated values
        fit <- findRI(data) # use refineR to get model fit
        result <- getRI(fit, Scale='original') # get RI estimate from model

        write.csv(result, paste0('./refineR_predictions/', outfile))
        
    }
}
