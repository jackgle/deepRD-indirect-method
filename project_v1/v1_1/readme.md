
This section of the repo contains code for creating and evaluating a neural network model for indirect prediction of reference intervals.

The model is trained and evaluated using the RIbench dataset (https://cran.r-project.org/web/packages/RIbench/index.html)

The performance of the trained model is compared against refineR, the state-of-the-art statistical method for indirect RI estimation (https://cran.r-project.org/web/packages/refineR/index.html)

Steps to reproduce results:

1. Make sure you have compatible versions of Python, R, and the necessary Python packages:

    * Python >= 3.0
    * R >= 3.2.0

    The python packages used are listed in neural_net/requirements.txt

2. Generate RIbench samples

     1. run `Rscript RIbench/requirements.R` to install the necessary package
     2. run `Rscript RIbench/generateRIbench.R` to generate the simulated data samples

3. Train and evaluate the baseline neural network

    The baseline model is trained on all analytes (distribution types) simulated in the RIbench dataset. 

     1. run `python neural_net/experiment_baseline/create_training_data.py`
     2. run `python neural_net/experiment_baseline/train.py`
     3. run the notebook `neural_net/experiment_baseline/evaluate.ipynb` to generate statistics and plots related to model performance

4. Train and evaluate the leave-one-out cross-validation (LOO) models

    The LOO models are a set of 10 models, where for each model, a different analyte was withheld from the training data, and only used in the test set.

    The same dataset that was generated for the baseline model can be used, it will just be split into train/test sections in accordance with the LOO validation.

    4.1 run `python neural_net/experiment_loo/train.py`
    4.1 run the notebook `neural_net/experiment_loo/evaluate.ipynb` to generate statistics and plots related to model performance

5. Generate and evaluate refineR predictions for comparison

    refineR is the state-of-the-art statistical approach for indirect RI estimation based on the RIbench dataset. We will compare the results of this method to the neural network.

    5.1 run `Rscript refineR/requirements.R` to install the necessary package
    5.2 run `python refineR/create_file_list.py`
    5.3 run `Rscript refineR/predict.R`
    5.4 run the notebook `refineR/evaluate.ipynb` to generate statistics and plots related to model performance
    
6. Compare the methods on real data (e.g. Cancer Antigen 125)

    6.1 see notebook `neural_net/experiment_baseline/cancer_ag125/predict_age.ipynb`
    6.2 see notebook `refineR/cancer_ag125/predict_age.R`



