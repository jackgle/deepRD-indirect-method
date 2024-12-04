import os
import numpy as np
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture


# define GMM prediction
def predict_gmm(data, max_n_components=4, init_weights=True, bic_logic=True, target_n_components=None):

    fits = [] # for saving fits with different n_components
    bics = [] # for saving BIC of each fit
    for n_components in range(1,max_n_components+1): # test several numbers of components
        
        weights_init=None # when testing >1 component, initialize weights so one component has majority (reference)
        if n_components>1 and init_weights:
            weights_init=[0.55, *(0.45/(n_components-1),)*(n_components-1)] 
        
        gm = GaussianMixture( # fit GMM
            n_components=n_components, 
            weights_init=weights_init,
            max_iter=1000,
            random_state=0
        ).fit(data)
        
        fits.append(gm) # save results
        bics.append(gm.bic(data))

    if target_n_components is not None:
        gm = fits[target_n_components-1]
    elif bic_logic:
#         if 1-min(bics)/max(bics)<0.03: # if there is not a significant difference in BICs use 1 component - likely a very low pathological fraction
#             gm = fits[0]
#         else:
#             gm = fits[np.argmin(bics)] # get best fit based on BIC
        gm = fits[best_fit(bics)-1]
    else:
        gm = fits[np.argmin(bics)] # get best fit based on BIC

    best_component = np.argmax(gm.weights_) # get main component based on weights
    
    mean = gm.means_[best_component] # retrieve statistics
    cov = gm.covariances_[best_component] # for 2d outputs
    
    result_dict = {
        'mean': mean, 
        'covariance': cov,
        'gmm_model': gm
    }
    
    return result_dict

def best_fit(bics, improvement_per_component=0.02):
    """ Determines an optimal fit based on BIC with a penalty for 
        adding more components
    """
    best_fit = 1 # the number of components of the best fit, assume 1 at start
    best_bic = float('inf') # the BIC of the best fit, assume infinity
    for i in range(len(bics)): # loop number of components
        
        # if the % change of the current BIC from the best BIC
        # is greater than 
        # improvement_per_component * number of additional compnents
        # it is the new best fit
        if (1-bics[i]/best_bic) > improvement_per_component*(i+1-best_fit):
            best_bic = bics[i]
            best_fit = i+1
        else:
            continue
    return best_fit

# def best_fit(bics, weights, improvement_per_component=0.02, active_min_weight=0.10):
#     best_fit = 0
#     best_bic = float('inf')
#     active_components = [len(np.where(i>=active_min_weight)[0]) for i in weights]
#     for i in range(len(bics)):
#         if (1-bics[i]/best_bic) > improvement_per_component*(active_components[i]-active_components[best_fit]):
#             best_bic = bics[i]
#             best_fit = i
#         else:
#             continue
#     return best_fit

# # define GMM prediction
# def predict_gmm(data):

#     fits = [] # for saving fits with different n_components
#     gm = BayesianGaussianMixture(
#         n_components=5, 
#         max_iter=1500,
#         tol=1e-3,
#         weight_concentration_prior_type='dirichlet_process', 
#         weight_concentration_prior=0.01,
#         n_init=10,
#     )
#     gm.fit(data)

#     best_component = np.argmax(gm.weights_) # get main component based on weights
    
#     mean = gm.means_[best_component] # retrieve statistics
#     stdev = np.sqrt(gm.covariances_[best_component][0]) # for 1D outputs
#     cov = gm.covariances_[best_component] # for 2d outputs
    
#     result_dict = {
#         'mean': mean, 
#         'std': stdev, 
#         'covariance': cov,
#         'gmm_model': gm
#     }
    
#     return result_dict

