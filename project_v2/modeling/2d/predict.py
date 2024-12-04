import os
import numpy as np

def extract_features(data):
    '''
    Function for extracting features from 2d inputs
    
    Arguments:
        data:      2D numpy array (samples, features)
    Returns:
        features:  feature vector
    '''
    
    # ensure correct shape
    data = data.squeeze()
    assert data.shape[1]==2, 'Error: input data is not 2D'
    
    # check for standardized data
    assert all([np.abs(0-i)<1e-2 for i in data.mean(axis=0)]), 'Error: data not standardized'
    assert all([np.abs(1-i)<1e-2 for i in data.std(axis=0)]), 'Error: data not standardized'
    
    # extract features
    fgridrng = (-4, 4)
    fgridpts = 100
    features = np.histogram2d(data[:,0], data[:,1], np.linspace(fgridrng[0], fgridrng[1], fgridpts+1), density=True)[0]
    features = (features - features.min())/(features.max() - features.min())
    
    return features

def predict(data, model, scaler):
    '''
    Use 2D model to make a prediction
    
    Args:
        data:   2D numpy array
        model:  trained tensorflow-keras model
        scaler: model output scaler
        
    Returns:
        result_dict: dictionary of predicted statistics
    
    '''
    
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    data = (data-mean)/std
    
    # extract features and statistics
    features = extract_features(data)
    
    # predict
    p = model.predict(features[np.newaxis,...,np.newaxis], verbose=0)
    p = scaler.inverse_transform(p)[0]
    
    # scale the predicted statistics back to original
    p_mean = p[:2]
    p_std = p[2:4]
    p_cor = p[4]
    scaled_mean = (p_mean*std)+mean
    scaled_std = p_std*std
    
    # convert predicted stds and correlation to covariance matrix
    cov = correlation_to_covariance(np.array([[1, p_cor], [p_cor, 1]]), scaled_std)
    
#     # scale the covariance matrix back to original scale
#     scaling_matrix = np.diag(std)
#     scaled_covariance_matrix = np.dot(np.dot(scaling_matrix, cov), scaling_matrix.T)
    
    result_dict = {
        'mean': scaled_mean,
        'covariance': cov,
        'std': scaled_std,
        'correlation': p_cor,
        'reference_fraction': p[5]
    }
    
    return result_dict

def correlation_to_covariance(correlation_matrix, std_devs):
    '''
    Converts a correlation matrix to covariance matrix
    
    Args:
        correlation_matrix: correlation matrix
        std_devs: vector of variable standard deviations
    Returns:
        covariance_matrix
    '''
    
    # calculate covariance matrix using the formula: cov(X, Y) = corr(X, Y) * std_dev(X) * std_dev(Y)
    covariance_matrix = np.outer(std_devs, std_devs) * correlation_matrix
        
    return covariance_matrix
    