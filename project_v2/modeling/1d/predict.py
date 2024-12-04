import os
import numpy as np

def extract_features(data):
    '''
    Function for extracting features from 2d inputs
    
    Arguments:
        data:      1D numpy array (samples, features)
    Returns:
        features:  feature vector
    '''
    
    # check for standardized data
    assert np.abs(0-data.mean())<1e-2, 'Error: data not standardized'
    assert np.abs(1-data.std())<1e-2, 'Error: data not standardized'
    
    # extract features
    fgridrng = (-4, 4)
    fgridpts = 100
    features = np.histogram(data, np.linspace(fgridrng[0], fgridrng[1], fgridpts+1), density=True)[0]
    features = (features - features.min())/(features.max() - features.min())
    
    return features

def predict(data, model, scaler):
    '''
    Use 1D model to make a prediction
    
    Args:
        data:   1D numpy array
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
    p_mean = p[0]
    p_std = p[1]
    scaled_mean = (p_mean*std)+mean
    scaled_std = p_std*std
    
#     # scale the covariance matrix back to original scale
#     scaling_matrix = np.diag(std)
#     scaled_covariance_matrix = np.dot(np.dot(scaling_matrix, cov), scaling_matrix.T)
    
    result_dict = {
        'mean': scaled_mean,
        'covariance': np.array([[scaled_std**2]]),
        'std': scaled_std,
        'correlation': np.nan,
        'reference_fraction': p[2]
    }
    
    return result_dict


    