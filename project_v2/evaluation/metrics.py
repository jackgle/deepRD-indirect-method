# functions used for evaluating Gaussian or RI estimates

import numpy as np
from scipy.stats import norm
from scipy.stats import chi2
from matplotlib.patches import Ellipse


def bhattacharyya_gaussian_distance(mu1, cov1, mu2, cov2):
    """
    Compute the Bhattacharyya distance between two Gaussian distributions.

    Parameters:
    mu1 : array_like
        Mean of the first Gaussian distribution.
    cov1 : array_like
        Covariance matrix of the first Gaussian distribution.
    mu2 : array_like
        Mean of the second Gaussian distribution.
    cov2 : array_like
        Covariance matrix of the second Gaussian distribution.

    Returns:
    float
        Bhattacharyya distance between the two Gaussian distributions.
    """
    
    if cov1.shape==(2,2):
        assert mu1.shape==(2,), 'Error: single value given for mean'

    cov = (1 / 2) * (cov1 + cov2)

    T1 = (1 / 8) * (
        (mu1 - mu2) @ np.linalg.inv(cov) @ (mu1 - mu2).T
    )
    T2 = (1 / 2) * np.log(
        np.linalg.det(cov) / np.sqrt(np.linalg.det(cov1) * np.linalg.det(cov2))
    )

    return T1 + T2


def kl_gaussian_divergence(mu1, cov1, mu2, cov2):
    """
    Compute the Kullback-Leibler divergence between two Gaussian distributions.

    Parameters:
    mu1 : array_like
        Mean of the first Gaussian distribution.
    cov1 : array_like
        Covariance matrix of the first Gaussian distribution.
    mu2 : array_like
        Mean of the second Gaussian distribution.
    cov2 : array_like
        Covariance matrix of the second Gaussian distribution.

    Returns:
    float
        Kullback-Leibler divergence between the two Gaussian distributions.
    """

    k = len(mu1)
    cov2_inv = np.linalg.inv(cov2)
    
    T1 = np.trace(
        cov2_inv @ cov1
    )
    
    T2 = (mu2 - mu1) @ cov2_inv @ (mu2 - mu1).T
    
    T3 = np.log(
        np.linalg.det(cov2) / np.linalg.det(cov1)
    )

    return 1/2 * (T1 - k + T2 + T3)


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


def output_to_stats_1d(array):
    """
    Converts a 1D model output array to mean and covariance matrix
    """
    
    mean = array[0]
    std = array[1]
    
    return np.array([mean]), np.array([[std**2]])
    

def output_to_stats_2d(array):
    """
    Converts a 2D model output array to mean and covariance matrix
    """
    
    mean = array[:2]
    correlation = np.minimum(1, np.maximum(-1, array[4])) # limit predicted correlation to possible bounds
    cov = correlation_to_covariance(
        np.array([[1, correlation], [correlation, 1]]), # correlation matrix from output
        array[2:4] # add std devs
    )
    
    return mean, cov


def output_to_stats_3d(array):
    """
    Converts a 2D model output array to mean and covariance matrix
    """
    
    mean = array[:3]
    correlation = np.minimum(1, np.maximum(-1, array[6:])) # limit predicted correlation to possible bounds
    cov = correlation_to_covariance(
        np.array([
            [1, correlation[0], correlation[1]], 
            [correlation[0], 1, correlation[2]], 
            [correlation[1], correlation[2], 1], 
        
        ]), # correlation matrix from output
        array[3:6] # add std devs
    )
    return mean, cov


def get_errors(targets, p):
    """ Get BD errors from an array of targets and predicted parameters
    """
    errors = [
        bhattacharyya_gaussian_distance(
            i[0],i[1],j[0],j[1]) for i,j in zip(targets, p)
    ]
    return np.array(errors)
    

def norm_err(y_test, p_test):
    """
    Calculates the normalized error between a true and predicted reference interval as proposed in CA-125 paper submitted to Sci. Reports
    
    Arguments:
        y_test:   true RI
        p_test:   predicted RI
        
    Returns:
        normalized error
    """
    assert len(y_test)==2, 'Error: len(y_test) not equal to 2'
    assert len(p_test)==2, 'Error: len(p_test) not equal to 2'
    
    return np.mean([np.abs(i-j) for i,j in zip([-1, 1], (p_test-y_test.mean())/y_test.std())])


def stats_to_ri(mean, std_dev):
    """
    Converts mean and standard deviation to reference interval (central 95%)
    
    """
    return np.array([norm.ppf(0.025, loc=mean, scale=std_dev), norm.ppf(0.975, loc=mean, scale=std_dev)])


def plot_cov_ellipse(mean, cov, percentage=0.95, **kwargs):
    """
    Plots an ellipse representing the covariance matrix `cov` centered at `mean`.
    
    Parameters:
    - cov: 2x2 covariance matrix
    - mean: 2-element array-like representing the center of the ellipse
    - P: Percentage of density to enclose
    - kwargs: Additional keyword arguments to be passed to Ellipse patch.
    
    Returns:
    - The matplotlib Ellipse patch object.
    """
    # Get number of stds for ellipse axes based on percentage argument
    nstd = np.sqrt(chi2.ppf(percentage, 2))

    # Eigenvalues and eigenvectors of the covariance matrix
    eigvals, eigvecs = np.linalg.eigh(cov)
    
    # Compute the angle of the ellipse
    angle = np.degrees(np.arctan2(*eigvecs[:,0][::-1]))
    
    # Width and height of the ellipse
    width, height = 2 * nstd * np.sqrt(eigvals)

    # Create the ellipse patch
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, **kwargs)

    return ellipse

