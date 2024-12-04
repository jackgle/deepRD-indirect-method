import scipy.stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import scipy.stats

colormap = matplotlib.colormaps.get_cmap('plasma')


def boxcox(x, lmbda):
    # box-cox transformation
    if lmbda!=0:
        x = (x**lmbda-1)/lmbda
    else:
        x = np.log(x)
    return x


def zscore_transform(ri, mu, sigma, lmbda, shft):
    # ribench z-score transform for z-score deviation
    return (boxcox(ri-shft, lmbda)-mu)/sigma


def zscore_deviation(y, x):
    # ribench z-score deviation
    xt = zscore_transform(x,     y[2], y[3], y[4], y[5])
    yt = zscore_transform(y[:2], y[2], y[3], y[4], y[5])
    return np.mean(np.abs(xt-yt))
    
    
def normp(y, p):
    # proposed standardization for norm. error
    if len(y)>2: # files prepared in create_training_data.py have other ground truth statistics after RI limits in first two indices
        y = y[:2]
    return (y-y.mean())/y.std(), (p-y.mean())/y.std()


def get_path_fraction(file, test_set_meta):
    # get pathological fraction from metadata
    fileid = int(file.split('/')[-1].split('_')[0])
    target = test_set_meta[test_set_meta.Index==fileid].fractionPathol.values[0]
    return target


def get_sample_size(file, test_set_meta):
    # get sample size from metadata
    fileid = int(file.split('/')[-1].split('_')[0])
    target = test_set_meta[test_set_meta.Index==fileid].N.values[0]
    return target


def compute_scores(groups, p_test, y_test, files_test):
    
    ''' Computes z-score dev and norm. error for each group
    
    Args:
        groups: A list of either 
            1) analytes
            2) list of analytes
    Returns:
        zscore_devs: average z-score deviations for each group
        norm_errors: average norm. error for each group
    '''
    
    analytes = np.array([i.split('_')[-3] for i in files_test]) # get analyte for each test file

    zscore_devs = []
    norm_errors = []
    for i in groups: # loop analytes (groups)
        idx = np.where([j in i for j in analytes])[0] # get indices of test samples for current analyte
        tmpzs = [] # temp array for z-score dev
        tmpne = [] # temp array of norm. error
        for j in idx:
            if 'CRP' in files_test[j]: # CRP is the only analyte with only an upper limit
                # z-score dev
                xt = zscore_transform(p_test[j][1], y_test[j][2], y_test[j][3], y_test[j][4], y_test[j][5])
                yt = zscore_transform(y_test[j][1], y_test[j][2], y_test[j][3], y_test[j][4], y_test[j][5])
                tmpzs.append(np.abs(xt-yt))

                # norm. error
                normy, nrmp = normp(y_test[j], p_test[j])
                tmpne.append(np.abs(normy[1] - nrmp[1]))

            else:
                # z-score dev
                if p_test[j,0]-y_test[j,5]<=0: # z-score deviation won't work with negative lower limit
                    tmpzs.append(np.nan)
                else:
                    tmpzs.append(zscore_deviation(y_test[j], p_test[j]))

                # norm. error
                normy, nrmp = normp(y_test[j], p_test[j])
                tmpne.append(np.mean(np.abs(normy - nrmp)))

        zscore_devs.append(np.nanmean(tmpzs))
        norm_errors.append(np.nanmean(tmpne))

        if (len(np.where(np.isnan(tmpzs))[0])>0):
            print(i, ' - ', round(len(np.where(np.isnan(tmpzs))[0])/len(tmpzs), 3), " z-score dev. nan's")
        if (len(np.where(np.isnan(tmpne))[0])>0):
            print(i, ' - ', round(len(np.where(np.isnan(tmpne))[0])/len(tmpne), 3), " norm. error nan's")
    
    return zscore_devs, norm_errors


def full_evaluation(p_test, y_test, files_test):
    
    ''' Generates statistics and plots showing the model performance
    
    Args:
        p_test: predicted RIs for test samples
        y_test: ground truth RIs for test samples
        files_test: file name for each test sample
        
    '''
    
    # plot the number of samples per analyte in the test set
    print('\nNumber of samples per analyte in test set')
    ans_test = np.array([i.split('/')[-2] for i in files_test])
#     pd.DataFrame({'an':ans_test})['an'].value_counts().plot.bar(rot=0).grid(axis='y')
#     plt.gcf().set_facecolor('white')
#     plt.gcf().show()
    print(pd.DataFrame({'an':ans_test})['an'].value_counts())
    
    
    # plot example predictions for various analytes
    print('\nExample predictions')
    nr = 5
    plt.figure(figsize=(15,15))
    sample = np.random.choice(range(y_test.shape[0]), nr**2)

    # optionally set the specific analyte to visualize examples of
    sample = np.where(ans_test=='IgE')[0]

    for c,i in enumerate(sample[:nr**2]):
        plt.subplot(nr, nr, c+1)
        data = pd.read_csv(files_test[i], header=None).values
        data = data[(data>=np.quantile(data,0.001)) & (data<=np.quantile(data,0.999))]
        y = y_test[i]
        p = p_test[i]  

        normy, nrmp = normp(y, p)

        plt.hist(data, 50, density=True);
        if not ans_test[i]=='CRP':
            plt.axvline(y[0], c='r')
            plt.axvline(p[0], c='g')
        plt.axvline(y[1], c='r')
        plt.axvline(p[1], c='g')
        plt.gca().set_yticks([])
        plt.title(round(np.mean(np.abs(normy-nrmp)), 3))

    plt.gcf().patch.set_facecolor('white')
    plt.gcf().tight_layout()
    plt.show()

    print('\nError per analyte')
    groups = ['Hb', 'Ca', 'FT4', 'AST', 'LACT', 'GGT', 'TSH', 'IgE', 'CRP', 'LDH']
    zscore_devs, norm_errors = compute_scores(groups, p_test, y_test, files_test)
    df = pd.DataFrame({'analyte': groups,
                       'zscore_dev': zscore_devs,
                       'norm_errors': norm_errors})
    print(df)
    print('Average z-score deviation:\t', df.zscore_dev.mean())
    print('Average norm. error:\t\t', df.norm_errors.mean())
    
    print('\nError per skew group')
    groups = [['Hb', 'Ca', 'FT4'], ['AST', 'LACT', 'GGT'], ['TSH', 'IgE'], ['LDH']] # NOTE - CRP  TAKEN OUT!
    zscore_devs, norm_errors = compute_scores(groups, p_test, y_test, files_test)
    df = pd.DataFrame({'group': ['normal', 'skewed', 'heavily_skewed', 'skewed_and_shifted'],
                       'zscore_dev': zscore_devs,
                       'norm_errors': norm_errors})
    print(df)
    
    print('\nError vs. pathological fraction')
    # compute all errors
    errors = []
    for i in range(p_test.shape[0]):
        normy, nrmp = normp(y_test[i][:2], p_test[i])
        errors.append(np.mean(np.abs(normy - nrmp)))
    errors = np.array(errors)
    
    path = '../../data/RIbench/'
    test_set_meta = pd.read_csv(path+'BMTestSets_meta.csv', index_col=0)

    # read all pathological fractions
    path_frac_test = []
    for i in files_test:
        path_frac_test.append(get_path_fraction(i, test_set_meta))
    path_frac_test = np.array(path_frac_test)
    print('Pearson correlation between error and path. frac.:')
    # pearson correlation between error and path frac
    print(scipy.stats.pearsonr(path_frac_test, errors))

    colors = colormap(np.linspace(0, 0.75, len(set(path_frac_test)))) # generate colors for each path. frac. group
    plt.figure(dpi=100) # increase dpi for higher resolution
    for c,i in enumerate(sorted(list(set(path_frac_test)))):
        sample = errors[path_frac_test==i] # get errors for samples with current pathological fraction
        kde = scipy.stats.gaussian_kde(sample) # fit kernel density estimator for current error (instead of just histogram)
        x = np.linspace(0, 2, 1000) # get density values across a desired range
        density = kde(x)
        plt.plot(x, density, color=colors[c], alpha=1, label=i) # add to plot
    plt.legend(title='Pathological Fraction');
    plt.gcf().set_facecolor('white')
    plt.grid();
    plt.xlabel('Normalized Error')
    plt.ylabel('Density')
    plt.xlim([0, 1]);
    plt.title('Baseline')
    plt.gca().set_yticks([])
    plt.show()
    
    print('\nError vs. sample size')
    # read all sample sizes
    sample_size_test = []
    for i in files_test:
        sample_size_test.append(get_sample_size(i, test_set_meta))
    sample_size_test = np.array(sample_size_test)
    
    print('Pearson correlation between error and sample size:')
    # pearson correlation between error and sample size
    print(scipy.stats.pearsonr(sample_size_test, errors))

    colors = colormap(np.linspace(0, 0.75, len(set(sample_size_test))))
    plt.figure(dpi=100)
    for c,i in enumerate(sorted(list(set(sample_size_test)), reverse=True)):
        sample = errors[sample_size_test==i]
        kde = scipy.stats.gaussian_kde(sample)
        x = np.linspace(0, 2, 1000)
        density = kde(x)
        plt.plot(x, density, color=colors[c], alpha=1, label=int(i))
    plt.legend(title='Sample Size');
    plt.gcf().set_facecolor('white')
    plt.grid();
    plt.xlabel('Normalized Error')
    plt.ylabel('Density')
    plt.gca().set_yticks([])
    plt.xlim([0, 1]);

