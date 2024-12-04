import os
import shutil
import numpy as np
import pandas as pd

path = os.path.abspath('../../data/RIbench/')
files = sorted(np.hstack([[path+'/Data/'+i+'/'+j for j in os.listdir(path+'/Data/'+i)] for i in os.listdir(path+'/Data/')]))

test_set_meta = pd.read_csv(path+'/BMTestSets_meta.csv', index_col=0)

def get_y(file):
    # use the BMTestSets_meta.csv file to get ground truth values
    # associated with an RIbench sample via file ID
    
    fileid = int(file.split('/')[-1].split('_')[0])
    target = (
        test_set_meta[test_set_meta.Index==fileid].GT_LRL.values[0],
        test_set_meta[test_set_meta.Index==fileid].GT_URL.values[0],
        test_set_meta[test_set_meta.Index==fileid].nonp_mu.values[0],
        test_set_meta[test_set_meta.Index==fileid].nonp_sigma.values[0],
        test_set_meta[test_set_meta.Index==fileid].nonp_lambda.values[0],
        test_set_meta[test_set_meta.Index==fileid].nonp_shift.values[0]        
    )
    return np.array(target), fileid
    
# load data
print('Loading RIbench files')
x = []
y = []
means = []
stds = []
fileids = []
for c,i in enumerate(files):
    
    if c%500==0:
        print('\t', c, ' files')

    data = pd.read_csv(i).values
    gt, fileid = get_y(i)
    fileids.append(fileid)

    if i.split('/')[-2]=='CRP':
        gt[0] = 0

    # normalization
    means.append(data.mean())
    stds.append(data.std())

    gtri = (gt[:2]-data.mean())/data.std()
    data = (data-data.mean())/data.std()
    
    # outlier removal
    data = data[(data >= np.quantile(data, 0.01)) & (data <= np.quantile(data, 0.99))]

    # quantile array
    data = np.quantile(data, np.linspace(0, 1, 500))
    x.append(data)
    y.append(np.hstack([gtri, gt[2:]]).T)
    
x = np.vstack(x)
y = np.vstack(y)
means = np.vstack(means)
stds = np.vstack(stds)

# save
print('Saving')
if not os.path.exists('./data/'):
    os.mkdir('./data/')
else:
    shutil.rmtree('./data/')
    os.mkdir('./data/')
np.save('./data/x.npy', x)
np.save('./data/y.npy', y)
np.save('./data/means.npy', means)
np.save('./data/stds.npy', stds)
np.save('./data/files.npy', files)
np.save('./data/fileids.npy', fileids)

