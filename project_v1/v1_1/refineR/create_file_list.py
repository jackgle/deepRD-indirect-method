import os
import pandas as pd

path = '../data/RIbench/Data/'

files = []
for i in sorted(os.listdir(path)):
    for file in sorted(os.listdir(path+i)):
        files.append(path+i+'/'+file)
pd.DataFrame({'files':files}).to_csv('./files_list.csv')
