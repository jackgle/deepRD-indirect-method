import os
import shutil
import pickle
import numpy as np
import pandas as pd
import sklearn.metrics
from sklearn.model_selection import train_test_split
import tensorflow as tf
os.sys.path.append('../lib/')
import evaluate
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler


# training parameters
epochs = 100
batch_size = 8

datapath = '../experiment_baseline/data/'

# load data
x_all = np.load(datapath+'x.npy')
y_all = np.load(datapath+'y.npy')
means_all = np.load(datapath+'means.npy')
stds_all = np.load(datapath+'stds.npy')
files_all = np.load(datapath+'files.npy')

    
def get_model():
    
    loss = tf.keras.losses.MeanAbsoluteError()
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    
    # define model
    inputs = tf.keras.layers.Input(shape=(x_all.shape[1],))
    x = tf.keras.layers.Dense(512, activation='relu')(inputs)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    o = tf.keras.layers.Dense(2, activation='linear')(x)
    m = tf.keras.models.Model(inputs, o)

    # define loss and optimizer
    m.compile(loss = loss,
              optimizer = opt)
    return m
    
def train_model(holdout_analyte, x, y, files, means, stds, models_path):
    
    print('\nTraining holdout model - ', holdout_analyte, '\n') 
    
    model_path = models_path+'/model_'+holdout_analyte
    
    if os.path.exists(model_path):
        shutil.rmtree(model_path)
        os.makedirs(model_path)
    else:
        os.makedirs(model_path)
    
    idx = np.array([holdout_analyte in i.split('/')[-2] for i in files]) # use a single analyte for testing
    
    # train
    x_train = x[~idx]
    y_train = y[~idx]
    files_train = files[~idx]
    means_train = means[~idx]
    stds_train = stds[~idx]

    # test
    x_test = x[idx]
    y_test = y[idx]
    files_test = files[idx]
    means_test = means[idx]
    stds_test = stds[idx]

    # create validation    
    idx_test, idx_val = train_test_split(range(x_test.shape[0]), train_size=0.5, random_state=42)
    
    x_val = x_test[idx_val]
    y_val = y_test[idx_val]
    files_val = files_test[idx_val]
    means_val = means_test[idx_val]
    stds_val = stds_test[idx_val]

    x_test = x_test[idx_test]
    y_test = y_test[idx_test]
    files_test = files_test[idx_test]
    means_test = means_test[idx_test]
    stds_test = stds_test[idx_test]

    if holdout_analyte=='Hb':
        print('Train shape: ', x_train.shape)
        print('Val shape: ', x_val.shape)
        print('Test shape: ', x_test.shape)

    # scale data
#     scalerx = RobustScaler()

#     x_train = scalerx.fit_transform(x_train)
#     x_test = scalerx.transform(x_test)
#     x_val = scalerx.transform(x_val)

    # save training data
    os.mkdir(model_path+'/data/')

    np.save(model_path+'/data/x_train.npy', x_train)
    np.save(model_path+'/data/y_train.npy', y_train)
    np.save(model_path+'/data/means_train.npy', means_train)
    np.save(model_path+'/data/stds_train.npy', stds_train)
    np.save(model_path+'/data/files_train.npy', files_train)

    np.save(model_path+'/data/x_val.npy', x_val)
    np.save(model_path+'/data/y_val.npy', y_val)
    np.save(model_path+'/data/means_val.npy', means_val)
    np.save(model_path+'/data/stds_val.npy', stds_val)
    np.save(model_path+'/data/files_val.npy', files_val)

    np.save(model_path+'/data/x_test.npy', x_test)
    np.save(model_path+'/data/y_test.npy', y_test)
    np.save(model_path+'/data/means_test.npy', means_test)
    np.save(model_path+'/data/stds_test.npy', stds_test)
    np.save(model_path+'/data/files_test.npy', files_test)


    model = get_model()


    # define best model checkpoint
    checkpoint_filepath = model_path+'/model_checkpoint.h5'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        verbose=1
    )


    # train
    history = model.fit(
        x_train, 
        y_train[:,:2],
        validation_data = (x_val, y_val[:,:2]),
        epochs = epochs,
        batch_size = batch_size,
        callbacks = [model_checkpoint_callback],
        verbose = 0
    )
    with open(model_path+'/model_history.npy', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    # predict test set
    model.load_weights(model_path+'/model_checkpoint.h5') # load best weights
    p = model.predict(x_test)
    ps = (p*stds_test)+means_test
    y_tests = np.hstack([(y_test[:,:2]*stds_test)+means_test, y_test[:,2:]])
    np.save(model_path+'/model_test_p_scaled.npy', ps)
    np.save(model_path+'/model_test_y_scaled.npy', y_tests)
    pd.DataFrame({'files_test':files_test}).to_csv(model_path+'/data/files_test.csv')
    
    
# partition
holdout_analytes = ['Hb', 'Ca', 'FT4', 'AST', 'LACT', 'GGT', 'TSH', 'IgE', 'CRP', 'LDH']

# create model
for holdout_analyte in holdout_analytes:
    
    train_model(holdout_analyte, x_all, y_all, files_all, means_all, stds_all, './models/')

    
