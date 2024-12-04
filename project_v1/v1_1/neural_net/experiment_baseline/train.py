import os
os.sys.path.append('../lib/')
import pickle
import numpy as np
import pandas as pd
import sklearn.metrics
from sklearn.model_selection import train_test_split
import tensorflow as tf
import evaluate
from sklearn.preprocessing import RobustScaler


# training parameters
epochs = 100
batch_size = 8
loss = tf.keras.losses.MeanAbsoluteError()
opt = tf.keras.optimizers.Adam(learning_rate=0.0001)


# load data
x = np.load('./data/x.npy')
y = np.load('./data/y.npy')
means = np.load('./data/means.npy')
stds = np.load('./data/stds.npy')
files = np.load('./data/files.npy')
fileids = np.load('./data/fileids.npy')

# partition

# since samples are effectively duplicated across different sample sizes (N)
# we will use the dist parameters to find unique samples, and split the train/test based on those
# so there will be no samples that have the same columns below in train and test

# read RIbench metadata
test_set_meta = pd.read_csv('../../data/RIbench/BMTestSets_meta.csv', index_col=0)

# list columns for determining duplicates
cols = [
    'Analyte',
    'fractionPathol',
    'left_mu',
    'left_sigma',
    'left_ratio',
    'right_mu',
    'right_sigma',
    'right_ratio',
#     'N' # without N, unique elements decreases, but N minimally effects variation, effectively causing train/test overlap
]

# get duplicate and unique rows
mask = test_set_meta.duplicated(subset=cols, keep='first')
duplicate_rows = test_set_meta[mask]
unique_rows = test_set_meta[~mask]
print('unique rows:',len(unique_rows))

# split unique rows into train/val/test
train_unique, test_unique = train_test_split(unique_rows, test_size=0.2, random_state=42)
test_unique, val_unique = train_test_split(test_unique, test_size=0.5, random_state=42)

# merge unique partitions with their respective duplicates
# get file indices of train/val/test samples
idx_train = list(set(pd.merge(train_unique, test_set_meta, on=cols, how='inner')[['Index_x', 'Index_y']].values.flatten()))
idx_test = list(set(pd.merge(test_unique, test_set_meta, on=cols, how='inner')[['Index_x', 'Index_y']].values.flatten()))
idx_val = list(set(pd.merge(val_unique, test_set_meta, on=cols, how='inner')[['Index_x', 'Index_y']].values.flatten()))

# convert file indices to array indices
idx_train = np.where([i in idx_train for i in fileids])[0]
idx_test = np.where([i in idx_test for i in fileids])[0]
idx_val = np.where([i in idx_val for i in fileids])[0]

assert len(set(idx_train).intersection(idx_test))==0
assert len(set(idx_train).intersection(idx_val))==0
assert len(set(idx_test).intersection(idx_val))==0


x_train = x[idx_train]
y_train = y[idx_train]
files_train = files[idx_train]
means_train = means[idx_train]
stds_train = stds[idx_train]

x_val = x[idx_val]
y_val = y[idx_val]
files_val = files[idx_val]
means_val = means[idx_val]
stds_val = stds[idx_val]

x_test = x[idx_test]
y_test = y[idx_test]
files_test = files[idx_test]
means_test = means[idx_test]
stds_test = stds[idx_test]

print('x_train:',x_train.shape)
print('x_test:',x_test.shape)
print('x_val:',x_val.shape)

assert len(set([tuple(i) for i in y_train]).intersection([tuple(i) for i in y_test]))==0
assert len(set([tuple(i) for i in y_train]).intersection([tuple(i) for i in y_val]))==0
assert len(set([tuple(i) for i in y_val]).intersection([tuple(i) for i in y_test]))==0


# scale data
scalerx = RobustScaler()

x_train = scalerx.fit_transform(x_train)
x_test = scalerx.transform(x_test)
x_val = scalerx.transform(x_val)

# save training data
if not os.path.exists('./data/'):
    os.mkdir('./data/')
    
np.save('./data/x_train.npy', x_train)
np.save('./data/y_train.npy', y_train)
np.save('./data/means_train.npy', means_train)
np.save('./data/stds_train.npy', stds_train)
np.save('./data/files_train.npy', files_train)

np.save('./data/x_val.npy', x_val)
np.save('./data/y_val.npy', y_val)
np.save('./data/means_val.npy', means_val)
np.save('./data/stds_val.npy', stds_val)
np.save('./data/files_val.npy', files_val)

np.save('./data/x_test.npy', x_test)
np.save('./data/y_test.npy', y_test)
np.save('./data/means_test.npy', means_test)
np.save('./data/stds_test.npy', stds_test)
np.save('./data/files_test.npy', files_test)

del x, y, means, stds, files


# define model
inputs = tf.keras.layers.Input(shape=(x_train.shape[1],))
x = tf.keras.layers.Dense(512, activation='relu')(inputs)
x = tf.keras.layers.Dense(512, activation='relu')(x)
x = tf.keras.layers.Dense(512, activation='relu')(x)
# x = tf.keras.layers.Dropout(0.1)(x)
o = tf.keras.layers.Dense(2, activation='linear')(x)
model = tf.keras.models.Model(inputs, o)


# define loss and optimizer
model.compile(loss = loss,
              optimizer = opt)

model.summary()


# define best model checkpoint
checkpoint_filepath = './nn_output/model_checkpoint.h5'
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
    verbose = 2
)

# save model artifacts
with open('./nn_output/model_history.npy', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

model_json = model.to_json()
with open('./nn_output/model.json', 'w') as json_file:
    json_file.write(model_json)
        
with open('./nn_output/feature_scaler.pickle', 'wb') as file_pi:
    pickle.dump(scalerx, file_pi)

# predict test set
model.load_weights('./nn_output/model_checkpoint.h5') # load best weights
p = model.predict(x_test)
ps = (p*stds_test)+means_test
y_tests = np.hstack([(y_test[:,:2]*stds_test)+means_test, y_test[:,2:]])
np.save('./nn_output/model_test_p_scaled.npy', ps)
np.save('./nn_output/model_test_y_scaled.npy', y_tests)
pd.DataFrame({'files_test':files_test}).to_csv('./data/files_test.csv')
