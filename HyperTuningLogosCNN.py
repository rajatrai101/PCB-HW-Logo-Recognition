# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
get_ipython().run_line_magic('reload_ext', 'tensorboard')
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
import datetime


# %%
# Importing the modules for collecting and building the dataset
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "true"
import re
import cv2
# Image processing...
from PIL import Image
from pathlib import Path


# %%
# Importing standard ML set - numpy, pandas, matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec

# Importing keras and its deep learning tools - neural network model, layers, contraints, optimizers, callbacks and utilities
from tensorflow.keras import Sequential
import keras
from keras.layers import Activation, Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.constraints import maxnorm
from keras.optimizers import Adam, RMSprop, SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import np_utils
from keras.regularizers import l2
from keras.initializers import RandomNormal, VarianceScaling

# Importing scikit-learn tools
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


# %%
# testPath = './../fics-logoaugmentator[Final]/Test/'
# trainPath = './../fics-logoaugmentator[Final]/Train/'
testPath = './../fics-logoaugmentator[Final]/AugTest/'
trainPath = './../fics-logoaugmentator[Final]/AugTrain/'
imgSize = 100
n_channels = 1


# %%
def scale(im):
    return cv2.cvtColor(cv2.resize(im, (imgSize,imgSize)),cv2.COLOR_BGR2GRAY)


# %%
# Setting up the image pool
def load_image_files(container_path):
    image_dir = Path(container_path)
    folders = [directory for directory in image_dir.iterdir() if directory.is_dir()]
    categories = [fo.name for fo in folders]
    print("Total Manufacturers:",len(categories))
    count = 0
    imgs = []
    y = []
    for i, direc in enumerate(folders):
        count = 0
        for file in direc.iterdir():
            count += 1
            imgs.append(scale(np.array(Image.open(file).convert("RGB"))).flatten())
            y.append(i)
    print(len(imgs),imgs[0].shape)
    X = np.array(imgs, order='F', dtype='uint8')
    y = np.asarray(y, dtype='uint8')
    return [X,y,categories]


# %%
X_test,y_test,_ = load_image_files(testPath)
X_train,y_train,logos = load_image_files(trainPath)


# %%
# Some additional data preparation needs to be done before blasting the images into the neural net
X_test,y_test = shuffle(X_test, y_test, random_state=42) # 42, what else? ;)
X_train,y_train = shuffle(X_train, y_train, random_state=42) # 42, what else? ;)


# %%
# We have to prepare the dataset to fit into the CNN
# X_train, X_test, y_train, y_test = train_test_split(train_data[0], train_data[1],test_size=0.2,stratify=train_data[1])

# we will maintain a copy of the test set, as we will do a couple of transformation to it
X_test_img = X_test.copy()
y_test_img = y_test.copy()
# let's bring back the images, like above
X_train = X_train.reshape(X_train.shape[0], imgSize, imgSize, n_channels)
X_test = X_test.reshape(X_test.shape[0], imgSize, imgSize, n_channels)

# Now for some convergence-friendly procedure (is supposed to converge a lot faster when brought to 0-1 float)...
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# ...and some additional pre-processing, to zero-center the data...
X_train = np.subtract(X_train, 0.5)
X_test = np.subtract(X_test, 0.5)

# ...and to scale it to (-1, 1)
X_train = np.multiply(X_train, 2.0)
X_test = np.multiply(X_test, 2.0)

# Labels have to be transformed to categorical
Y_train = np_utils.to_categorical(y_train, num_classes=len(logos))
Y_test = np_utils.to_categorical(y_test, num_classes=len(logos))


# %%
batch=8
n_classes = len(logos)
n_epochs =30


# %%
HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([32,64,128]))
HP_DROPOUT = hp.HParam('dropout', hp.Discrete([.1,0.3,0.5,0.7]))
HP_L_RATE = hp.HParam('l_rate', hp.Discrete([.001,.002,.005,.0001]))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))
HP_IS_DENSE = hp.HParam('is_deep', hp.Discrete(['0','1']))
# HP_IS_DEEP = hp.HParam('is_dense', hp.Discrete(['0','1']))
HP_K_REG = hp.HParam('k_reg', hp.Discrete([0.001,0.0001,0.0001]))
METRIC_ACCURACY = 'accuracy'
with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
  hp.hparams_config(
    hparams=[HP_NUM_UNITS, HP_DROPOUT, HP_OPTIMIZER,HP_K_REG, HP_IS_DENSE,HP_L_RATE],
    metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')])


# %%
def train_test_model(hparams):
    model = Sequential()
    model.add(Conv2D(32, (3,3),
                    input_shape=(imgSize,imgSize,n_channels),
                    padding='valid',
                    bias_initializer='glorot_uniform',
                    kernel_regularizer=l2(hparams[HP_K_REG]),
                    kernel_initializer=VarianceScaling(scale=2.0, mode='fan_in', distribution='normal', seed=None),
                    activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(64, (3,3),
                    padding='valid',
                    bias_initializer='glorot_uniform',
                    kernel_regularizer=l2(hparams[HP_K_REG]),
                    kernel_initializer=VarianceScaling(scale=2.0, mode='fan_in', distribution='normal', seed=None),
                    activation='relu'))

    model.add(Conv2D(128, (3,3),
                    padding='valid',
                    bias_initializer='glorot_uniform',
                    kernel_regularizer=l2(hparams[HP_K_REG]),
                    kernel_initializer=VarianceScaling(scale=2.0, mode='fan_in', distribution='normal', seed=None),
                    activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    
    if hparams[HP_IS_DENSE]=='1':
      model.add(Dense(hparams[HP_NUM_UNITS], activation='relu', bias_initializer='glorot_uniform'))
      model.add(Dropout(hparams[HP_DROPOUT]))

      # model.add(Dense(hparams[HP_NUM_UNITS], activation='relu', bias_initializer='glorot_uniform'))
      # model.add(Dropout(hparams[HP_DROPOUT]))
    
    model.add(Dense(len(logos), activation='softmax'))        
    
    early_stopping = EarlyStopping(patience=0, monitor='val_loss')
    # take_best_model = ModelCheckpoint(filepath=checkpoint_path,save_weights_only=True,verbose=1, save_best_only=True)

    log_dir = "logs/hparam_tuning/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    hparams_callback = hp.KerasCallback(log_dir, hparams)
    model.summary()
    if hparams[HP_OPTIMIZER]=='adam':
        opt=Adam(learning_rate=hparams[HP_L_RATE])
    else:
        opt=SGD(lr=hparams[HP_L_RATE])
    model.compile(
        optimizer=opt,
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    model.fit(X_train, Y_train, batch_size=batch, shuffle=True, epochs=n_epochs, verbose=1, validation_data=(X_test, Y_test),steps_per_epoch=X_train.shape[0]//8, workers=4,
        callbacks=[early_stopping, hparams_callback]) 
    
    _, accuracy = model.evaluate(X_test, Y_test)
    
    return accuracy


# %%
def run(run_dir, hparams):
  with tf.summary.create_file_writer(run_dir).as_default():
    hp.hparams(hparams)  # record the values used in this trial
    accuracy = train_test_model(hparams)
    tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)


# %%
session_num = 0
for l_rate in HP_L_RATE.domain.values:
  for optimizer in HP_OPTIMIZER.domain.values:
    for k_reg in HP_K_REG.domain.values:
      for is_dense in HP_IS_DENSE.domain.values:
        if is_dense=='1':
          for num_units in HP_NUM_UNITS.domain.values:
            for dropout_rate in HP_DROPOUT.domain.values:
              hparams = {
                  HP_NUM_UNITS: num_units,
                  # HP_IS_DEEP: is_deep,
                  HP_K_REG: k_reg,
                  HP_L_RATE: l_rate,
                  HP_DROPOUT: dropout_rate,
                  HP_OPTIMIZER: optimizer,
                  HP_IS_DENSE: is_dense
              }
              run_name = "run-%d" % session_num
              print('--- Starting trial: %s' % run_name)
              print({h.name: hparams[h] for h in hparams})
              run('logs/hparam_tuning/' + run_name, hparams)
              session_num += 1
        else:
          hparams = {
                HP_NUM_UNITS: 0,
                # HP_IS_DEEP: is_deep,
                HP_K_REG: k_reg,
                HP_L_RATE: l_rate,
                HP_DROPOUT: 0,
                HP_OPTIMIZER: optimizer,
                HP_IS_DENSE: is_dense
            }
          run_name = "run-%d" % session_num
          print('--- Starting trial: %s' % run_name)
          print({h.name: hparams[h] for h in hparams})
          run('logs/hparam_tuning/' + run_name, hparams)
          session_num += 1

# %%
