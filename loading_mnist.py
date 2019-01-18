# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 18:22:02 2019

@author: hp
"""

#loading mnist model  
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
model_mnist = keras.models.load_model("mnist_model.h5")
#%%  
model_mnist.summary()
  
#%%
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(28,28,1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(33, activation='softmax'))
model.summary()
#%%
layers_mnist = model_mnist.layers
layers = model.layers
count = 0
for lyrs in  layers_mnist[:-3]:
    layers[count].set_weights(lyrs.get_weights())
    count+=1
for lyrs in layers[:-3]:
    lyrs.trainable = False
print(model.summary())
#%%
import numpy as np
x_train = np.load("D:\\machinelearning projects\\licenseplatetextdet\\Numpy data\\X_train.npy")
x_train = x_train.reshape(x_train.shape[0],28,28, 1)
#%%
y_train= np.zeros(x_train.shape[0])
for i in range(33):
    y_train[i*15:(i+1)*15] = i
#%%
y_train1 = keras.utils.to_categorical(y_train,33)
#%%
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
perm = np.random.permutation(x_train.shape[0])
xshuf_train =  x_train[perm,:,:,:]
yshuf_train = y_train1[perm,:]
#%%
batch_size = 8
epochs = 20
xshuf_train/= 255
model.fit(xshuf_train,yshuf_train,batch_size= batch_size, epochs= epochs,verbose = 1,validation_data=(xshuf_train[:30,:,:,:],yshuf_train[:30,:]))
#%%
score = model.evaluate(xshuf_train, yshuf_train, verbose=0)
print('Train loss:', score[0])
print('Train accuracy:', score[1])