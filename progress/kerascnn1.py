import numpy as np
#import pandas as pd
import os
import glob
from PIL import Image
#import matplotlib.pyplot as plt
#import cv2
import tensorflow.keras as keras
from time import sleep
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPool2D,Dropout,Flatten,BatchNormalization,MaxPooling2D,Activation
from tensorflow.keras.optimizers import RMSprop,Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
if K.image_data_format() == 'channels_first':
    input_shape = (3, 48,16)
else:
    
    input_shape = (48, 16, 3) 
def createModel():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=1))
    model.add(Dropout(0.25))
 
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=1))
    model.add(Dropout(0.25))
 
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),strides=1))
    model.add(Dropout(0.25))
 
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.05))
    model.add(Dense(1, activation='sigmoid'))
     
    return model



model1 = createModel()
#%%
model1.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])

train_data = []
for i in range(303):
    p = np.load('D:\\Education\\Others\\LPR project\\finalnpys\\shadow\\npyy\\'+'img'+str(i)+'.jpg.npy')
    p.resize((48,16,3))
    train_data.append(p)
for i in range(77):
    p = np.load('D:\\Education\\Others\\LPR project\\finalnpys\\shadow\\npyn\\'+'img'+str(i)+'.jpg.npy')
    p.resize((48,16,3))
    train_data.append(p) 
train_data = np.array(train_data)

train_labels = [1 for i in range(303)] + [0 for i in range(77)]

model1.fit(train_data, train_labels, epochs=10, verbose=1)

model1.save('D:\\Education\\Others\\LPR project\\model1.h5')

img = Image.open("D:\\Education\\Others\\LPR project\\coins.jpg")
data = np.array(img, dtype = 'uint8')
size = data.shape
print(size)
for i in range(0,size[0]-48,2):
    for j in range(0,size[1]-16,2):
        res = model1.predict(data[i:i+48][j:j+16])
        if(res==1):
            pass