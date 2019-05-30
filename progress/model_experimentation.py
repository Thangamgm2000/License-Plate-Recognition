# -*- coding: utf-8 -*-
"""
Created on Tue May 28 19:32:32 2019

@author: hp
"""

from keras.layers import Input,Conv2D,MaxPooling2D,ZeroPadding2D
from keras.models import Model 
inputs = Input(shape = (54,190,3))
x = Conv2D(32,3)(inputs)
x = Conv2D(64,3)(x)
x = Conv2D(128,3)(x)
x = MaxPooling2D(pool_size=(1,2))(x)
x = Conv2D(256,3,padding='same')(x)
x = MaxPooling2D(pool_size=(2,2))(x)
x = Conv2D(128,3)(x)
x = ZeroPadding2D(padding=(1,0))(x)
x = Conv2D(256,3)(x)
x = MaxPooling2D(pool_size=(2,2))(x)
x = Conv2D(128,3,padding='same')(x)
x = MaxPooling2D(pool_size=(2,2),strides=1)(x)
x = ZeroPadding2D(padding=(1,0))(x)
x = Conv2D(256,3)(x)
x = Conv2D(128,3)(x)
x = Conv2D(64,3)(x)
x = Conv2D(41,1)(x)
model = Model(inputs=inputs,outputs=x)
print(model.summary())