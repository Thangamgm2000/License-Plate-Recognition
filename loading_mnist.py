# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 18:22:02 2019

@author: hp
"""

#loading mnist model  
import tensorflow.keras as keras
model = keras.models.load_model("mnist_model.h5")
#%%  
  model.summary()
  