# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 09:56:01 2019

@author: hp
"""
#%% loading modules
import  numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers
import os
from string import ascii_lowercase 
#%% loading training data
X_train = np.zeros((33*15,28,28))
temp_string = ascii_lowercase
temp_string=temp_string.replace('q','')
temp_string=temp_string.replace('i','')
temp_string=temp_string.replace('o','')
a = list(range(10))
for i in a :
   a[i] = chr(ord('0')+(a[i]))
a1=a+ list(temp_string)
classes = list(range(34))
char_dict = dict(zip(a1, classes))
count = 0
path = "D:\\machinelearning projects\\licenseplatetextdet\\ocr image data\\char"
for ch in char_dict.keys():
    files_list = os.listdir(path+ch)
    for file_name in files_list[:15]:
        img = (Image.open(path+ch+"\\"+file_name)).convert('LA')
        resized_img = img.resize((28,28))
        X_train[count,:,:] = np.array(resized_img)[:,:,0]
        count+=1
    print(path+ch+" "+str(len(files_list)))
#%% model
    
