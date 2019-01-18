# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 19:06:04 2019

@author: hp
"""

from tensorflow import keras 
import  numpy as np
from PIL import Image 
model = model_mnist = keras.models.load_model("sample_model.h5")
img = Image.open("D:\\machinelearning projects\\licenseplatetextdet\\ocr image data\\char9\\img4.jpg")
img = img.resize((28,28))
img.convert('LA')
Xtest = np.array(img)[:,:,0]
Xtest = np.reshape(Xtest,(1,28,28,1))
output = model.predict(Xtest)
print(np.argmax(output))
img.show()