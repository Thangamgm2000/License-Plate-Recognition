# -*- coding: utf-8 -*-
"""
Created on Tue May 28 19:32:32 2019

@author: hp
"""

from keras.layers import Input,Conv2D,MaxPooling2D,ZeroPadding2D,LeakyReLU,BatchNormalization
from keras.models import Model
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
#import tensorflow as tf
#from tensorflow.nn import sparse_softmax_cross_entropy_with_logits
#%%loading data
xtrain = np.load('xarr.npy')
xtrain /= 255.0
print(xtrain.shape)
ytrain = np.load('target.npy')
print(ytrain.shape)
#%% model parameters
grid_x = 14
grid_y = 6
img_width = 54
img_height = 190
training_samples = ytrain.shape[0]
no_of_true = np.sum(ytrain[:,:,:,0])
#%%
'''
def grid_cells():
    col_val = np.linspace(0,1,num= grid_x+1)
    row_val = np.linspace(0,1,num= grid_y+1)
    gridcellx = np.zeros(training_samples,grid_y,grid_x)
    for i in range(training_samples):
        for j in range(grid_y):
            gridcellx[i,j,:] = col_val
    gridcelly = np.zeros(training_samples,grid_y,grid_x)
    for i in range(training_samples):
        for j in range(grid_x):
            gridcelly[i,:,j] = row_val
    return (convert_to_tensor(gridcellx),convert_to_tensor(gridcelly))
gridcellx,gridcelly = grid_cells()'''
def softmax(x, axis=-1):
    y = np.exp(x - np.max(x, axis, keepdims=True))
    return y / np.sum(y, axis, keepdims=True)
def sigmoid(x):
    return 1. / (1. + np.exp(-x))
def plot_trainig(history):
    plt.plot(history.history['yolo_metrics'])
    plt.plot(history.history['cls_loss'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['yolo_loss'])
    #plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
def yolo_loss(y_true,y_pred):
    #shape_sor = K.shape(y_true)
    #shape = K.eval(shape_sor)
    #res+=(obj + lc(1-obj)) * (p[0]-a[0])**2
    pc_true = y_true[:,:,:,0]
    pc_pred = y_pred[:,:,:,0]
    bx_true = y_true[:,:,:,1]
    by_true = y_true[:,:,:,2]
    bx_pred = y_pred[:,:,:,1]
    by_pred = y_pred[:,:,:,2]
    w_true = y_true[:,:,:,3]
    w_pred = y_pred[:,:,:,3]
    h_true = y_true[:,:,:,4]
    h_pred = y_pred[:,:,:,4]
    classes_true = y_true[:,:,:,5:]
    classes_pred = y_pred[:,:,:,5:]
    softmaxed_pred = K.softmax(classes_pred)  
    score_pred = K.sigmoid(pc_pred) 
    adjx_pred = K.sigmoid(bx_pred) 
    adjy_pred = K.sigmoid(by_pred) 
    adjw_pred = K.sigmoid(w_pred)  
    adjh_pred = K.sigmoid(h_pred)  
    #no_of_true = K.sum(pc_true)
    #class_labels = K.argmax(classes_true,-1)
    losspc = K.sum(K.square(pc_true - score_pred))/training_samples
    lossxy = K.sum(pc_true*(K.square(bx_true-adjx_pred) + K.square(by_true-adjy_pred)))/no_of_true
    
    losswh = K.sum(pc_true*(K.square(w_true-adjw_pred) + K.square(h_true-adjh_pred)))/no_of_true
    
    losscls = 50*K.sum(pc_true*K.sum(K.square(classes_true-softmaxed_pred),axis=-1))/no_of_true
    
    loss = (losspc + lossxy +losswh +losscls)
    #loss = total_loss
    
    
    return loss
#%%
def yolo_metrics(y_true,y_pred):
    pc_true = y_true[:,:,:,0]
    classes_true = y_true[:,:,:,5:]
    softmaxed = K.softmax((y_pred[:,:,:,5:])*K.expand_dims(pc_true,axis=-1))
    classt = K.argmax(classes_true,axis=-1)
    classp = K.argmax(softmaxed,axis=-1)
    eq = (K.cast(K.equal(classt,classp),'int8')) * K.cast(pc_true,'int8')
    acc = K.sum(eq)
    return acc
#%%
def cls_loss(y_true,y_pred):
    classes_true = y_true[:,:,:,5:]
    classes_pred = y_pred[:,:,:,5:]
    softmaxed_pred = K.softmax(classes_pred)
    losscls = 50*K.sum(y_true[:,:,:,0]*K.sum(K.square(classes_true-softmaxed_pred),axis=-1))/no_of_true
    return losscls
#%%
def metrics(yt,yp):
    #yt = K.eval(y_true)
    #yp = K.eval(y_pred)
    pc_count =0
    cls_count =0
    pc_acc =0
    for i in range(training_samples):
        for j in range(grid_y):
            for k in range(grid_x):
                if(yt[i,j,k,0]==1):
                    pc_count += 1
                    if(sigmoid(yp[i,j,k,0])>=0.5):
                        pc_acc +=1
                    ct = yt[i,j,k,5:]
                    cp = yp[i,j,k,5:]
                    lp = np.argmax(softmax(cp))
                    lt = np.argmax(ct)
                    if(lp==lt):
                        cls_count+=1
    print(pc_count,pc_acc,cls_count)
    #print(characters.shape,no_of_char,correctly_classified)
   # print(yt[characters,5:].shape)
    #print(np.sum(np.abs(yt[characters,0] - yp[characters,0])))
    #print(np.sum(yp[characters,0]))
    return cls_count/pc_count
#%%
def numpy_loss(yt,yp):
    pc = yt[:,:,:,0]
    score_pred = sigmoid(yp[:,:,:,0])
    xt = yt[:,:,:,1]
    xp = sigmoid(yp[:,:,:,1])
    byt = yt[:,:,:,2]
    byp = sigmoid(yp[:,:,:,2])
    wt = yt[:,:,:,3]
    wp = sigmoid(yp[:,:,:,3])
    ht = yt[:,:,:,4]
    hp = sigmoid(yp[:,:,:,4])
    clst = yt[:,:,:,5:]
    clsp = softmax(yp[:,:,:,5:])
    pc_loss = np.sum(np.square(pc -score_pred))/training_samples
    xy_loss = (np.sum(pc*(np.square(xt-xp)+np.square(byt-byp))))/no_of_true
    wh_loss = (np.sum(pc*(np.square(ht-hp)+np.square(wt-wp))))/no_of_true
    class_loss = np.sum(pc[:,:,:,np.newaxis]*np.square(clst-clsp))/training_samples
    print(pc_loss,xy_loss,wh_loss,class_loss)
    acc = np.sum(pc*(np.argmax(clst,axis=-1)==np.argmax(clsp,axis=-1)))
    lbls = np.argmax(clsp,axis=-1)
    print((lbls[((pc==1) * (np.argmax(clst,axis=-1)==np.argmax(clsp,axis=-1)))]))
    return acc
#%%
                
inputs = Input(shape = (54,190,3))
x = Conv2D(32,3)(inputs)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = Conv2D(64,3)(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = Conv2D(128,3)(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = MaxPooling2D(pool_size=(1,2))(x)
x = Conv2D(256,3,padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = MaxPooling2D(pool_size=(2,2))(x)
x = Conv2D(128,3)(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = ZeroPadding2D(padding=(1,0))(x)
x = Conv2D(256,3)(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = MaxPooling2D(pool_size=(2,2))(x)
x = Conv2D(128,3,padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = MaxPooling2D(pool_size=(2,2),strides=1)(x)
x = ZeroPadding2D(padding=(1,0))(x)
x = Conv2D(256,3)(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = Conv2D(128,3)(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = Conv2D(64,3)(x)
x = BatchNormalization()(x)
x = Conv2D(41,1)(x)
model = Model(inputs=inputs,outputs=x)
print(model.summary())
print(K.int_shape(x))
#%%,odel compilation
model.compile('Adam',loss=yolo_loss,metrics=[yolo_metrics,cls_loss])

#%%
history = model.fit(x=xtrain,y=ytrain,batch_size=124,epochs=4,verbose=1)
#%% predict
ypred = model.predict(xtrain)
#%% accuracy/
print(metrics(ytrain,ypred))
#%%
plot_trainig(history)