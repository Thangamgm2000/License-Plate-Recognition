import numpy as np
import msvcrt as m
from matplotlib import pyplot as plt
def wait():
    m.getch()
#import pandas as pd
import PIL.Image as I
from sklearn.externals import joblib 
from skimage.morphology import square,opening,closing
from skimage import measure as ms
#from skimage.filters import threshold_otsu
from keras.models import load_model
clf2=load_model("model1.h5")
#clf3=joblib.load("textsvmply.pkl")
    #fn = 'img2.jpg'
def slidewin(path):
    img=I.open(path)
    opt=np.array(list(img.getdata()))
    opt=np.reshape(opt,(img.size[1],img.size[0],3))
    opt1=np.zeros((opt.shape[0],opt.shape[1]))
    #opt2=np.zeros(opt.shape)
    img = I.open(path);
    strides=[(2,5),(5,10),(10,20)]
    step_sizes={2:1,5:2,10:3,20:5}
    for strd in strides:
        
        x_stride,y_stride=strd
        x1 = 0
        y1 = 0
        x_step = step_sizes[x_stride]
        y_step = step_sizes[y_stride]
        x2 = x_stride
        y2 = y_stride
        count = 0
        cnt=0
        print(img.size)
        
        while(1):    
            x1 = 0
            x2 = x_stride
            if(y2+y_step>=img.size[0]):
                    y2 = img.size[0]
            while(1): 
                if(x2+x_step>=img.size[1]):
                    x2 = img.size[1]
                img2=img.crop((y1,x1,y2,x2))
                img2.load()
                #disp = img2.resize((img2.size[0]*10,img2.size[1]*10))
                #yyprint(x1,y1,x2,y2,cnt)
                #img2=img2.resize((10,20))
                dta = np.array(img2, dtype = 'uint8')
                #dta=np.reshape(dta,(20,10,3))
                #dta=dta.flatten()
                #dta=np.reshape(dta,(1,np.size(dta)))
                dta.reshape((48,16,3))
                Ym=clf2.predict(dta)
                if(Ym==1):
                    opt1[x1:x2+1,y1:y2+1]=opt1[x1:x2+1,y1:y2+1]+4
                    count=count+1
                    '''       Ym1=clf3.predict(dta)
                if(Ym1==1):
                    opt2[x1:x2+1,y1:y2+1,:]=opt2[x1:x2+1,y1:y2+1,:]+5
                    '''
                x2+=x_step
                x1+=x_step 
                cnt+=1
                if(x2>img.size[1]) :
                    break
                
        
            y2+=y_step
            y1+=y_step
            if(y2>img.size[0]) :
                break
    #opt1=opt1/s        
    print(count)
    img1=I.fromarray(opt1.astype('uint8'),'L')
    #img2=I.fromarray(opt2.astype('uint8'),'RGB')
    #img1.show()
    '''thresh= threshold_otsu(img1)
    img2=img1>=thresh
    img2=closing(img2,square(3))
    cntrs= ms.find_contours(img2,0.8)
    #print((cntrs))'''
    '''img2=I.fromarray(img2.astype('uint8'),'L')
    img2.show()
    img.show()'''
    img1.show()
    img.show()
    print(opt1)

    #input()
'''for i in range(1,12):    
 path = 'img_ ('+str(i)+').jpg'
 slidewin(path)'''
slidewin("D:\\Education\\Others\\LPR project\\coins.jpg")