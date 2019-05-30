# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 17:33:31 2018

@author: DELL
"""



#import lxml.etree as et
import xml.etree.ElementTree as ET
#import math
import cv2
import pandas as pd
import numpy as np
#from PIL import Image
def rotate_xml(filename, count):
    """ Parse a PASCAL VOC xml file """
    #Alpha=math.radians(Alpha)
    tree = ET.parse(filename)
    p=tree.find('path').text
    print(p)
    #objects = []
    for obj in tree.findall('object'):
        # if not check(obj.find('name').text):
        #     continue
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text) - 1,
                              int(bbox.find('ymin').text) - 1,
                              int(bbox.find('xmax').text) - 1,
                              int(bbox.find('ymax').text) - 1]
        
        # Read the original coordination
        x1=int(bbox.find('xmin').text)
        y1=int(bbox.find('ymin').text)
        x2=int(bbox.find('xmax').text)
        y2=int(bbox.find('ymax').text)
        img=cv2.imread(p)
        '''img.load()
        crop_img=img.crop((x1,y1,x2,y2))
        print(x1,y1,x2,y2)
        print(crop_img)
        crop_img=crop_img.resize((100,30))
        if(count%10==1):
            crop_img.show()
            img.show()
        crop_img.save("C:\\Users\\DELL\\Desktop\\Thangam\\croped_img\img_"+str(count)+".jpg")'''
        
        crop_img = img[y1:y2,x1:x2]
        resized_image = cv2.resize(crop_img, (40, 40))
        #cv2.imshow("resized",resized_image)
        #resized_image.show()
        #cv2.waitKey(0)
        #f=open("C:\\Users\\DELL\\Desktop\\Thangam\\Labelledimages\\image_0007.txt","a")
        k=resized_image.flatten()
        k=np.reshape(k,(1,np.size(k,0)))
        '''for j in k:
            f.write(str(j))
        f.write('\n')
        f.close()'''
        # Transformation
        df_img=pd.DataFrame(k)
        #print(df_img)
        df_img.to_csv("D:\\Education\\Others\\LPR project\\grayscale_back.csv",mode='a',header=False)
        #tree.write(filename)
for i in range(1,52):
    rotate_xml("D:\\Education\\Others\\LPR project\\labelled xml\\grayscale\\back\\img ("+str(i)+").xml", i)
#rotate_xml("C:\\Users\\DELL\\Desktop\\Thangam\\Labelledimages\\image_"+str(2)+".xml", 1)   

