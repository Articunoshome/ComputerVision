# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 17:12:44 2021

@author: Aryaan
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

img=cv2.imread('brain.png',0)

# Code for creating the histogram
#---------------------------------
plt.hist(img.ravel(),256,[0,256]); 
plt.savefig('Brain Histogram')
plt.show()
#---------------------------------


# Code for creating the two segmented images
#-----------------------------------------------
ret,thresh1 = cv2.threshold(img,196,210,cv2.THRESH_BINARY)
ret,thresh2 = cv2.threshold(img,0,255,cv2.THRESH_OTSU)

and_img_binary = cv2.bitwise_and(img, thresh1, mask=None)

and_img_otsu = cv2.bitwise_and(img, thresh2, mask=None)

titles = ['Original Brain','BINARY Brain','Otsu Brain','AND_BINARY Brain', 'AND_Otsu Brain']
images = [img, thresh1, thresh2, and_img_binary, and_img_otsu ]

for i in range(5):
    plt.subplot(111)
    plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
    plt.savefig(titles[i])

plt.show()
#--------------------------------------------------

