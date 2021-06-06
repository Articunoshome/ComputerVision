# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 23:14:31 2021

@author: Aryaan
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

img=cv2.imread('tray.png',0)

# Code for creating the histogram
#---------------------------------
plt.hist(img.ravel(),256,[0,256])
plt.savefig('Tray Histogram') 
plt.show()
#---------------------------------


# Code for creating the two segmented images
#-----------------------------------------------
ret,thresh1 = cv2.threshold(img,130,255,cv2.THRESH_BINARY)


blur = cv2.GaussianBlur(img,(85,85),0)

titles = ['Original Tray','BINARY Tray','Subtract Tray']
images = [img, thresh1,cv2.subtract(img,blur)]

for i in range(3):
    plt.subplot(111)
    plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
    plt.savefig(titles[i])

plt.show()
#--------------------------------------------------