# -*- coding: utf-8 -*-
"""
Created on Sat May  1 11:15:53 2021

@author: Aryaan
"""
import numpy as np
import cv2 
import sys
def nothing(x):
 pass
filename = 'umbrellaR.png'
img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

cv2.namedWindow('Edge Detected Image', cv2.WINDOW_NORMAL)
    
    
# create trackbars for color change
cv2.createTrackbar('min','Edge Detected Image',0,300,nothing)
cv2.createTrackbar('max','Edge Detected Image',0,400,nothing)

while True:
        minimum = cv2.getTrackbarPos('min','Edge Detected Image')
        
        
        maximum = cv2.getTrackbarPos('max','Edge Detected Image')
        
        edges = cv2.Canny(img,minimum,maximum)

        
        

        
        # Show result
        cv2.imshow('Edge Detected Image', edges)
        key = cv2.waitKey(1)
        if key == ord(' ') or key == 27:
            break

cv2.destroyAllWindows()