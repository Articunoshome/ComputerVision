# -*- coding: utf-8 -*-
"""
Created on Tue May  4 12:30:59 2021

@author: Aryaan
"""

import numpy as np
import cv2
import sys
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt

f = 5806.559
b = 174.019
doffs=114.291

X = []
Y = []
Z = []

def nothing(x):
 pass
# ================================================
#
def getDisparityMap(imL, imR, numDisparities, blockSize):
    stereo = cv2.StereoBM_create(numDisparities=numDisparities, blockSize=blockSize)

    disparity = stereo.compute(imL, imR)
    disparity = disparity - disparity.min() + 1 # Add 1 so we don't get a zero depth, later
    disparity = disparity.astype(np.float32) / 16.0 # Map is fixed point int with 4 fractional bits

    return disparity # floating point image

def plot(disparity):
    # This just plots some sample points.  Change this function to
    # plot the 3D reconstruction from the disparity map and other values
    
    height, width = disparity.shape[:2]
    
    
    for h in range(height): 
        for w in range(width): 
            
            if (disparity[h,w] > 0):
               z = (b)*((f)/(disparity[h,w]+doffs))
               y = z*(h/f)
               x = (z*(w/f)) - (b/2)
               if z < 8000:
                  X.append(x)
                  Y.append(y)
                  Z.append(z)
    
    # Plt depths
    ax = plt.axes(projection ='3d')
    ax.scatter3D(X, Y, Z, c='g', marker = 'o')

    # Labels
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(0,270) #0,360 for side view and 270,90 for top view
    plt.savefig('topplot.png', bbox_inches='tight') # Can also specify an image, e.g. myplot.png
    plt.show()
    
    
       
if __name__ == '__main__':

    # Load left image
    filename = 'umbrellaL.png'
    imgL = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    imgL = cv2.Canny(imgL,48,145)
    #
    if imgL is None:
        print('\nError: failed to open {}.\n'.format(filename))
        sys.exit()


    # Load right image
    filename = 'umbrellaR.png'
    imgR = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    imgR = cv2.Canny(imgR,48,145)
    
    #
    if imgR is None:
        print('\nError: failed to open {}.\n'.format(filename))
        sys.exit()
    
    disparity = getDisparityMap(imgL, imgR, 64, 5 ) 
    
    plot(disparity)
    
    
            