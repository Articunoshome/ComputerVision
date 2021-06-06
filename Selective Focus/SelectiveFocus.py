# -*- coding: utf-8 -*-
"""
Created on Tue May  4 13:11:26 2021

@author: Aryaan
"""

import numpy as np
import cv2
import sys
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt

def nothing(x):
 pass

def getDisparityMap(imL, imR, numDisparities, blockSize):
    stereo = cv2.StereoBM_create(numDisparities=numDisparities, blockSize=blockSize)

    disparity = stereo.compute(imL, imR)
    disparity = disparity - disparity.min() + 1 # Add 1 so we don't get a zero depth, later
    disparity = disparity.astype(np.float32) / 16.0 # Map is fixed point int with 4 fractional bits

    return disparity 
def getDepth(disparity, K):
    height, width = disparity.shape[:2]
    depth = np.empty(disparity.shape)
    for y in range(height):
        for x in range(width):
            depth[y,x] = 1/(disparity[y,x]+K)
    return depth        
if __name__ == '__main__':

    # Load left image
    filename = 'girlL.png'
    imgL = cv2.imread(filename, 1)
    imgLG = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY)
    print(type(imgL[0][0][0]))
    print(imgL.shape)
    
    #
    if imgL is None:
        print('\nError: failed to open {}.\n'.format(filename))
        sys.exit()


    # Load right image
    filename = 'girlR.png'
    imgR = cv2.imread(filename, 1)
    imgRG = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY)
    
    
    #
    if imgR is None:
        print('\nError: failed to open {}.\n'.format(filename))
        sys.exit()

    #This commented snippet of code is for finding the optimum parameter values for creating the disparity map
    #If you wan to run it uncomment this code and comment the rest afterwards
    #---------------------------------------------------------------------------------
    '''
    # Create a window to display the image in
    cv2.namedWindow('Disparity', cv2.WINDOW_NORMAL)
    
    
    # create trackbars for color change
    cv2.createTrackbar('N','Disparity',0,26,nothing)
    cv2.createTrackbar('B','Disparity',5,255,nothing)
    
    # create switch for ON/OFF functionality
    #switch = '0 : OFF \n1 : ON'
    #cv2.createTrackbar(switch, 'Disparity',0,1,nothing)
    
    

    # Show result
    #cv2.imshow('Disparity', disparityImg)

    # Show 3D plot of the scene
    #plot(disparity)

    # Wait for spacebar press or escape before closing,
    # otherwise window will close without you seeing it
    while True:
        ND = cv2.getTrackbarPos('N','Disparity')
        ND = 16*ND
        
        BS = cv2.getTrackbarPos('B','Disparity')
        if BS%2==0:
           BS+=1
        

        disparity = getDisparityMap(imgLG, imgRG, ND, BS )
        

        # Normalise for display
        disparityImg = np.interp(disparity, (disparity.min(), disparity.max()), (0.0, 1.0))
        # Show result
        cv2.imshow('Disparity', disparityImg)
        key = cv2.waitKey(1)
        if key == ord(' ') or key == 27:
            break

    cv2.destroyAllWindows()
    '''
    #----------------------------------------------------------------------------------
    #----------------------------------------------------------------------------------
    #This commented snippet of code is for finding the optimum parameter values for creating the depth map
    #If you wan to run it uncomment this code and comment the rest afterwards and the commented code above
    '''
    cv2.namedWindow('Disparity',cv2.WINDOW_NORMAL)
    cv2.namedWindow('Depth', cv2.WINDOW_NORMAL)
    cv2.createTrackbar('K','Depth',1,100,nothing)
    disparity=getDisparityMap(imgLG, imgRG, 32, 49)
    disparityImg = np.interp(disparity, (disparity.min(), disparity.max()), (0.0, 1.0))
    
    
    while True:
        K = cv2.getTrackbarPos('K','Depth')
        
        depth = getDepth(disparityImg,K)
        
        
        
        
        depth*=255
        depth = depth.astype(np.uint8)
        
        
        cv2.imshow('Disparity', disparityImg)
        cv2.imshow('Depth',depth)
        key = cv2.waitKey(1)
        if key == ord(' ') or key == 27:
            break
    cv2.destroyAllWindows()            
    '''
    #----------------------------------------------------------------------------------
    
    disparity=getDisparityMap(imgLG, imgRG, 32,51)
    disparityImg = np.interp(disparity, (disparity.min(), disparity.max()), (0.0, 1.0))
    disparityImg = disparityImg.astype(np.float32)
    plt.imshow(disparityImg,'gray')
    plt.title('Disparity Map')
    plt.xticks([]),plt.yticks([])
    plt.savefig('DI')

    plt.show()
    depth = getDepth(disparityImg,1)
    
    height, width = depth.shape[:2]
    depth*=255
    depth = depth.astype(np.uint8)
    depth = cv2.cvtColor(depth,cv2.COLOR_GRAY2BGR)
    print(type(depth[0][0][0]))
    print(depth.shape)
    #depth = depth.astype(np.uint8)
    plt.imshow(depth,'gray')
    plt.title('Depth Map')
    plt.xticks([]),plt.yticks([])
    plt.savefig('DE')

    plt.show()
    '''
    cv2.imshow('Depth',depth)
    if cv2.waitKey(0) & 0xff == 27: 
       cv2.destroyAllWindows()
    
    cv2.imshow('ImgL', imgL)
    
    # De-allocate any associated memory usage  
    if cv2.waitKey(0) & 0xff == 27: 
       cv2.destroyAllWindows() 
    '''
    mask = cv2.threshold(depth, 249, 255, cv2.THRESH_BINARY)[1]
    plt.imshow(mask,'gray')
    plt.title('Mask')
    plt.xticks([]),plt.yticks([])
    plt.savefig('Mask')

    plt.show()
    '''
    cv2.imshow('mask',mask)
    if cv2.waitKey(0) & 0xff == 27: 
       cv2.destroyAllWindows()
    '''   
    blurredImage = cv2.GaussianBlur(imgL,(25,25),0) 
    plt.imshow(cv2.cvtColor(blurredImage, cv2.COLOR_BGR2RGB))
    plt.title('Blurred Image')
    plt.xticks([]),plt.yticks([])
    plt.savefig('BL')

    plt.show()
    (h, w, d) = depth.shape
    new_image = np.zeros((h, w, d),dtype=np.uint8)
    for i in range(len(mask)):
        for j in range(len(mask[i])):
                if np.array_equal(np.array([255,255,255]),mask[i][j])==True:
                    new_image[i,j,0] = blurredImage[i,j,0]
                    new_image[i,j,1] = blurredImage[i,j,1]
                    new_image[i,j,2] = blurredImage[i,j,2]
                else:
                    new_image[i,j,0] = imgL[i,j,0]
                    new_image[i,j,1] = imgL[i,j,1]
                    new_image[i,j,2] = imgL[i,j,2]
    plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
    plt.title('Selective Focussed Image')
    plt.xticks([]),plt.yticks([])
    plt.savefig('SFI')

    plt.show()
    '''
    cv2.imshow('result',new_image)
    if cv2.waitKey(0) & 0xff == 27: 
       cv2.destroyAllWindows()
    cv2.imwrite('Result.jpg', new_image)
    '''