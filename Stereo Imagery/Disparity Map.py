import numpy as np
import cv2
import sys
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt




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

if __name__ == '__main__':

    #Enter choice for whether you want to create disparity map for original image
    #or edge connected image
    #Choices are o for original image and e for edge connected image
    
    #c = input('Enter you choice')
    
    # Load left image based on choice provided
    #-----------------------------
    filenameL = 'umbrellaL.png'
    filenameR = 'umbrellaR.png'
    '''
    if c == 'o':
       imgL = cv2.imread(filenameL, cv2.IMREAD_GRAYSCALE)
    elif c== 'e': 
       imgL = cv2.imread(filenameL, cv2.IMREAD_GRAYSCALE) 
       imgL = cv2.Canny(imgL,48,145)
    
    if imgL is None:
        print('\nError: failed to open {}.\n'.format(filenameL))
        sys.exit()
    #-----------------------------

    # Load right image
    #-----------------------------
    filenameR = 'umbrellaR.png'
    if c == 'o':
       imgR = cv2.imread(filenameR, cv2.IMREAD_GRAYSCALE)
    elif c== 'e': 
       imgR = cv2.imread(filenameR, cv2.IMREAD_GRAYSCALE) 
       imgL = cv2.Canny(imgL,48,145)
    if imgR is None:
        print('\nError: failed to open {}.\n'.format(filenameR))
        sys.exit()
    #-----------------------------    
    '''
    '''
    # Create a window to display the image in
    cv2.namedWindow('Disparity', cv2.WINDOW_NORMAL)
    
    
    # create trackbars for color change
    cv2.createTrackbar('N','Disparity',0,26,nothing)
    cv2.createTrackbar('B','Disparity',5,255,nothing)
    

    

    # Wait for spacebar press or escape before closing,
    # otherwise window will close without you seeing it
    while True:
        ND = cv2.getTrackbarPos('N','Disparity')
        ND = 16*ND
        
        BS = cv2.getTrackbarPos('B','Disparity')
        if BS%2==0:
           BS+=1
        

        disparity = getDisparityMap(imgL, imgR, ND, BS )
        

        # Normalise for display
        disparityImg = np.interp(disparity, (disparity.min(), disparity.max()), (0.0, 1.0))
        # Show result
        cv2.imshow('Disparity', disparityImg)
        key = cv2.waitKey(1)
        if key == ord(' ') or key == 27:
            break

    cv2.destroyAllWindows()
    '''
    imgL = cv2.imread(filenameL, cv2.IMREAD_GRAYSCALE)
    imgL = cv2.Canny(imgL,48,145)
    imgR = cv2.imread(filenameR, cv2.IMREAD_GRAYSCALE)
    imgR = cv2.Canny(imgR,48,145)
    disparity = getDisparityMap(imgL, imgR, 64, 5 )
        

    # Normalise for display
    disparityImg = np.interp(disparity, (disparity.min(), disparity.max()), (0.0, 1.0))
    plt.imshow(disparityImg,'gray')
    plt.title('Edge-Connected Disparity Image')
    plt.xticks([]),plt.yticks([])
    plt.savefig('EDI')

    plt.show()