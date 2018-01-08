
"""
this is an implementation of image representation in different color space w.r.t. rgb space

@author: atpandey
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from functions import plot3d,convert_color_bgr



if __name__ == '__main__':
    #Look at different color spaces
    img = cv2.imread('../cutouts/cutouts/red_car_for_color_hist.png')
    # Select a small fraction of pixels to plot by subsampling it
    scale = max(img.shape[0], img.shape[1], 64) / 64  # at most 64 rows and columns
    img_small = cv2.resize(img, (np.int(img.shape[1] / scale), np.int(img.shape[0] / scale)), interpolation=cv2.INTER_NEAREST)
    #img_small=bin_spatial(img,size=(np.int(img.shape[1] / scale), np.int(img.shape[0] / scale)))
    #print(img_small.shape)
                          
    # Convert subsampled image to desired color space(s)
    img_small_RGB = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)  # OpenCV uses BGR, matplotlib likes RGB
    img_small_HSV = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)
    img_small_rgb = img_small_RGB / 255.  # scaled to [0, 1], only for plotting
    img_small_YUV = cv2.cvtColor(img_small, cv2.COLOR_BGR2YUV)
    img_small_YCrCb = cv2.cvtColor(img_small, cv2.COLOR_BGR2YCrCb)
    img_small_Lab = cv2.cvtColor(img_small, cv2.COLOR_BGR2Lab)
    # Plot and show
    plot3d(img_small_RGB, img_small_rgb)
    plt.show()
    
    plot3d(img_small_HSV, img_small_rgb, axis_labels=list("HSV"))
    plt.show()
    
    plot3d(img_small_YCrCb, img_small_rgb, axis_labels=list("YCrCb"))
    plt.show()
    
    plot3d(img_small_YUV, img_small_rgb, axis_labels=list("YUV"))
    plt.show()
    
    plot3d(img_small_Lab, img_small_rgb, axis_labels=list("Lab"))
    plt.show()