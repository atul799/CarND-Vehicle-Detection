# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 15:52:36 2018

@author: atpandey
"""


import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from functions import get_hog_features,convert_color_bgr

if __name__ == '__main__':
    cimg = cv2.imread('../cutouts/cutouts/red_car_for_color_hist.png')
    crgb=convert_color_bgr(cimg,color_space='RGB')
    cyuv=convert_color_bgr(cimg,color_space='YUV')
    cycrcb=convert_color_bgr(cimg,color_space='YCrCb')
    cychsv=convert_color_bgr(cimg,color_space='HSV')
    
    notcimg = cv2.imread('../cutouts/cutouts/nocar1_car_for_color_hist.png')
    notcrgb=convert_color_bgr(notcimg,color_space='RGB')
    notcyuv=convert_color_bgr(notcimg,color_space='YUV')
    notcycrcb=convert_color_bgr(notcimg,color_space='YCrCb')
    notchsv=convert_color_bgr(notcimg,color_space='HSV')
    
    
#    orient=8
#    pix_per_cell=8
#    cells_per_block=2
    orient=9
    pix_per_cell=8
    cells_per_block=2
#    orient=12
#    pix_per_cell=8
#    cells_per_block=8
 
    orient=11
    pix_per_cell=8
    cells_per_block=2   
##rgb sapce    
#    fcar_y, car_hog_y = get_hog_features(crgb[:,:,0], orient, pix_per_cell, cells_per_block, vis=True, feature_vec=True)
#    fnotcar_y, notcar_hog_y = get_hog_features(notcrgb[:,:,0], orient, pix_per_cell, cells_per_block, vis=True, feature_vec=True)
#    
#    fcar_u, car_hog_u = get_hog_features(crgb[:,:,0], orient, pix_per_cell, 8, vis=True, feature_vec=True)
#    fnotcar_u, notcar_hog_u = get_hog_features(notcrgb[:,:,0], orient, pix_per_cell, cells_per_block, vis=True, feature_vec=True)
#    
#    fcar_v, car_hog_v = get_hog_features(crgb[:,:,2], orient, pix_per_cell, cells_per_block, vis=True, feature_vec=True)
#    fnotcar_v, notcar_hog_v = get_hog_features(notcrgb[:,:,2], orient, pix_per_cell, cells_per_block, vis=True, feature_vec=True)
#    
#    # Visualize 
#    fh, axh = plt.subplots(2, 4, figsize=(10,10))
#    fh.subplots_adjust(hspace =.2, wspace=.1)
#    axh=axh.ravel()
#    axh[0].imshow(crgb)
#    axh[0].set_title('Car Image', fontsize=10)
#    axh[1].imshow(car_hog_y, cmap='gray')
#    axh[1].set_title('Car R HOG', fontsize=10)
#    axh[2].imshow(car_hog_u, cmap='gray')
#    axh[2].set_title('Car G HOG', fontsize=10)
#    axh[3].imshow(car_hog_v, cmap='gray')
#    axh[3].set_title('Car B HOG', fontsize=10)
#    
#    
#    axh[4].imshow(notcrgb)
#    axh[4].set_title('Not-Car Image', fontsize=10)
#    axh[5].imshow(notcar_hog_y, cmap='gray')
#    axh[5].set_title('Not-Car R  HOG', fontsize=10)
#    axh[6].imshow(notcar_hog_u, cmap='gray')
#    axh[6].set_title('Not-Car G  HOG', fontsize=10)
#    axh[7].imshow(notcar_hog_v, cmap='gray')
#    axh[7].set_title('Not-Car B  HOG', fontsize=10)    
    
    
##yuv sapce    
#    fcar_y, car_hog_y = get_hog_features(cyuv[:,:,0], orient, pix_per_cell, cells_per_block, vis=True, feature_vec=True)
#    fnotcar_y, notcar_hog_y = get_hog_features(notcyuv[:,:,0], orient, pix_per_cell, cells_per_block, vis=True, feature_vec=True)
#    
#    fcar_u, car_hog_u = get_hog_features(cyuv[:,:,0], orient, pix_per_cell, 8, vis=True, feature_vec=True)
#    fnotcar_u, notcar_hog_u = get_hog_features(notcyuv[:,:,0], orient, pix_per_cell, cells_per_block, vis=True, feature_vec=True)
#    
#    fcar_v, car_hog_v = get_hog_features(cyuv[:,:,2], orient, pix_per_cell, cells_per_block, vis=True, feature_vec=True)
#    fnotcar_v, notcar_hog_v = get_hog_features(notcyuv[:,:,2], orient, pix_per_cell, cells_per_block, vis=True, feature_vec=True)
#    
#    # Visualize 
#    fh, axh = plt.subplots(2, 4, figsize=(10,10))
#    fh.subplots_adjust(hspace =.2, wspace=.1)
#    axh=axh.ravel()
#    axh[0].imshow(crgb)
#    axh[0].set_title('Car Image', fontsize=10)
#    axh[1].imshow(car_hog_y, cmap='gray')
#    axh[1].set_title('Car Y HOG', fontsize=10)
#    axh[2].imshow(car_hog_u, cmap='gray')
#    axh[2].set_title('Car U HOG', fontsize=10)
#    axh[3].imshow(car_hog_v, cmap='gray')
#    axh[3].set_title('Car V HOG', fontsize=10)
#    
#    
#    axh[4].imshow(notcrgb)
#    axh[4].set_title('Not-Car Image', fontsize=10)
#    axh[5].imshow(notcar_hog_y, cmap='gray')
#    axh[5].set_title('Not-Car Y  HOG', fontsize=10)
#    axh[6].imshow(notcar_hog_u, cmap='gray')
#    axh[6].set_title('Not-Car U  HOG', fontsize=10)
#    axh[7].imshow(notcar_hog_v, cmap='gray')
#    axh[7].set_title('Not-Car V  HOG', fontsize=10)

#ycrcb sapce    
    fcar_y, car_hog_y = get_hog_features(cycrcb[:,:,0], orient, pix_per_cell, cells_per_block, vis=True, feature_vec=True)
    fnotcar_y, notcar_hog_y = get_hog_features(notcycrcb[:,:,0], orient, pix_per_cell, cells_per_block, vis=True, feature_vec=True)
    
    fcar_u, car_hog_u = get_hog_features(cycrcb[:,:,0], orient, pix_per_cell, 8, vis=True, feature_vec=True)
    fnotcar_u, notcar_hog_u = get_hog_features(notcycrcb[:,:,0], orient, pix_per_cell, cells_per_block, vis=True, feature_vec=True)
    
    fcar_v, car_hog_v = get_hog_features(cycrcb[:,:,2], orient, pix_per_cell, cells_per_block, vis=True, feature_vec=True)
    fnotcar_v, notcar_hog_v = get_hog_features(notcycrcb[:,:,2], orient, pix_per_cell, cells_per_block, vis=True, feature_vec=True)
    
    # Visualize 
    fh, axh = plt.subplots(2, 4, figsize=(10,10))
    fh.subplots_adjust(hspace =.2, wspace=.1)
    axh=axh.ravel()
    axh[0].imshow(crgb)
    axh[0].set_title('Car Image', fontsize=10)
    axh[1].imshow(car_hog_y, cmap='gray')
    axh[1].set_title('Car Y HOG', fontsize=10)
    axh[2].imshow(car_hog_u, cmap='gray')
    axh[2].set_title('Car Cr HOG', fontsize=10)
    axh[3].imshow(car_hog_v, cmap='gray')
    axh[3].set_title('Car Cb HOG', fontsize=10)
    
    
    axh[4].imshow(notcrgb)
    axh[4].set_title('Not-Car Image', fontsize=10)
    axh[5].imshow(notcar_hog_y, cmap='gray')
    axh[5].set_title('Not-Car Y  HOG', fontsize=10)
    axh[6].imshow(notcar_hog_u, cmap='gray')
    axh[6].set_title('Not-Car Cr  HOG', fontsize=10)
    axh[7].imshow(notcar_hog_v, cmap='gray')
    axh[7].set_title('Not-Car Cb  HOG', fontsize=10)


##hsv sapce    
#    fcar_y, car_hog_y = get_hog_features(cychsv[:,:,0], orient, pix_per_cell, cells_per_block, vis=True, feature_vec=True)
#    fnotcar_y, notcar_hog_y = get_hog_features(notchsv[:,:,0], orient, pix_per_cell, cells_per_block, vis=True, feature_vec=True)
#    
#    fcar_u, car_hog_u = get_hog_features(cychsv[:,:,0], orient, pix_per_cell, 8, vis=True, feature_vec=True)
#    fnotcar_u, notcar_hog_u = get_hog_features(notchsv[:,:,0], orient, pix_per_cell, cells_per_block, vis=True, feature_vec=True)
#    
#    fcar_v, car_hog_v = get_hog_features(cychsv[:,:,2], orient, pix_per_cell, cells_per_block, vis=True, feature_vec=True)
#    fnotcar_v, notcar_hog_v = get_hog_features(notchsv[:,:,2], orient, pix_per_cell, cells_per_block, vis=True, feature_vec=True)
#    
#    # Visualize 
#    fh, axh = plt.subplots(2, 4, figsize=(10,10))
#    fh.subplots_adjust(hspace =.2, wspace=.1)
#    axh=axh.ravel()
#    axh[0].imshow(crgb)
#    axh[0].set_title('Car Image', fontsize=10)
#    axh[1].imshow(car_hog_y, cmap='gray')
#    axh[1].set_title('Car H HOG', fontsize=10)
#    axh[2].imshow(car_hog_u, cmap='gray')
#    axh[2].set_title('Car S HOG', fontsize=10)
#    axh[3].imshow(car_hog_v, cmap='gray')
#    axh[3].set_title('Car V HOG', fontsize=10)
#    
#    
#    axh[4].imshow(notcrgb)
#    axh[4].set_title('Not-Car Image', fontsize=10)
#    axh[5].imshow(notcar_hog_y, cmap='gray')
#    axh[5].set_title('Not-Car H  HOG', fontsize=10)
#    axh[6].imshow(notcar_hog_u, cmap='gray')
#    axh[6].set_title('Not-Car S  HOG', fontsize=10)
#    axh[7].imshow(notcar_hog_v, cmap='gray')
#    axh[7].set_title('Not-Car V  HOG', fontsize=10)
    
    fh.tight_layout(pad=1.0, w_pad=0.5, h_pad=2.0)