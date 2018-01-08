# -*- coding: utf-8 -*-
"""
this module uses find_cars function (hog subsampling) with multiple image sizes
to detect cars in an image
then apply heat map and threshold, use labels from scipy and draw bbox on image

@author: atpandey
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import time
import pickle
from sklearn.preprocessing import StandardScaler
# for scikit-learn >= 0.18 use:
from sklearn.model_selection import train_test_split
# from sklearn.cross_validation import train_test_split
from scipy.ndimage.measurements import label

from functions import find_cars,add_heat,apply_threshold,draw_labeled_bboxes


if __name__ == '__main__':
    
    for_pickle=pickle.load(open( 'colorspace_YCrCb_spatialbin_True_colhist_True.p', 'rb' ))
    
    svc=for_pickle['svc']
    X_scaler=for_pickle['scaler']
    color_space=for_pickle['color_space'] 
    spatial_size=for_pickle['spatial_size']
    hist_bins=for_pickle['hist_bins']
    orient=for_pickle['orient']
    pix_per_cell=for_pickle['pix_per_cell']
    cell_per_block=for_pickle['cell_per_block']
    hog_channel=for_pickle['hog_channel']
    spatial_feat=for_pickle['spatial_feat']
    hist_feat=for_pickle['hist_feat']
    hog_feat=for_pickle['hog_feat']
    
    show_scalar=False
    #ffc,axfc=plt.subplots(1,2)
    test_img = mpimg.imread('../test_images/test1.jpg')
    print("img shape",test_img.shape)
    ystart = 400
    ystop = 656
    scale = 1.5

    print("spatial enabled:",spatial_feat,"Hist enabled:",hist_feat)
    param_list=((400,464,1.0),(416,480,1.0),(400,496,1.5),(432,528,1.5),(400,528,2.0),(432,560,2.0),(400,596,3.5),(464,660,3.5))
    bbox=[]
    for j in   param_list:
        ystart=j[0]
        ystop=j[1]
        scale=j[2]
        bbox_l = find_cars(test_img, ystart, ystop, scale, color_space, hog_channel, svc, 
                 X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins,spatial_feat,hist_feat,axfc=None,show_scalar=show_scalar)
        bbox.extend(bbox_l)
    print(len(bbox), 'bboxes found in image')

    im_bbox=draw_boxes(test_img, bbox, color=(0, 0, 255), thick=6)
    f,p=plt.subplots(1,4,figsize=(20,15))
    #p[0].imshow(test_img)
    #p[0].set_title("Orig image")
    p[0].imshow(im_bbox)
    p[0].set_title("Multi Window Vehicle match")
    
    #Heat map
    heat = np.zeros_like(test_img[:,:,0])
    heat = add_heat(heat, bbox)
    p[1].imshow(heat, cmap='hot')
    p[1].set_title("Heat Map")
    
    #apply threshold
    heat_thres = apply_threshold(heat, 1)
    p[2].imshow(heat_thres, cmap='hot')
    p[2].set_title("HeatMap thresholded")

    heatmap = np.clip(heat_thres, 0, 255)
    labels = label(heatmap)
    print(labels[1], 'cars found')
    draw_img = draw_labeled_bboxes(np.copy(test_img), labels)
    p[3].imshow(draw_img)
    p[3].set_title("Heatmapped BBox")