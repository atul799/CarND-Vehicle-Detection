# -*- coding: utf-8 -*-
"""
use of slide window and search_windows mmethod to find vehicles

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

from functions import search_windows,extract_features,get_hog_features,bin_spatial,color_hist,convert_color_rgb


if __name__ == '__main__':
    
    for_pickle=pickle.load(open( 'colorspace_YCrCb_spatialbin_True_colhist_True.p', 'rb' ))
    
    
    
    img_name='../cutouts/cutouts/bbox-example-image.jpg'
    img = cv2.imread(img_name)
    
    
    image=convert_color_rgb(img,color_space='RGB')
    
    
    draw_image = np.copy(image)
    
    

    y_start_stop = [400, 700]
    
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
    
    for_pickle={'svc':svc,'scaler':X_scaler,'orient':orient,'pix_per_cell':pix_per_cell,
                            'cell_per_block':cell_per_block,'spatial_size':spatial_size,
                            'hist_bins':hist_bins,'spatial_feat':spatial_feat,'hist_feat':hist_feat,'hog_feat':hog_feat,'hog_channel':hog_channel,'color_space':color_space}
    
    
    # Uncomment the following line if you extracted training
    # data from .png images (scaled 0 to 1 by mpimg) and the
    # image you are searching is a .jpg (scaled 0 to 255)
    #image = image.astype(np.float32)/255
    #xy_window=(96, 96)
    xy_window=(64, 64)
    
    windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop, 
                        xy_window=xy_window, xy_overlap=(0.5, 0.5))
    
    hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)                       
    
    
    
    
    
    window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)                    
    
    plt.imshow(window_img)