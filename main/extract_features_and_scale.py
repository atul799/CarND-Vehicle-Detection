
"""
this module extracts car and not car features either all three types
color_hist/spatial_bin or hog features are extracted or only hog is extracted
and scales them using StandardScalar method

NOTE!! in extract_features function
matplotlib.image.imread (png read as float 0-1 instead of uint8 0-255) is used 
hence scaling is very improtant especially for color_hist


@author: atpandey
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import time

from sklearn.preprocessing import StandardScaler
# for scikit-learn >= 0.18 use:
from sklearn.model_selection import train_test_split
# from sklearn.cross_validation import train_test_split

from functions import extract_features,get_hog_features,bin_spatial,color_hist,convert_color_bgr


if __name__ == '__main__':
    
    #read car/notcar data
    cars = glob.glob('../vehicles/**/*.png', recursive=True)
    notcars = glob.glob('../non-vehicles/**/*.png', recursive=True)

    print("Number of car images:",len(cars))
    print("Number of noncar images:",len(notcars))
    
    
    
    
    # Feature extraction parameters
    #color_space='HSV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    #color_space = 'YUV' 
    color_space='YCrCb'
    
    orient = 11
    pix_per_cell = 16
    cell_per_block = 2
    hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
    
    spatial_size = (32, 32) # Spatial binning dimensions
    #spatial_size = (64, 64) # Spatial binning dimensions
    hist_bins = 32    # Number of histogram bins
    spatial_feat = True # Spatial features on or off
    hist_feat = True # Histogram features on or off
    hog_feat = True # HOG features on or off
    
    
    #checking effect of mpimg on color and spatial hist
    ft,axt=plt.subplots(1,3,figsize=(12,8))
    
    t = time.time()
    
    car_features = extract_features(cars, color_space=color_space, 
                                        spatial_size=spatial_size, hist_bins=hist_bins, 
                                        orient=orient, pix_per_cell=pix_per_cell, 
                                        cell_per_block=cell_per_block, 
                                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                                        hist_feat=hist_feat, hog_feat=hog_feat,axt=axt)
    ft.tight_layout()
    #plt.show()
    
    ftnc,axtnc=plt.subplots(1,3,figsize=(12,8))
    notcar_features = extract_features(notcars, color_space=color_space, 
                                        spatial_size=spatial_size, hist_bins=hist_bins, 
                                        orient=orient, pix_per_cell=pix_per_cell, 
                                        cell_per_block=cell_per_block, 
                                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                                        hist_feat=hist_feat, hog_feat=hog_feat,axt=axtnc)
    ftnc.tight_layout()
    
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to extract features...')
    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)  
    print('shape of feature array',X.shape)
    
     
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)
    
    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
    
    
    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)
    
    print('Using:',orient,'orientations',pix_per_cell,
        'pixels per cell and', cell_per_block,'cells per block')
    print('Feature vector length:', len(X_train[0]))