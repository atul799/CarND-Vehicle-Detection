
"""
This module implements the complete pipeline for vehicle detection

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

from functions import find_cars,add_heat,apply_threshold,draw_labeled_bboxes,Frame_Detect


def pipeline(img):
        
    
    
    bbox=[]
    for j in   param_list:
        ystart=j[0]
        ystop=j[1]
        scale=j[2]
        bbox_l = find_cars(img, ystart, ystop, scale, color_space, hog_channel, svc, 
                 X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins,spatial_feat,hist_feat,axfc=None,show_scalar=show_scalar)
        bbox.extend(bbox_l)
    #print(len(bbox), 'bboxes found in image')
    
    if len(bbox) > 0:
        frames_track.add_bbox(bbox)
    #######################################
    heat = np.zeros_like(img[:,:,0])
    #heat = add_heat(heat, bbox)
    #heat = apply_threshold(heat, 1)
    for box in frames_track.prev_bbox:
        heat = add_heat(heat, box)
    
    #divide the heatmap image by number of frames it been tracked for
    heat=heat//(frames_track.n_frames)
    #heat=heat//len(frames_track.prev_bbox)
    heat = apply_threshold(heat, 1)
    
    
    heatmap = np.clip(heat, 0, 255)
    ########################################
    
    labels = label(heatmap)
    #print(labels[1], 'cars found')
    draw_img = draw_labeled_bboxes(np.copy(img), labels)
    
    
    #######################################
    #prepare heatmap to be added to main image
    #resize heatmap
    hmcopy=np.copy(heatmap)
    heat_to_add=cv2.resize(hmcopy,(320,192),cv2.INTER_CUBIC)
    #resize image
    resized_img = cv2.resize(img,(320,192),cv2.INTER_CUBIC)
    resized_gray = cv2.cvtColor(resized_img , cv2.COLOR_RGB2GRAY)
    res_R=np.copy(resized_gray)
    res_R[(heat_to_add>0)] =255
    res_G =np.copy(resized_gray)
    res_G[(heat_to_add>0)] =0
    res_B =np.copy(resized_gray)
    res_B[(heat_to_add>0)] =0
    
    draw_img[0:192,0:320,0]=res_R     #np.clip(res_R,0,255)
    draw_img[0:192,0:320,1]=resized_gray
    draw_img[0:192,0:320,2]=resized_gray
    
    
    

    return draw_img


if __name__=='__main__':
    #pipeline_no smoothing
    param_list=((400,464,1.0),(416,480,1.0),(400,496,1.5),(432,528,1.5),
                (400,528,2.0),(432,560,2.0),(400,596,3.5),(464,660,3.5))

    #param_list = [(400, 600, 1.2), (400, 470, 1.0), (420, 480, 1.0), 
    #                  (400, 500, 1.5), (430, 530, 1.5), (400, 530, 2.0),
    #                  (470, 660, 3.0)]
    
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
    frames_track=None
    frames_track = Frame_Detect()
    frames_track.n_frames=1
    test_images = glob.glob('../test_images/test*.jpg')

    fig, axs = plt.subplots(3, 2, figsize=(16,14))
    fig.subplots_adjust(hspace = .004, wspace=.002)
    axs = axs.ravel()

    for i, im in enumerate(test_images):
        print("idx:",i,"imname:",im)
        axs[i].imshow(pipeline(mpimg.imread(im)))
        axs[i].axis('off')
        
    figl, axsl = plt.subplots(figsize=(16,14))
    axsl.imshow(pipeline(mpimg.imread(im)))