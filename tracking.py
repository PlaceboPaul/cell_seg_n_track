    # TRACKING EXPLORATION #


## Library ##
import os
import numpy as np
import random
import math
from time import sleep
#
from skimage import io
from skimage.filters import threshold_otsu, gaussian
from skimage.feature import canny, peak_local_max
from skimage.morphology import binary_opening, binary_closing, flood_fill, binary_erosion, binary_dilation, erosion
from skimage.measure import label, regionprops
from skimage.segmentation import watershed, clear_border
import scipy.ndimage as ndi
from scipy.optimize import linear_sum_assignment
#
import cv2
import matplotlib.pyplot as plt
import pandas as pd
    ## Library ##


    ## Tracking Preparation ##

### Test Data
gt_imgs = list(np.load("Objects/gt_list.npy", allow_pickle=True))
seg_imgs_canny = list(np.load("Objects/seg_list_canny.npy", allow_pickle=True)) # We use canny because it gave best results and machine learning would take too long
t91_gt = gt_imgs[91][60:320, 730:1030]; plt.figure(); plt.imshow(t91_gt); plt.axis("off")
t91_gt2 = gt_imgs[91][355:, :365]; plt.figure(); plt.imshow(t91_gt2); plt.axis("off")
t091_seg = seg_imgs_canny[91]; plt.figure(); plt.imshow(t091_seg,"gray"); plt.axis("off")
t91_sub = clear_border(t091_seg[60:320, 730:1030]); plt.figure(); plt.imshow(t91_sub,"gray"); plt.axis("off") #subset with some cells
t91_sub2 = clear_border(t091_seg[355:, :365]); plt.figure(); plt.imshow(t91_sub2,"gray"); plt.axis("off") #subset with some cells

### Creating pixel objects (centroids) for each cell
    #regionprobs
t91_sub_lab = label(t91_sub, background=0); plt.figure(); plt.imshow(t91_sub_lab,"gray"); plt.axis("off")
props = regionprops(t91_sub_lab)
plt.figure(); plt.axis('off')
plt.imshow(t91_sub, cmap='gray')
centroids = np.zeros(shape=(len(np.unique(t91_sub_lab)),2))
for i,prop in enumerate(props):
    my_centroid = prop.centroid
    centroids[i,:]= my_centroid
    plt.plot(my_centroid[1],my_centroid[0],'r.')
plt.show(); plt.title("t91[60:320, 730:1030] Labeled and regionprobs ") # print(centroids)
plt.savefig("My_Imgs/Testing/t091_label-regionprobs.png", dpi=300)
    # Peak local maxima
distance = ndi.distance_transform_edt(t91_sub); plt.figure(); plt.imshow(distance,"gray"); plt.axis("off")
coords = peak_local_max(distance, footprint=np.ones((15,15)), labels=t91_sub, threshold_abs=4) # threshold_abs needs to be adjusted to exclude small objects --> 4 determined by trying. Small objects but also very tiny nuclei removed. Trade-off
                                                                                                # footrprint best with 15 because with 10 some objects had two pixels with more than 10x10 distance
mask = np.zeros(distance.shape, dtype=bool); mask[tuple(coords.T)] = True                       # coords contain rows for each pixel with Y,X coord
centroids = erosion(label(binary_dilation(mask,selem=np.ones((5,5))), background=0, connectivity=1), selem=np.ones((5,5))); plt.figure(); plt.imshow(centroids) # Connectivity doesn't go beyond 2 so we can't label pixels with small distance the same
                                                                                                                                                                        # solution: binary_dilate the centroid pixels to connect close pixels, label the big object, then use normal erosion to revert back to one pixel but now they have same label
# another subset of image                                                                                                                                                 # selem 5,5 best to connect close pixels
distance2 = ndi.distance_transform_edt(t91_sub2)
coords2 = peak_local_max(distance2, footprint=np.ones((15,15)), labels=t91_sub2, threshold_abs=4) # with 15,15 we lose one nucleus of cells that overlap but if we go lower we get unwanted splits. Trade-off
mask2 = np.zeros(distance2.shape, dtype=bool); mask2[tuple(coords2.T)] = True
centroids2 = erosion(label(binary_dilation(mask2,selem=np.ones((5,5))), background=0, connectivity=1), selem=np.ones((5,5))); plt.figure(); plt.imshow(centroids2)
plt.figure(); plt.imshow(t91_gt)
for i in range(coords.shape[0]):
    plt.plot(coords[i, 1], coords[i, 0], "r.") # cool that plt.plot uses cooridates of image (y starting from top at zero and increases downwards). Usually it plots with y increasing towards top
plt.savefig("My_Imgs/Testing/t091_label-localmax.png", dpi=300)
# whole image
distance3 = ndi.distance_transform_edt(t091_seg)
coords3 = peak_local_max(distance3, footprint=np.ones((15,15)), labels=t091_seg, threshold_abs=4)
mask3 = np.zeros(distance3.shape, dtype=bool); mask3[tuple(coords3.T)] = True
centroids3 = erosion(label(binary_dilation(mask3,selem=np.ones((5,5))), background=0, connectivity=1), selem=np.ones((5,5))); plt.figure(); plt.imshow(centroids3)
plt.figure(); plt.imshow(t091_seg, "gray")
for i in range(coords3.shape[0]):
    plt.plot(coords3[i, 1], coords3[i, 0], "r.")
plt.savefig("My_Imgs/Testing/t091_whole-localmax.png", dpi=300)

### Document centroid coordinates for each frame in Data Frame
# one frame
first_frame = np.zeros(coords.shape[0], dtype="int")
ID_col = centroids[tuple(coords.T)] #this seems to be fine. When I change ID_col it doesn't alter centroids
x_col = coords[:,1]; y_col = coords[:,0]
tracking_DF_test = pd.DataFrame(dict(zip(["frame", "id", "x", "y", "track_id"],[first_frame, ID_col, x_col, y_col, first_frame]))).drop_duplicates(subset=["id"]).drop_duplicates(subset=["id"])
# all frames
def tracking_DataFrame(seg_list):
    tracking_DF = pd.DataFrame(columns=["frame", "track_id", "id", "x", "y"])
    for index, seg in enumerate(seg_list):
        # centroids
        distance = ndi.distance_transform_edt(seg)
        coords = peak_local_max(distance, footprint=np.ones((15, 15)), labels=seg, threshold_abs=4)
        mask = np.zeros(distance.shape, dtype=bool); mask[tuple(coords.T)] = True
        centroids = erosion(label(binary_dilation(mask, selem=np.ones((5, 5))), background=0, connectivity=1), selem=np.ones((5, 5)))
        # data frame
        frame = np.full(coords.shape[0], index, dtype="int"); track_id = np.zeros(coords.shape[0], dtype="int")
        id = centroids[tuple(coords[::-1,:].T)]; x = coords[::-1, 1]; y = coords[::-1, 0] # changed coords so that id column doesn't sart with highest id for each frame but with lowest
        frame_DF = pd.DataFrame(dict(zip(["frame", "track_id", "id", "x", "y"], [frame, track_id, id, x, y]))).drop_duplicates(subset=["id"]) # changed order of columns
        tracking_DF = pd.concat([tracking_DF, frame_DF], ignore_index=True)
    return tracking_DF
#tracking_DF = tracking_DataFrame(seg_imgs_canny); tracking_DF.to_csv("Objects/tracking_DF.csv", index=False)
tracking_DF = pd.read_csv("Objects/tracking_DF_new.csv")

### Visualizing frames as video
img_list = list(np.load("Objects/img_list.npy", allow_pickle=True)) #list of all images.
seg_list = list(np.load("Objects/seg_list_canny.npy", allow_pickle=True)) #list of all canny segmentation.
img_list_8bit = list(np.load("Objects/img_list_8bit.npy", allow_pickle=True)) # others are 32 bit which makes problems with cv2 but 8 bit works fine when plotting!!!
    #try with matplotlib
for i in range(len(img_list)):
    plt.imshow(img_list[i], "gray"); plt.axis("off")
    frame_centroids = tracking_DF[["x", "y"]][tracking_DF["frame"] == i]
    plt.plot(frame_centroids["x"], frame_centroids["y"], "r.")
    plt.clf()
    # plt.show(), plt.clf() did not work either. Same problem...
    sleep(3) #problem: the plotting works but somehow no image is shown for each loop.... not even when I only plot the image each loop
            # the images don't show until I break the loop then the current one appears with centroids... but sequence doesn't work
    plt.close("all")
#i give up. lets use openCV!!!
    # Opencv



    ## Tracking Protocol ##

### Dataframe preparation
# test this
#1) Extract Row indices for each frame
def get_frame_ind(dataframe, frame):
    return dataframe.index[dataframe.frame == frame]
# row_inds = get_frame_ind(tracking_DF, 0) #test
#2) Assign unused track_IDs to points with 0. Later important to give unmerged points a new track ID indicating a new track
def assign_new_IDs(dataframe, row_indices):
    max_ID = dataframe.track_id.max() # last ID that was assigned previously
    for row in row_indices:
        if dataframe.track_id.at[row] == 0: # if 0 then it gets new ID
            max_ID += 1 # new unused ID
            dataframe.track_id.at[row] = max_ID
    return
# assign_new_IDs(tracking_DF, row_inds)

### Unique Occurences
#3) Get index of row with most recent occurence of each track_id within certain range from current frame
def get_all_tracks(dataframe, start_frame, current_frame):
    track_rows = []
    index_subset = dataframe.index[dataframe.frame.isin(np.arange(start_frame, current_frame))]
    unique_IDs = np.unique(dataframe.track_id[index_subset])
    for uni in unique_IDs:
        if uni != 0:
            last_occ = index_subset[dataframe.iloc[index_subset].track_id == uni][-1] # last occurence
            track_rows.append(last_occ)
    return track_rows
# test_tracks = get_all_tracks(tracking_DF, 0,1)

### Calculate Distances. Points in Current Frame with every last recent occurance of all available track_IDs
#4) distance between two rows. If distance too large, return infinity. Cost of merging them
def calculate_cost(dataframe, row_current, row_last, thresh):
    r1 = dataframe.iloc[row_current]; r2 = dataframe.iloc[row_last] # rows of each point
    distance = math.sqrt((r1.x - r2.x)**2 + (r1.y - r2.y)**2) # euclidean distance
    if distance > thresh:
        distance = infinity # math.inf gives you weird infinity object which causes problems later so we do like this
    return distance
# calculate_cost(tracking_DF, 0,5,50) # test
#5) calculate cost matrix
def cost_matrix(dataframe, rows_current_frame, rows_previous_tracks, thresh):
    cost_array = [calculate_cost(dataframe, i, j, thresh) for i in rows_current_frame for j in rows_previous_tracks] #list comprehension cuase we lit
    return np.array(cost_array).reshape(len(rows_current_frame), -1) # rows of cost matrix should be objects of current frame
# test_matrix = cost_matrix(tracking_DF, get_frame_ind(tracking_DF, 1), get_frame_ind(tracking_DF, 0), 50)
#6) use Hungarian (also Kuhn-Munkres) Algorithm to determine assignment with lowest cost and assign new track IDs
def merge_tracks(dataframe, rows_current_frame, rows_previous_tracks, cost_mat):
    cost_rows, cost_cols = linear_sum_assignment(cost_mat)
    for row, col in zip(cost_rows, cost_cols):
        if cost_mat[row, col] != infinity: # algorithm assigns infinities too, but we don't want em so here we make sure
            dataframe.track_id[rows_current_frame[row]] = dataframe.track_id[rows_previous_tracks[col]]
    return
# merge_tracks(tracking_DF, get_frame_ind(tracking_DF, 1), get_frame_ind(tracking_DF, 0), test_matrix)
# assign_new_IDs(tracking_DF, get_frame_ind(tracking_DF, 1))

### The tracking function
def track_nuclei(dataframe, linking_thresh, frame_range):
    assign_new_IDs(dataframe, get_frame_ind(dataframe, 0))
    for frame in dataframe.frame.unique():
        if frame == 0: continue # ignore frame 0
        print("Processing frame %s..." % frame)
        #
        current_rows = get_frame_ind(dataframe, frame) # rows of current frame
        recent_occ_rows = get_all_tracks(dataframe, frame-frame_range, frame) # rows of last occurrence of unique IDs from within some frame range ago
        #
        cost_mat = cost_matrix(dataframe, current_rows, recent_occ_rows, linking_thresh)
        merge_tracks(dataframe, current_rows, recent_occ_rows, cost_mat)
        #
        assign_new_IDs(dataframe, current_rows) # zeros get new label
    print("Done")
    return
infinity = 1000000000
tracking_DF = pd.read_csv("Objects/tracking_DF_new.csv")
track_nuclei(tracking_DF, linking_thresh=50, frame_range=3) # With 15, 2 we have 373 as highest assigned track ID
                                                            # With 25, 3 we have 267 as highest
                                                            # With 50, 4 we have 209 as highest
                                                            # With 50, 3 we have 218 as highest
                                                            # With 50, 2 we have 244 as highest
                                                            # With 100, 4 we have 164 as highest
                                                            # With 100, 2 we have 199 as highest... highest label (id) in a frame was 135
tracking_DF.to_csv("Objects/tracking_DF_tracked_50-3.csv", index=False)

##


import seaborn as sns
palette = sns.color_palette(palette="tab10", n_colors=200)
