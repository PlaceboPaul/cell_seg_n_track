        # VISUALIZING THE TRACKS #

    ## Library ##
import numpy as np
# from time import sleep
#
#
import cv2
import pandas as pd
import seaborn as sns

    ## Data ##
tracking_DF = pd.read_csv("Objects/tracking_DF_tracked_50-3.csv")
img_list_8bit = list(np.load("Objects/img_list_8bit.npy", allow_pickle=True)) # 8 but necessary for correct cv2 display
imgs_bgr = []
for img in img_list_8bit: # needs to be in BGR for red circles to be drawn on it. BGR instead of RGB because red is (0,0,255)
    bgr_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    imgs_bgr.append(bgr_img)

    ## Visualizing ##
track_cells = tracking_DF.track_id[tracking_DF.frame == 90] #cells up to which frame should leave trajectory. Cell IDs terminated before that frame or started afterwards are not included.
col_palette_rgb = sns.color_palette(palette="tab10", n_colors=len(track_cells)) # RGB for each cell
col_palette = col_palette_rgb[:][::-1] #BGR

frame = 0
while True:
    image = imgs_bgr[frame]
    #
    frame_subset = tracking_DF[tracking_DF.frame == frame] # the indices are subsetted too so we need to either use those subsetted indices or iloc to start from 0
    for i in range(len(frame_subset)): # for this i we need to subset with iloc
        cv2.circle(image, center=(frame_subset.x.iloc[i], frame_subset.y.iloc[i]), radius=3, color=(0, 0, 255), thickness=-1) #draw circle and fill with red. thickness=-1 so that circle is filled and not translucent i guess
        cv2.putText(image, text=str(frame_subset.track_id.iloc[i]), org=(frame_subset.x.iloc[i], frame_subset.y.iloc[i] + 10), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0,255,0), thickness=1) #put label slightly above centroid
    #
    for cell_ind in range(len(track_cells)):
        cell_subset = tracking_DF[np.logical_and(tracking_DF.track_id == track_cells.iloc[cell_ind], tracking_DF.frame <= frame)]
        color = np.array(col_palette[cell_ind]) * 255 # otherwise its numbers from 0 to 1 which end up being black/gray
        for j in range(1, len(cell_subset)):
            point1 = (cell_subset.x.iloc[j-1], cell_subset.y.iloc[j-1])
            point2 = (cell_subset.x.iloc[j], cell_subset.y.iloc[j])
            cv2.line(image, pt1=point1, pt2=point2, color=color, thickness=2)
    #
    imageS = cv2.resize(image, (int(1100*0.9), int(700*0.9))) #otherwiese image doesn't fit on my screen :(
    cv2.imshow("img", imageS)
    frame += 1 # got one frame forward
    if frame == 92: # 91 is last index of last image so reset
        frame = 0
    key = cv2.waitKey(200) # wait 200 ms between each frame
    if key & 0xFF == ord('q'): # no idea why this works but if I press q then the whole loop is exited
        break                   # ord() gives Unicode of the character and waitKey I guess is any button I press
cv2.destroyAllWindows() # close all windows

# Save the images for video
for i in range(len(imgs_bgr)):
    cv2.imwrite(f'My_Imgs/Tracking/Trajectories_50-3/traject_50-3_img0{str(i).zfill(2)}.jpg', imgs_bgr[i], [cv2.IMWRITE_JPEG_QUALITY, 100])

#Evaluation
    # tracking with 15,2 looks good but let's see if we can increase performance with
        # higher frame range because sometimes nuclei weren't detected in some frames
            # I think 4 is too high because some nuclei move a lot and 4 frames ago a nuclei might have been where another is in current frame...
            # 3 seems ok
        # higher distance threshold because sometimes they move very fast from frame to frame leading to high distance
            # 100 is too high because sometimes distant nuclei somehow switch IDs
            # 50 seems ok with not too many nuclei on a lump
        # cells popping in and out of frame increase track_id number every time they pop in
    # Evaluation method missing
        # Optimizations:
            # One could increase the clear border footprint so that cells frequently touching the frame border are constantly exlcuded and don't increase track_ids
            # Increase footprint of peak local maxima and subsequent dilation-labelling to prevent double nuclei in stretched cells.
                #--> However, this'll also prevent detection of overlapping or just very close small nuclei
        # Outlook: What can we do with this information? --> Research
