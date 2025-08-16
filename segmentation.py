        # SEGMENTATION METHODS #



    ## Library ##
import os
import numpy as np
import random
import time
#
from skimage import io
from skimage.filters import threshold_otsu, gaussian
from skimage.feature import canny, peak_local_max
from skimage.morphology import binary_opening, binary_closing, flood_fill, binary_erosion
from skimage.segmentation import watershed, clear_border
import scipy.ndimage as ndi
#
import matplotlib.pyplot as plt
#
# import pandas as pd
    ## Library ##



    ## Data Exploration Results ##
# We don't convert image to 8bit since its not that important and we don't have to worry about quality loss
    # But let's subtract the min value so it starts at 0
# Don't put all 92 images in an array. Load each in an iteration and override the previous??? I think that's better



    ## Reading Data ##
t000 = io.imread("01/t000.tif"); t000 = t000 - t000.min()
#gt
t000_gt = io.imread("01_ST/SEG/man_seg000.tif"); t000_gt = t000_gt > 0



    ## Segmentation ##


### Dice Coefficient and Jaccard Distance
def evaluate_seg(H, G): # H: My segmentation, G: Ground truth
    dice = 2 * (H * G).sum() / (H.sum() + G.sum())
    jacc = (H * G).sum() / ((H.sum() + G.sum()) - (H * G).sum())  # HnG / (H+G - HnG) same as IoU (Intersection of Union)
    return dice, jacc

### Function for mean Dice of all images with given method
def mean_evaluations(image_dir, gt_dir, seg_method):
    dices = []
    jaccs = []
    for i in range(len(os.listdir(image_dir))):
        img = io.imread(f'{image_dir}t0{str(i).zfill(2)}.tif')
        gt = io.imread(f'{gt_dir}man_seg0{str(i).zfill(2)}.tif') > 0 # make gt binary
        img_seg = seg_method(img)
        dices.append(evaluate_seg(img_seg, gt)[0])
        jaccs.append(evaluate_seg(img_seg, gt)[1])
    return f'Mean dice = {np.mean(dices).round(4)}', f'Mean Jaccard = {np.mean(jaccs).round(4)}'

### Image and GT lists Otsu and Canny for object bases metric evaluation
def get_image_lists(image_dir, seg_method, subset=[]): # if you don't want all images, use subset. add gt_dir if needed
    segs = []; gts = []
    if len(subset) == 0:
        for i in range(len(os.listdir(image_dir))):
            img = io.imread(f'{image_dir}t0{str(i).zfill(2)}.tif')
            # gt = io.imread(f'{gt_dir}man_seg0{str(i).zfill(2)}.tif')
            img_seg = seg_method(img)
            segs.append(img_seg)#; gts.append(gt)
    else:
        for i in subset:
            img = io.imread(f'{image_dir}t0{str(i).zfill(2)}.tif')
            #gt = io.imread(f'{gt_dir}man_seg0{str(i).zfill(2)}.tif')
            img_seg = seg_method(img)
            segs.append(img_seg)#; gts.append(gt)
    return segs#, gts

### Image directories
image_dir = '01/'; gt_dir = '01_ST/SEG/'

### Complete Otsu thresholding in a function #added gauss function afterwards (02.05)
def otsu_method(img):
    T = threshold_otsu(img)
    return clear_border(img > T)
def otsu_gauss(img):
    img_gauss = gaussian(img, sigma=5, preserve_range=True)
    T = threshold_otsu(img_gauss)
    return clear_border(img_gauss > T)

### Segmentation with Canny-Operator and filling edges
def canny_segmentation(img):
    img_canny = canny(img, sigma=1, low_threshold=35, high_threshold=75)
    img_closed = binary_closing(img_canny, selem=np.ones((5,5))) # close open edges
    img_invert = np.invert(img_closed)
    for i in np.arange(0, img.shape[0], 50): # try different seeds points for flood_fill
        img_flood = flood_fill(img_invert.astype("int"), (i, 0), 0, connectivity=1)
        if np.sum(img_flood)/(700*1100) < 0.1:
            break # if a nucleus blocked the corner (0,0) try next seed point
    return clear_border(binary_opening(img_closed + img_flood, selem=np.ones((3,3)))) # selem of 3,3 or 5,5 don't change mean dicerino. 5,5 removes small nois objects but also small barely detected nuclei -> bad for object based metric
                                                                                                # lets do 3,3 as a compromise
                                                                                    # clear_border removes objects touching border because gt images don't include them --> increased dice

### Watershed for labelling
def watershedding(seg_imgs): #list of segmented imgs to label with watershed
    seg_list_lab = []
    for image in seg_imgs:
        dist_img = ndi.distance_transform_edt(image)
        coords_img = peak_local_max(dist_img, footprint=np.ones((10, 10)), labels=image)
        mask = np.zeros(dist_img.shape, dtype=bool)
        mask[tuple(coords_img.T)] = True
        markers, _ = ndi.label(mask, structure=np.ones((3, 3)))
        seg_list_lab.append(watershed(-dist_img, markers, mask=image))
    return seg_list_lab

### Otsu method
mean_dicerino_otsu = mean_evaluations(image_dir, gt_dir, otsu_method)
mean_dicerino_otsu # ('Mean dice = 0.68', 'Mean Jaccard = 0.52') after clear_border only Jaccard moved up by 0.01
mean_dicerino_otsu_gauss = mean_evaluations(image_dir, gt_dir, otsu_gauss); mean_dicerino_otsu_gauss # sigma=4 ('Mean dice = 0.72', 'Mean Jaccard = 0.57')
                                                                                                    # sigma=5 ('Mean dice = 72.47%', 'Mean Jaccard = 56,88%')
                                                                                                    # sig=6 (Mean dice = 0.72', 'Mean Jaccard = 0.56)

### Edge detection for segmentation - Canny
mean_dicerino_canny = mean_evaluations(image_dir, gt_dir, canny_segmentation) #ca 1 min
mean_dicerino_canny # ('Mean dice = 0.90', 'Mean Jaccard = 0.81') massive improvement but still not sure how to utilize thresholds in canny well
                    # ('Mean dice = 90.63%', 'Mean Jaccard = 82,93%') when using clear_border.... so border objects didn't make thaaat much difference after all...
                    # ('Mean dice = 0.88', 'Mean Jaccard = 0.79') with sigma = 2 instead of 1

### Object-based evaluation
from segmetrics.study     import *
from segmetrics.regional  import *
from segmetrics.boundary  import *
from segmetrics.detection import *

# lists
subset = random.sample(range(92), 10) # [64, 24, 0, 18, 32, 59, 36, 89, 62, 68]
start = time.time(); seg_list = get_image_lists(image_dir, canny_segmentation); end = time.time(); end-start
start = time.time(); seg_list_lab = watershedding(seg_list); end = time.time(); end-start
#
# np.save("Objects/seg_list_otsu-gauss_water.npy", seg_list_lab, allow_pickle=True) # saves as array of arrays, so we have to revert back to list when loading
otsu_list = list(np.load("Objects/seg_list_otsu-gauss.npy", allow_pickle=True))
otsu_list_water = list(np.load("Objects/seg_list_otsu-gauss_water.npy", allow_pickle=True))
gt_list = list(np.load("Objects/gt_list.npy", allow_pickle=True))
# REGIONAL
study_regional = Study()
study_regional.add_measure(ISBIScore(), 'isbi') # so idk exactly how isbi works (research) but it's listed with regional (pixel-wise) metrics in module "regional.py"
study_regional.add_measure(Hausdorff(mode='sym'), 'HSD (sym)')
study_regional.add_measure(Dice(), 'Dice')
study_regional.add_measure(JaccardSimilarityIndex(), 'Jaccard')
for groundtruth, result in zip(gt_list, seg_list):
    study_regional.set_expected(groundtruth, unique=True)
    study_regional.process(result, unique=False) #not labeled
study_regional.print_results()  # Otsu (10 imgs) Dice:68.72%, HSD(sym):172.159, Jaccard:52.47%, isbi:43.67%
                                # Otsu (92 imgs) Dice:68.19%, HSD(sym):169.571, Jaccard:51.82 %, isbi:44.35%
                                # Canny (same 10 imgs) Dice:91.00 %, HSD(sym):118.518, Jaccard:83.56%, isbi:76.02%
                                # Canny (same 92 imgs) Dice:90.63%, HSD(sym):138.807, Jaccard:82.93%, isbi: 74.97 % # subset was representative enough of whole data set
# OBJECT-BASED legacy
study_object = Study()
study_object.add_measure(ObjectBasedDistance(ISBIScore(), skip_fn=True), 'isbi') #legacy 20s vs 5 min for not-legacy (for 92 imgs that's 3 min vs 45 min
study_object.add_measure(ObjectBasedDistance(Hausdorff(mode='sym'), skip_fn=True), 'HSD (sym)')
study_object.add_measure(ObjectBasedDistance(Dice(), skip_fn=True), 'Dice')
study_object.add_measure(ObjectBasedDistance(JaccardSimilarityIndex(), skip_fn=True), 'Jaccard')
for groundtruth, result in zip(gt_list, otsu_list_water): # seg list or seg_list_lab but then adjust 'unique' parameter!!!
    study_object.set_expected(groundtruth, unique=True)
    study_object.process(result, unique=True) # True when watershedded!!!
study_object.print_results()    # Otsu (92 imgs) Dice:67.76%, HSD(sym):6.63648, Jaccard:58.59%, isbi:56.56%
                                # Otsu_gauss Dice: Dice: 69.49 %, Jaccard: 58.11 %, isbi: 53.78 %, HSD (sym): 11.6947
                                # Otsu+watershed (92) Dice:62.50%, HSD(sym):6.79969, Jaccard:53.08%, isbi:46.65% #worse :(( maybe its because of the subset
                                # Otsu_gauss+water Dice: 67.69 %, Jaccard: 58.11 %, isbi: 51.37 %, HSD (sym): 6.13228
                                # Canny no-legacy (same 10, took 5min) Dice:77.56%, HSD(sym):7.92574, Jaccard:72.20%, isbi:71.77%
#legacy comparison only here    # Canny legacy (same 10, 20s) Dice:85.96%, HSD(sym):6.12971, Jaccard:78.82 %, isbi: 78.35% better result when using legacy (skip falase negatives???)
                                            # we can use this and add false negatives to that? i mean the average of false negatives? or on last 10 imgs because most nuclei
                                # Canny (92 imgs) Dice:85.45%, HSD(sym):6.23803, Jaccard: 78.17 %, isbi:77.44%
                                # Canny+watershed (92) Dice:77.65%, HSD(sym):4.23553, Jaccard:71.25%, isbi:66.28% # worse than not watershed :(((
                                    # don't really know why cause to me it seems like watershed did a good job overall
# OBJECT-BASED not-legacy (without skip_fn)
study_object = Study() # legacy skips FN and therefore makes results better
study_object.add_measure(ObjectBasedDistance(ISBIScore()), 'isbi')
study_object.add_measure(ObjectBasedDistance(Hausdorff(mode='sym')), 'HSD (sym)')
study_object.add_measure(ObjectBasedDistance(Dice()), 'Dice')
study_object.add_measure(ObjectBasedDistance(JaccardSimilarityIndex()), 'Jaccard')
for groundtruth, result in zip(gt_list, seg_list_lab):
    study_object.set_expected(groundtruth, unique=True)
    study_object.process(result, unique=True)
study_object.print_results()    # Canny (90) Dice: 75.92 %, HSD (sym): 8.51964, Jaccard: 70.73 %, isbi: 70.06 %
study_object.print_results()    # Canny+water (90) Dice: 69.37 %, HSD (sym): 6.65303, Jaccard: 63.75 %, isbi: 59.21 % baaaaaaaaaaaad
#
plt.figure(); plt.imshow(ndi.label(seg_list[0],structure=np.ones((3, 3)))[0]); plt.title("label")
plt.figure(); plt.imshow(seg_list_lab[0]); plt.title("water")
plt.figure(); plt.imshow(gt_list[0]); plt.title("gt")
# Objectives
    # why watershed bad -> see below
    # find out isbi sometime to know what its about
    # in general idk what HSD says and it high or low is good
    # dooo some tracking shit yalla!!!!!!!!!!!! https://www.youtube.com/watch?v=MxK7Fe4xfXM&ab_channel=Enthought


# Maybe later for images with large amount of nuclei to see false split and stuff
study_detection = Study()
study_detection.add_measure(FalseSplit(), 'Split') # compare otsu with canny !!!!!!!!!!!!
study_detection.add_measure(FalseMerge(), 'Merge')
study_detection.add_measure(FalsePositive(), 'FP')
study_detection.add_measure(FalseNegative(), 'FN')
study_detection.reset()
study_detection.set_expected(gt_list[45], unique=True)
study_detection.process(otsu_list_water[45], unique=True) #True when water
study_detection.print_results()
            #   FN    FP  Merge   Split
#canny_list[0]    2     0     2        0
#canny_water[0]   2     0     0        8   #of course FN don't change cause no more nuclei are detected
#canny_list[45]   7     9     7        1
#canny_water[45]  7     12    0        16  #so we get more false detected objects presumable due to more false splits
#canny_list[91]   4     15    15       0
#canny_water[91]  4     19    2        16  # but we remove false merges... apparently false splits are more detrimental for metric
#------------------------------------------
#otsu_seg[0]      15    0     2        0
#otsu_water[0]    15    0     0        2    # way more FN than with canny so canny detects better
#otsu_seg[45]     33    1     8        0    # false merges are similar to canny
#otus_water[45]   33    0     1        5
#otsu_seg[91]     37    4     14       0    # with otsu we have less FPs in higher images. Probably some artifacts we should remove. And wateshed doesn't increase the number
#otsu_water[91]   37    4     5        8

fig, ax = plt.subplots(2,2)
ax[0,0].imshow(gt_list[45],"gray"); ax[0,0].axis("off")
ax[0,1].imshow(seg_list_water[45], "gray"); ax[0,1].axis("off")
ax[1,0].imshow(study_detection.measures["FP"].result, "gray"); ax[1,0].axis("off"); ax[1,0].set_title("FP")
ax[1,1].imshow(study_detection.measures["FN"].result, "gray"); ax[1,1].axis("off"); ax[1,1].set_title("FN")


#Scores for optimal segmentation
ref = np.zeros((500, 500), 'uint8')
ref[100:200, 100:200] = 1
ref[300:400, 100:200] = 2
ref[100:200, 300:400] = 3
ref[300:400, 300:400] = 4

seg = np.zeros((500, 500), 'uint8')
seg[100:200, 100:200] = 1
seg[300:400, 100:200] = 2
seg[100:200, 300:400] = 3
seg[300:400, 300:400] = 4

plt.figure(figsize=(12,4)); plt.subplot(121);plt.imshow(ref, 'gray')
plt.title('ref'); plt.subplot(122); plt.imshow(seg, 'gray'); plt.title('seg')

study_object = Study()
study_object.add_measure(ObjectBasedDistance(ISBIScore()), 'isbi')
study_object.add_measure(ObjectBasedDistance(Hausdorff(mode='sym')), 'HSD (sym)')
study_object.add_measure(ObjectBasedDistance(Dice()), 'Dice')
study_object.add_measure(ObjectBasedDistance(JaccardSimilarityIndex()), 'Jaccard')
study_object.set_expected(ref, unique=True)
study_object.process(seg, unique=True)
study_object.print_results()
study_object.print_results()


    ## Machine Learning Segmentation ##

### Packages
import numpy as np
import random
#
from skimage import io
from skimage.util import view_as_blocks, view_as_windows
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
#
import matplotlib.pyplot as plt

### Functions
#
def create_data_matrix(img, patch_size, patch="blocks",step=1):
    if patch == "blocks":
        data_matrix = np.array([j.flatten() for i in view_as_blocks(img, patch_size) for j in i]) # each i is a row of many blocks so j extracts each block
    elif patch == "windows":
        data_matrix = np.array([j.flatten() for i in view_as_windows(img, patch_size,step=step) for j in i]) #step determines how the windows are spaced
    return data_matrix
#
def create_gt_labels_vector(gt, patch_size, patch="blocks",step=1):
    if patch == "blocks":
        data_matrix = np.array([j.flatten() for i in view_as_blocks(gt, patch_size) for j in i])
    elif patch == "windows":
        data_matrix = np.array([j.flatten() for i in view_as_windows(gt, patch_size, step=step) for j in i])
    labels_vector = np.array([1 if np.sort(i)[int(len(i)/2-1)] == 1 else 0 if i.sum() != 0 else -1 for i in data_matrix]) # if majority (>50%) of block is foreground it receives 1. Any foreground gets 0. No foreground gets -1
    return labels_vector
#
def concat_dms_lvs(dms_list, lvs_list): # lists of dms and lvs you want to have concatenated
    first_dm = dms_list[0]
    first_lv = lvs_list[0]
    dms_conc = first_dm[first_lv != 0]; lvs_conc = first_lv[first_lv != 0] # !subset with lv being not zero, only -1 or 1 !
    for i in range(1,len(dms_list)):
        new_dm = dms_list[i]; new_lv = lvs_list[i]
        dms_conc = np.concatenate((dms_conc, new_dm[new_lv != 0]), axis=0)
        lvs_conc = np.concatenate((lvs_conc, new_lv[new_lv != 0]), axis=0)
    return dms_conc, lvs_conc
#
def predict_image(img, patch_size, clf, patch = "blocks", step=1):
    result = np.zeros(img.shape).astype(bool) #empty bool 2D array
    if patch == "blocks":
        patches = view_as_blocks(result, patch_size)
        dm = create_data_matrix(img, patch_size, patch=patch)  # dm of image as input for the model which throws out a label vector
    elif patch == "windows":
        patches = view_as_windows(result, patch_size, step=step)
        dm = create_data_matrix(img, patch_size, patch=patch, step=step)
    predicted_lv = clf.predict(dm).reshape(patches.shape[0:2]) # we put lv in same shape as dm in order to use next step
    patches[predicted_lv == 1] = True # works since lv and dm now have same shape
    return result
#
def machine_learn_train(train_imgs, train_gts, patch_size, patch="blocks", step=1):
    dms = []
    lvs = []
    for i in range(len(train_imgs)):
        dms.append(create_data_matrix(train_imgs[i], patch_size=patch_size, patch=patch, step=step))
        lvs.append(create_gt_labels_vector(train_gts[i], patch_size=patch_size, patch=patch, step=step))
    return concat_dms_lvs(dms, lvs) # trian_dms, train_lvs = concat_dms_lvs(dms, lvs)
    #
#clf = make_pipeline(StandardScaler(), SVC(class_weight='balanced', gamma=0.1))
#clf.fit(train_dms, train_lvs)
    #
def machine_learn_test(test_imgs, test_gts, clf, patch_size, patch="blocks", step=1):
    # predict and evaluate imgs
    segmented_imgs = []
    dices = []; jaccs = []
    for i in range(len(test_imgs)):
        seg = clear_border(binary_erosion(predict_image(test_imgs[i], clf=clf, patch_size=patch_size, patch=patch, step=step), selem=np.ones((5,5)))) # !! segmented nuclei with blocks or windows tends to be larger than ground truth. Better performance with erosion so it was added afterwards
                                                        # added clear border very late 22.04. After erosion because the size is closer to original and if it doesn't touch border then leave it
        segmented_imgs.append(seg)
        dices.append(evaluate_seg(seg, test_gts[i])[0]); jaccs.append(evaluate_seg(seg, test_gts[i])[1])
    print(f'Mean dice = {np.mean(dices).round(2)}', f'Mean Jaccard = {np.mean(jaccs).round(2)}')
    return segmented_imgs, dices, jaccs

### Create train and test data
train_imgs = []; train_gts = []; test_imgs = []; test_gts = [] # since we don't know at which point training will take too long lets start with only 10 data sets
random_indices = random.sample(range(92),20) # 10 train and 10 test data sets # sample instead of randint because we don't want duplicates
train_ind = random_indices[:len(random_indices)//2]; test_ind = random_indices[len(random_indices)//2:] # !so the number of nuclei increases with time (splitting i guess) which makes segmentation more difficult
for i in range(5): # 10 for blocks, only 5 for windows because time
    img_train = io.imread(rf'01/t0{str(train_ind[i]).zfill(2)}.tif')
    img_train -= img_train.min(); img_train = np.uint8(np.round(img_train/img_train.max()*255))
    train_imgs.append(img_train)
    train_gts.append(io.imread(f'01_ST/SEG/man_seg0{str(train_ind[i]).zfill(2)}.tif') > 0)
    img_test = io.imread(f'01/t0{str(test_ind[i]).zfill(2)}.tif')
    img_test -= img_test.min(); img_test = np.uint8(np.round(img_test / img_test.max() * 255))
    test_imgs.append(img_test)
    test_gts.append(io.imread(f'01_ST/SEG/man_seg0{str(test_ind[i]).zfill(2)}.tif') > 0)

### Test the functions
#
block_size_20 = (20,20)
block_size_10 = (10,10)
block_size_5 = (5,5)
window_size_6 = (6,6) # and we overlap them in half the size meaning step = 3
#
train_dms, train_lvs = machine_learn_train(train_imgs, train_gts, window_size_6, step=3, patch="windows") # if windows: (window_size_6, step=3, patch="windows")
clf = make_pipeline(StandardScaler(), SVC(class_weight='balanced', gamma=0.1))
clf.fit(train_dms, train_lvs) # block_size_20: took ca. 15 seconds with 10 train imgs
                                # block_size_10: took ca 3 min with 10 train imgs
                                # block_size_5: took 30 min
                                # window_size_6 and step=3, only 5 images took 1-2 h
segmented_imgs, dices, jaccs = machine_learn_test(test_imgs, test_gts, clf, window_size_6, step=3, patch="windows")
                                # block_size_20: Mean dice = 0.48 Mean Jaccard = 0.32 --> dices=0.53, jaccs = 0.36 with erosion
                                # block_size_10: Mean dice = 0.69 Mean Jaccard = 0.54, took ca 2 min --> dice=0.76, jaccs=0.62 with erosion
                                # block_size_5: Mean dice = 0.78 Mean Jaccard = 0.66, took ca 30 min, capturing of nuclei very nice but some nuclei missing
                                # window_size_6 and step=3, only 5 images: Mean dice = 0.78 Mean Jaccard = 0.64, took ca 20 min, not real improvement over block_5 but It looks like more nuclei are captured
                                        # --> !! Mean dice = 0.9, Mean Jaccard = 0.81 with binary_erosion(segmentation, selem=np.ones((5,5)))
                                        # --> Erosion was first tested on window_size_6 with step=3 where segmentation is closer to real size. For block_size_20 a greater erosion probably yields better dices/jaccs, but we can get closer anyway with smaller blocks
                                        # --> Clear borders made it slightly worse mean dice=0.86 mean jacc=0.75 but we also used other random images
                                        # todo check why the border objects aren't properly removed in the images. Somehow only outer frame is removed
#


        ## Watershed Algorithm for Segmentation ##
    # I think object-based metric doesn't try to separate objects that overlap in the segmentation and labels them as one object
        #--> so we should compare segmentation and watershedded segmentation as imputs and see what yields better results
### Packages

### Functions
