# DATA EXPLORATION #


## Library ##
import os
import numpy as np
#
import cv2 as cv
#
from skimage import io
from skimage.util import img_as_ubyte
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import binary_opening, binary_closing, binary_erosion
from skimage.segmentation import clear_border
# from skimage.color import rgb2gray
#
import matplotlib.pyplot as plt
#
# import pandas as pd
## Library ##


## Loading Data ##
#We got fluorescence microscopy images with 16 bit on grayscale in TIFF (.tif)
#ground truth (gt) and st (silver truth??) given.
#gt starts at 12 and skips many images while st contains all
#
t000 = io.imread("01/t000.tif"); t091 = io.imread("01/t091.tif") # reads the image in original settings:16 bit grayscale image
t000_norm = t000 - t000.min(); t091_norm = t091 - t091.min()
t000_norm1 = np.uint8(t000/t000.max()*255)
t000_norm2 = np.uint8(np.round(t000/t000.max()*255)) #why is this worse than norm1. in norm1 everything is round down
t000_norm_optimal = t000 - t000.min();  t000_norm_optimal = np.uint8(np.round(t000_norm_optimal/t000_norm_optimal.max()*255)) # first let t000 begin by 0 so you get full range 0-255 in 8 bit image then
t000_cv = cv.imread("01/t000.tif", cv.IMREAD_GRAYSCALE) # converts into 8 bit grayscale by value//256 meaning round DOWN
t000_8bit = img_as_ubyte(t000) # value//256 meaning round DOWN. But why is round down ok?
#gt
t000_gt = io.imread("01_ST/SEG/man_seg000.tif") > 0
# t000_gt_8 = np.uint8(np.round(t000_gt/t000_gt.max() * 255)) > 0 #here, np.round is better because without it we loose one nucleus in binarization
#plot
plt.figure(); plt.imshow(t000,"gray"), plt.colorbar(), plt.title("t000 Original")
plt.figure(); plt.imshow(t000_norm1,"gray"), plt.colorbar(), plt.title("t000 in 8bit manually")
plt.figure(); plt.imshow(t000_norm_optimal,"gray"), plt.colorbar(), plt.title("t000 in 8bit manually but round")
plt.figure(); plt.imshow(t000_cv,"gray"), plt.colorbar(), plt.title("t000 cv.imread")
plt.figure(); plt.imshow(t000_8bit,"gray"), plt.colorbar(), plt.title("t000 img_as_ubyte")
# gt
plt.figure(); plt.imshow(t000_gt, "gray"); plt.title("t000_gt")
plt.figure(); plt.imshow(t000_gt_8, "gray"); plt.title("t000_gt_8 binary") #binary image
## Loading Data ##

## Todo
#
#
##



## Evaluations ##

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
    return f'Mean dice = {np.mean(dices).round(2)}', f'Mean Jaccard = {np.mean(jaccs).round(2)}'



## Thresholding ##

### Dice Coefficient
def compute_dice(H, G): # H: My segmentation, G: Ground truth
#    assert bin1.dtype == numpy.bool
#    assert bin2.dtype == numpy.bool #it's not a boolean when you load it. It has 0 and 1
    dice = 2 * (H * G).sum() / (H.sum() + G.sum())
    return dice

### Determine T visually
plt.figure(figsize = (10,5)); plt.imshow(t000_norm1, "gray"); plt.colorbar()
plt.figure(); plt.hist(t000_norm1.flatten(),15) # most pixels are 240
plt.figure(); plt.hist(t000_norm2.flatten(),15) # most pixels are 241. Just a few at 240.
plt.figure(); plt.hist(t000_norm_optimal.flatten(),256) # ranges actually from 0 to 255
#
plt.figure(); plt.imshow(t000_norm1 >= 241, "gray"); plt.title("norm1 T=241") # best result. 242 is smoother but loss of some nuclei. Dice = 0.825
plt.figure(); plt.imshow(t000_norm2 >= 242, "gray"); plt.title("norm2 T=242") # best result but less nuclei captured as norm1 with 241. Dice = 0.875
plt.figure(); plt.imshow(t000_norm_optimal >= 10, "gray"); plt.title("norm_optimal T=10") # captures most nuclei but only has Dice = 0.783
plt.figure(); plt.imshow(binary_opening(t000_norm_optimal >= 10, selem=np.ones((2,2))), "gray"); plt.title("norm_optimal T=10") # Dice = 0.829. selem default is nice but little bit less gives best result OPTICALLY
plt.figure(); plt.imshow(binary_closing(binary_opening(t000_norm_optimal >= 10), selem=np.ones((10,10))), "gray"); plt.title("norm_optimal T=10 open-close, 10x10") # Dice = 0.812 worse. Connects some nuclei so adds more false positives than it removes
plt.figure(); plt.imshow(binary_erosion(binary_opening(t000_norm_optimal >= 10), selem = np.ones((3,3))), "gray"); plt.title("norm_optimal T=10 open-close, 10x10") # Dice = 0.0.895 best DICEICALLY but we lost one nucleus
#
plt.figure(); plt.imshow(clear_border(t000_norm >= 150), "gray"); plt.title("t000, T=150") # Dice = 0.89, Jacc = 0.80 less nuclei
plt.figure(); plt.imshow(clear_border(binary_erosion(t000_norm >= 65, selem=np.ones((5,5)))), "gray"); plt.title("t000, T=65, 5x5 erosion") # Dice = 0.89, Jacc = 0.81 NO trade-off for more nuclei :)
plt.figure(); plt.imshow(clear_border(t091_norm >= 200), "gray"); plt.title("t091, T=200") # Dice = 0.94 , Jacc = 0.89
plt.figure(); plt.imshow(clear_border(binary_erosion(t091_norm >= 90, selem=np.ones((5,5)))), "gray"); plt.title("t091, T=90, 5x5 erosion") # Dice = 0.91 , Jacc = 0.83, trade-off for more nuclei :(

##### Gauss fitler
t000_gauss = gaussian(t000_norm1,sigma=1,preserve_range=True)
plt.figure(); plt.imshow(t000_gauss, "gray"); plt.title("t000_norm1_gauss")
#
plt.figure(); plt.imshow(t000_gauss>=240.25, "gray"); plt.title("Gauß T=240.5") # Dice = 0.796. Smoother than norm1 with 241. But also range of norm1 is so small that one integer step makes a lot and here i can do smaller steps
# --> Gauss improved capturing by reducing white dots and making faint nuclei more clear. But nuclei in general got fatter

### Otsu's method
# ret, thresh1 = cv2.threshold(t000, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) #cv method
T_otsu_0 = threshold_otsu(t000_norm) # T_otsu = 452. otsu definition says that T is in second group but function description says T is in first group.. works of course but is unconventional
T_otsu_91 = threshold_otsu(t091_norm) # T_otsu = 424
plt.figure(); plt.hist(t000_norm1.flatten(),15); plt.axvline(T_otsu_0, color='r')
plt.figure(); plt.imshow(clear_border(t000_norm > T_otsu_0), "gray"); plt.title(f"t000 Otsu T = {T_otsu_0}") # Dice = 0.77, Jacc = 0.63. seperation of nuclei is good but many nuclei missing
plt.figure(); plt.imshow(clear_border(t091_norm > T_otsu_91), "gray"); plt.title(f"t091 Otsu T = {T_otsu_91}") # Dice = 0.73, Jacc = 0.57 #yes I did the clear_border in the evaluate_seg too!

#load multiple images
list_files_img = os.listdir('01')
image_list = []
for filename in list_files_img[:5]: # 92 files... maybe only do 5 wise or so
    image_list.append(io.imread('01/' + filename))
#
list_files_gt = os.listdir('01_ST/SEG')
gt_list = []
for filename in list_files_gt[:5]:
    gt_list.append(io.imread('01_ST/SEG/'+filename))
# image, histogram with otsu, segmentation, grond truth and dice
fig,ax = plt.subplots(nrows=5,ncols=4) #figsize=(20,5)
#
for i in range(len(image_list)):
    img = image_list[i]; img = np.uint8(img / img.max() * 255)
    gt = gt_list[i]; gt = np.uint8(gt / gt.max() * 255) > 0 # forgot np.round()!!!!! add and compare resulting dices
    T = threshold_otsu(img); img_seg = img > T
    dice = compute_dice(img_seg, gt)
    ax[i, 0].imshow(img,"gray"); ax[i, 0].set_title(f't0{str(i).zfill(2)}'); ax[i, 0].axis("off")
    ax[i, 1].hist(img.flatten(), bins=20); ax[i, 1].set_yticks([]); ax[i, 1].axvline(T, color='r') # no idea why first plot has decimal x axis
    ax[i, 2].imshow(img_seg,"gray"); ax[i, 2].set_title(f'Segmentation otsu = {T}'); ax[i, 2].axis("off")
    ax[i, 3].imshow(gt,"gray"); ax[i, 3].set_title(f'Ground Truth - Dice = {dice}'); ax[i, 3].axis("off") # the gts seem very inconsistent



## Edge Detection Segmentation ##

### Packages
import numpy as np
#
from skimage import io
from skimage.feature import canny
from skimage.morphology import binary_opening, binary_closing, binary_erosion, flood_fill
import matplotlib.pyplot as plt

# https://www.youtube.com/watch?v=cToG83MLkqw&ab_channel=KnowledgeAmplifier
# filter mask = bwareopen(imopen(imfill(imclose(edge(image, "canny"), strel("line",3,0)),"holes"), strel(ones(3,3))),1500)
    # edge("canny") canny operator --> skimage.featue.canny(img, sigma = 1) one by default
# https://scikit-image.org/docs/dev/api/skimage.morphology.html for following morphology operations
    # imclose close gaps in edges if there are any so that an edge encloses an object completely (dilation followed by erosion) --> skimage.morphology.binary_closing(image)
    # imfill fills areas enclosed by edges
    # imopen (erosion followed by dilation) cut off areas that are connected to object of interest via thin channels --> skimage.morphology.binary_opening(image)
    # bwareopen removes objects that contain less than given pixel amount --> cleaned = morphology.remove_small_objects(arr, min_size=2, connectivity = 0) default zero connectivity
                                                                        # --> cleaned = morphology.remove_small_holes(cleaned, min_size=2, connectivity=0)


# translated in python functions
t000_canny = canny(t000, sigma=1, low_threshold=35, high_threshold=75) # not 100% sure what thresholds mean but this gives the best result. 100,200 misses many nuclei and high_thresh below 60 includes background
plt.figure(); plt.imshow(t000_canny, "gray"); plt.title("t000_canny, sigma=1")
#close gaps in nuclei boundaries
t000_closed = binary_closing(t000_canny, selem=np.ones((5,5)))
plt.figure(); plt.imshow(t000_closed, "gray"); plt.title("t000_closed 5,5") # 3,3 is good but 5,5 yields one more nucleus because it closes the necessary gap and it increases dice
#invert image
t000_invert = np.invert(t000_closed); plt.figure(); plt.imshow(t000_invert, "gray"); plt.title("t000_invert")
# flood background of nuclei to get only nuclei inverted and background the same as original
t000_flood = flood_fill(t000_invert.astype("int"), (0, 0), 0, connectivity=1) #conncectivity 1 so that diagonal pixels are not seen as connected
plt.figure(); plt.imshow(t000_flood, "gray"); plt.title("t000 from 0/0 with 0") # !!! What do we do if a nuclei is in the corner????? after flood we should have <10% foreground pixels
# combine t000_closed and t000_flood
t000_filled = t000_closed + t000_flood
t000_fil_open = binary_opening(t000_filled, selem=np.ones((3,3))) #
plt.figure(); plt.imshow(t000_fil_open, "gray"); plt.title("t000_filled") # very nice. size is good with gt and separations of nuclei clear. lost two nuclei though... #opening is better than erosion in terms of dice
evaluate_seg(t000_fil_open,t000_gt) #0.925 very nice , with the last image that has more nuclei its slightly less: 0.89
### Put it in a function

def canny_segmentation(img):
    img_canny = canny(img, sigma=1, low_threshold=35, high_threshold=75)
    img_closed = binary_closing(img_canny, selem=np.ones((3, 3)))
    img_invert = np.invert(img_closed)
    for i in np.arange(0, img.shape[0], 50):
        img_flood = flood_fill(img_invert.astype("int"), (i, 0), 0, connectivity=1)
        if np.sum(img_flood)/(700*1100) < 0.1:
            break #if condition is fulfilled, flood_fill was successful and no nuclei was in the way. Stop the loop then.
    return clear_border(binary_opening(img_closed + img_flood)) # added clear_border

t001 = io.imread("01/t001.tif"); t001 = t001-t001.min() # it doesn't make a difference somehow if it starts at 0 or 30.000. So the thresholds don't refer to intensity
t001_filled = canny_segmentation(t001)
t001_gt = io.imread("01_ST/SEG/man_seg001.tif")
plt.figure(); plt.imshow(t001_filled, "gray"); plt.title("t001_filled") #ok
compute_dice(t001_filled, t001_gt>0) #0.90 but with clear border it is 0.92!!!!!!!!!
#
t000_canny = canny_segmentation(t000_norm); plt.figure(); plt.imshow(t000_canny, "gray"); plt.title("t000 Canny Segmentation")
t091_canny = canny_segmentation(t091_norm); plt.figure(); plt.imshow(t091_canny, "gray"); plt.title("t091 Canny Segmentation")


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
    dms_conc = first_dm[first_lv != 0]; lvs_conc = first_lv[first_lv != 0] # !subset with lv being not zero !
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
        seg = predict_image(test_imgs[i], clf=clf, patch_size=patch_size, patch=patch, step=step)
        segmented_imgs.append(seg)
        dices.append(evaluate_seg(seg, test_gts[i])[0]); jaccs.append(evaluate_seg(seg, test_gts[i])[1])
    print(f'Mean dice = {np.mean(dices).round(2)}', f'Mean Jaccard = {np.mean(jaccs).round(2)}')
    return segmented_imgs, dices, jaccs

### Create train and test data
train_imgs = []; train_gts = []; test_imgs = []; test_gts = [] # since we don't know at which point training will take too long lets start with only 10 data sets
random_indices = random.sample(range(92),20) # 10 train and 10 test data sets # sample instead of randint because we don't want duplicates
train_ind = random_indices[:len(random_indices)//2]; test_ind = random_indices[len(random_indices)//2:] # !so the number of nuclei increases with time (splitting i guess) which makes segmentation more difficult
for i in range(5): # now let's only do 5 when using windows because it takes so long
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
block_size_20 = (20,20) #block size
block_size_10 = (10,10)
block_size_5 = (5,5)
window_size_6 = (6,6) # and we overlap them in half the size meaning step = 3
#
train_dms, train_lvs = machine_learn_train(train_imgs, train_gts, window_size_6, step=3, patch="windows")
clf = make_pipeline(StandardScaler(), SVC(class_weight='balanced', gamma=0.1))
clf.fit(train_dms, train_lvs)
# block_size_20: took ca. 15 seconds with 10 train imgs
# block_size_10: took ca 3 min with 10 train imgs
# block_size_5: took 30 min
# window_size_6 and step=3, only 5 images took 1-2 h
segmented_imgs, dices, jaccs = machine_learn_test(test_imgs, test_gts, clf, window_size_6, step=3, patch="windows")
# block_size_20: Mean dice = 0.48 Mean Jaccard = 0.32
# block_size_10: Mean dice = 0.69 Mean Jaccard = 0.54, took ca 2 min
# block_size_5: Mean dice = 0.78 Mean Jaccard = 0.66, took ca 30 min, capturing of nuclei very nice but some nuclei missing
# window_size_6 and step=3, only 5 images: Mean dice = 0.78 Mean Jaccard = 0.64, took ca 20 min, not real improvement over block_5 but It looks like more nuclei are captured
                                            # Mean dice = 0.9 Mean Jaccard = 0.81 with binary_erosion(segmentation, selem=np.ones((5,5)))
improved_dices =[]; improved_jaccs = []
for i in range(5):
    improved_dices.append(evaluate_seg(binary_erosion(segmented_imgs[i], selem=np.ones((5,5))), test_gts[i])[0])
    improved_jaccs.append(evaluate_seg(binary_erosion(segmented_imgs[i], selem=np.ones((5,5))), test_gts[i])[1])
print(f'Mean dice = {np.mean(improved_dices).round(2)}', f'Mean Jaccard = {np.mean(improved_jaccs).round(2)}')

plt.figure(); plt.imshow(binary_erosion(segmented_imgs[0], selem=np.ones((5,5))), "gray")
plt.figure(); plt.imshow(test_imgs[0], "gray")
plt.figure(); plt.imshow(test_gts[0], "gray")
evaluate_seg(segmented_imgs[0],test_gts[0])
evaluate_seg(binary_erosion(segmented_imgs[0], selem=np.ones((5,5))),test_gts[0])

# lets also check canny on an image with higher nuclei count like 91!!


train_imgs = [t000_norm_optimal, t001_norm_optimal]
train_gts = [t000_gt, io.imread("01_ST/SEG/man_seg001.tif")>0]
train_dms, train_lvs = machine_learn_train(train_imgs, train_gts, block_size_20)
clf = make_pipeline(StandardScaler(), SVC(class_weight='balanced', gamma=0.1))
clf.fit(train_dms, train_lvs)


block_size_20 = (20,20) #block size
window_size_2 = (2,2)

### Testing
#
dm = create_data_matrix(t000_norm_optimal, block_size_20)
lv = create_gt_labels_vector(t000_gt, block_size_20)
clf = make_pipeline(StandardScaler(), SVC(class_weight='balanced', gamma=0.1))
clf.fit(dm[lv!=0], lv[lv!=0])
clf.score(dm[lv!=0], lv[lv!=0])
#
t002 = io.imread("01/t002.tif"); t002_norm_optimal = t002 - t002.min(); t002_norm_optimal = np.uint8(np.round(t002_norm_optimal/t002_norm_optimal.max()*255))
dm1 = create_data_matrix(t002_norm_optimal, block_size_20)
predicted = clf.predict(dm1)
H = predict_image(t002_norm_optimal, clf=clf, patch_size=block_size_20) # found out that i have to normalize my images because they all have different ranges
plt.figure(); plt.imshow(H, "gray") # now it works fine. After normalizing the image by turning it into 8 bit. any other normalization would've worked too
plt.figure(); plt.imshow(t002_norm_optimal, "gray")



## Watershed Algorithm for Segmentation ##

### Packages
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

from skimage.segmentation import watershed, clear_border
from skimage.feature import peak_local_max


### todo
    # implement object based segmentation metric :(

### Example with two overlapping circles
x, y = np.indices((80,80))
x1, y1, x2, y2, x3, y3, x4, y4 = 28, 28, 44, 52, 75, 15, 78, 60
r1, r2, r3, r4 = 16, 16, 10, 5
mask_circle1 = (x - x1)**2 + (y - y1)**2 < r1**2
mask_circle2 = (x - x2)**2 + (y - y2)**2 < r2**2
mask_circle3 = (x - x3)**2 + (y - y3)**2 < r3**2
mask_circle4 = (x - x4)**2 + (y - y4)**2 < r4**2
image = mask_circle1 + mask_circle2 + mask_circle3 + mask_circle4 #try now with an alone standing circle
plt.figure(); plt.imshow(image)
#
cleared = clear_border(image) # buffer size makes imaginary border larger so also objects not touching border directly are removed
plt.figure(); plt.imshow(cleared)
distance = ndi.distance_transform_edt(image) # distance of each foreground pixel to nearest background pixel
coords = peak_local_max(distance, footprint=np.ones((5,5)), labels=image) # footprint is area where only one maximum can be. Same with min_distance = 1
                                                                                # label apparently accepts it if the separate areas are all labeled as 1 as long as they don't touch
# now since second ball is smaller the 3,3 footprint isn't enough because it senses a local max in the overlap
mask = np.zeros(distance.shape, dtype=bool) # image with falses
mask[tuple(coords.T)] = True # add two centroids to label them subsequently
markers, _ = ndi.label(mask) # each maximum receives a label, here only 1 and 2, background = 0
labels = watershed(-distance, markers, mask=image) # negative distance because we want basins instead of peaks, mask determines what pixels are labeled in watershed. It won't make obejcts bigger/smaller than mask objects
                                                                                                            # -> so it won't make nuclei larger than mask
plt.figure(); plt.imshow(labels)

### Conlcusion Watershed
# Watershed requires you to segment the image first. It is useful for post-processing to separate overlapping nuclei, but it is not a segmentation method in itself
# DigitalSreeni used openCV in his video and there somehow you put the original image (even rgb) as input for watershed together with labels.
    # He also did some weird stuff with sure_bg and sure_fg. unknown = sure_bg - sure_fg. sure_fg was labeled (connected components) and used as markers. He labeled unknown area as 0. Background was some other label.
    # So openCV uses a different verison of watershed --> rgb image as input, no distance basins, unknown area filled with zero, labeled objects
# Watershed as method of labeling -> if nuclei overlap we ideally separate them with labels
    # we will only be able to use watershed once we're able to implement the object based assessment


### Test with canny image of t099
dist_img = ndi.distance_transform_edt(t000_fil_open); plt.figure(); plt.imshow(dist_img)
coords_img = peak_local_max(dist_img, footprint=np.ones((10,10)), labels=t000_fil_open)
mask = np.zeros(dist_img.shape, dtype=bool)
mask[tuple(coords_img.T)] = True; plt.figure(); plt.imshow(mask)
markers, _ = ndi.label(mask, structure=np.ones((3,3))); plt.figure(); plt.imshow(markers) # structure makes sure that maxima diagonally connected are taken together as one label, 180 labels
labels = watershed(-dist_img, markers, mask=t000_fil_open); plt.figure(); plt.imshow(labels)
# with otsu it looked kinda shitty.
# with canny the labeling is kinda good
# now try instead of marker pixels, marker areas by thresholding distance img
mask2 = np.zeros(dist_img.shape, dtype=bool)
mask2[dist_img > 0.6*dist_img.max()] = True; plt.figure(); plt.imshow(mask2)
markers2, _ = ndi.label(mask2, structure=np.ones((3,3))); plt.figure(); plt.imshow(markers2) # 96 labels
labels2 = watershed(-dist_img, markers2, mask=t000_fil_open); plt.figure(); plt.imshow(labels2) # way less nuclei but also less noise objects than in method above with marker pixels

#
for image in seg_imgs:
    dist_img = ndi.distance_transform_edt(image)
    coords_img = peak_local_max(dist_img, footprint=np.ones((10, 10)), labels=image)
    mask = np.zeros(dist_img.shape, dtype=bool)
    mask[tuple(coords_img.T)] = True
    markers, _ = ndi.label(mask, structure=np.ones((3, 3)))
    labels = watershed(-dist_img, markers, mask=image)


## Object Bases Metric Exploration ##

### Test Images
# We have one segmented-groundtruth pair (01) as pnd with few nuclei and one pair (02) with many nuclei
# The 01 seg is labeled but overlaps are not separated. 02 seg actually separated overlaps!!!
    #-> in 02 seg should be better because of separated objects i guess
# Seg and gt aren't labeled identically meaning the corresponding object doesn't have the same label
# gt also has border objects removed while seg does not
test_img = io.imread("segmetrics.py-master/testdata/01_result.png")
test_img2_gt = io.imread("segmetrics.py-master/testdata/02_groundtruth.tif")
test_img2 = io.imread("segmetrics.py-master/testdata/02_result.tif")

### Objective
# so it all works fine for me in the test jupyter
    # should be easily applicable, maybe even without the magic shit, maybe with them
# gt and result list should be easily generated by me
# labeling is done by function if unique = false so I can compare this with labeling through watershed
    # also compare the false merge, false split shit with and without watershed
# Maybe I'll have to use less images for the comparison because it took a while for the object-based metric to be computed
"%a is a trail" % (array2d[0])