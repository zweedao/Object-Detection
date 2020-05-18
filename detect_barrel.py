#Import packages
import pandas as pd
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as img
import matplotlib.cm as cm
from mpl_toolkits import mplot3d
import imageio as imo
import cv2 # in Terminal: conda install opencv
import os
from roipoly import RoiPoly #draw polygon on image to label object 
import skimage
from skimage import feature
from skimage import filters
from skimage.measure import label, regionprops
from skimage.filters import roberts, sobel, threshold_otsu
from sklearn.mixture import GaussianMixture
from sklearn import svm, datasets
from joblib import dump, load


#Prediction
model_folder = "./Model/"
test_folder = "./Test_Set/"
result_folder = "./Result/"

def detect_barrel(image_name, image_test):
    x_test = image_test.reshape(-1,3)
    
    gmm_barrel = load(model_folder + 'gmm_barrel.joblib')
    gmm_other = load(model_folder + 'gmm_other.joblib')
    log_prob_barrel = gmm_barrel.score_samples(x_test)
    log_prob_other = gmm_other.score_samples(x_test)
    
    mask_test = (log_prob_barrel > log_prob_other)
    mask_test = mask_test.reshape(900,1200)
    #plt.imshow(mask_test,interpolation='nearest', cmap="Greys")
    
    label_img = label(mask_test)
    #plt.imshow(label_img)
    regions = regionprops(label_img)
    
    fig, ax = plt.subplots()
    ax.imshow(image_test)
    
    for region in regions:
        if region.area >= 1000:
            y0, x0 = region.centroid
            orientation = region.orientation
            x1 = x0 + math.cos(orientation) * 0.5 * region.major_axis_length
            y1 = y0 - math.sin(orientation) * 0.5 * region.major_axis_length
            x2 = x0 - math.sin(orientation) * 0.5 * region.minor_axis_length
            y2 = y0 - math.cos(orientation) * 0.5 * region.minor_axis_length
            
            minr, minc, maxr, maxc = region.bbox
            bx = (minc, maxc, maxc, minc, minc)
            by = (minr, minr, maxr, maxr, minr)
            
            height = maxr - minr
            width = maxc - minc
            side_ratio = height/width
            axis_ratio = region.major_axis_length / region.minor_axis_length
            if region.solidity > 0.7 and ((side_ratio>1.15 and side_ratio<2.6) or (axis_ratio>1.3 and axis_ratio<1.6)): #
                print('label',region.label)
                print('height', height)
                print('side_ratio', side_ratio)
                print('axis_ratio', axis_ratio)
                print('extent',region.extent)
                print('filled area',region.filled_area)
                print('solidity', region.solidity)
                ax.plot(bx, by, '-b', linewidth=2.5)
                ax.plot(x0,y0,'.b', markersize=7)
                plt.text(x0-50, y0+100, 'X:'+ str(round(x0)) +'\nY:' + str(round(y0)), fontsize=9, color='green')
                #ax.plot((x0, x1), (y0, y1), '-g', linewidth=2.5)
                #ax.plot((x0, x2), (y0, y2), '-g', linewidth=2.5)
                distance = round(650 / height)
                plt.text(x0-70, y0-100, 'Distance: '+ str(distance), fontsize=9, color='green')
    
    print('distance',distance)
    plt.title('Image: ' + image_name
        +'\nBarrel distance: ' + str(distance) 
        +'\nCentroid X: ' + str(round(x0)) 
        + '; Centroid Y: ' + str(round(y0)))
    plt.savefig(result_folder + image_name_test, dpi=300)
    plt.show() 
    return

image_test_list = os.listdir(test_folder)

for image_name_test in image_test_list:
    print(image_name_test)
    if image_name_test[0] != '.':
        #image_name_test = '001.png'#'3.2.png' #'5.1.png'
        image_url_test = test_folder + image_name_test
        image_test = cv2.imread(image_url_test)
        image_test = cv2.cvtColor(image_test, cv2.COLOR_BGR2RGB)
        #plt.imshow(image_test)
        
        detect_barrel(image_name_test, image_test)
