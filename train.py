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


#Read data
## load roipoly mask
image_folder = "./Train_Set/"
mask_folder = "./Annotate/"
model_folder = "./Model/"

x_train_barrel = np.zeros((1,3))
x_train_other = np.zeros((1,3))

mask_list = os.listdir(mask_folder)
for mask_name in mask_list:
    if mask_name[0] != '.':
        print(mask_name)
        mask_url = mask_folder + mask_name
        mask = np.load(mask_url, allow_pickle=True)
        
        image_name = mask_name[:-4] + '.png'
        image_url = image_folder + image_name
        image = cv2.imread(image_url)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        x_train_barrel = np.concatenate((x_train_barrel, image[mask]))
        x_train_other = np.concatenate((x_train_barrel, image[~mask]))
        
x_train_barrel = np.delete(x_train_barrel, (0), axis=0)
x_train_other = np.delete(x_train_other, (0), axis=0)


#Train model
#x_train_barrel = image[mask]
gmm_barrel = gmm = GaussianMixture(n_components=1)
gmm_barrel.fit(x_train_barrel)
dump(gmm_barrel, model_folder + 'gmm_barrel.joblib')

#x_train_other = image[~mask]
gmm_other = gmm = GaussianMixture(n_components=1)
gmm_other.fit(x_train_other)
dump(gmm_other, model_folder + 'gmm_other.joblib')