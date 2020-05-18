import logging
import pandas as pd
import numpy as np
import matplotlib
#matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from roipoly import RoiPoly
import os
import cv2 # in Terminal: conda install opencv
import sys

logging.basicConfig(format='%(levelname)s ''%(processName)-10s : %(asctime)s '
                           '%(module)s.%(funcName)s:%(lineno)s %(message)s',
                    level=logging.INFO)

# Read data
train_folder = "./Train_Set/"
#output_file = "./barrel_rgb.txt"
mask_folder = "./Annotate/"

barrel_pixels = pd.DataFrame(columns = ['image','rgb'])
background_pixels = pd.DataFrame(columns = ['image','rgb'])
barrel_meta = pd.DataFrame(columns = ['image','distance','pixels','avg_rgb','avg_background'])

image_list = os.listdir(train_folder)

for image_name in image_list:
	# Select image
	img_url = train_folder + image_name #'2.3.png'
	img = cv2.imread(img_url)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	# Show the image
	fig = plt.figure()
	plt.imshow(img, interpolation='nearest')
	plt.colorbar()
	plt.title("left click: line segment         right click or double click: close region")
	plt.show(block=False)

	# Let user draw first ROI
	roi1 = RoiPoly(color='g', fig=fig)

	# Show the image with the first ROI
	fig = plt.figure()
	plt.imshow(img, interpolation='nearest')
	plt.colorbar()
	roi1.display_roi()
	plt.title('draw first ROI')
	plt.show(block=False)

	# Show ROI masks
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	mask1 = roi1.get_mask(img_gray)
	plt.imshow(mask1,
	           interpolation='nearest', cmap="Greys")
	plt.title('ROI masks')
	plt.show()

	np.save(mask_folder + image_name[:-4] + '.npy', mask1)

	barrel_rgb = img[mask1]
	background_rgb = img[~mask1]
	mean_rgb = barrel_rgb.mean(0) #mean rgb of barrel
	mean_background = background_rgb.mean(0) #mean rgb of background
	num_pixels = barrel_rgb.size #no. of barrel pixels
	distance = image_name.split('.')[0].split('_')[0] #distance of barrel
	row = round(np.mean(np.nonzero(mask1)[0])) #average row indice of non-zero values in mask
	col = round(np.mean(np.nonzero(mask1)[1])) #average column indice of non-zero values in mask
	centroid = np.array([row,col]) #centroid

	df_row = pd.DataFrame([[image_name, distance, num_pixels, mean_rgb, mean_background]], columns= barrel_meta.columns)
	barrel_meta = barrel_meta.append(df_row)

	# for pixel in barrel_rgb:
	# 	print(pixel)
	# 	df_row = pd.DataFrame([[image_name, pixel]], columns= barrel_pixels.columns)
	# 	barrel_pixels = barrel_pixels.append(df_row)

	# for pixel in background_rgb:
	# 	print(pixel)
	# 	df_row = pd.DataFrame([[image_name, pixel]], columns= background_pixels.columns)
	# 	background_pixels = background_pixels.append(df_row)

#print(barrels)
#barrel_meta.to_csv(index=False, header=True, path_or_buf="/Users/zweedaothaiduy/Dropbox/Document/Study/Cornell Modules/Autonomous ECE5242/Project 1/barrel_meta.csv")
#barrel_pixels.to_csv(index=False, header=True, path_or_buf="/Users/zweedaothaiduy/Dropbox/Document/Study/Cornell Modules/Autonomous ECE5242/Project 1/barrel_pixels.csv")
#background_pixels.to_csv(index=False, header=True, path_or_buf="/Users/zweedaothaiduy/Dropbox/Document/Study/Cornell Modules/Autonomous ECE5242/Project 1/background_pixels.csv")

