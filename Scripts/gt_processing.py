#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import cv2
import os
import csv
import matplotlib.pyplot as plt



""" ---------------------------------------------------------------------------------------------- """

input_folder = '../Ground_truth/'

oc_input_folder = input_folder + 'OC_GT/'
od_input_folder = input_folder + 'OD_GT/'

oc_files = os.listdir(oc_input_folder)
oc_files.sort()

od_files = os.listdir(od_input_folder)
od_files.sort()

oc_output_folder = input_folder + '_processed_oc_gt/'
od_output_folder = input_folder + 'processed_od_gt/'

try:
    os.makedirs(oc_output_folder)
except OSError:
    if not os.path.isdir(oc_output_folder):
        raise
        
try:
    os.makedirs(od_output_folder)
except OSError:
    if not os.path.isdir(od_output_folder):
        raise


""" ---------------------------------------------------------------------------------------------- """

liste_centers = []

# Read the csv file containing the coordinates of the found OD location

with open('../OD_detection/' + 'od_coordinates.csv') as csvfile:
    next(csvfile)
    spamreader = csv.reader(csvfile)
    for row in spamreader:
        if row and row.index != 0:
            liste_centers.append((int(row[1]), int(row[2])))
            
            
def process(image, coordinates):
    
    # Crop around the OD coordinates
    image = image[coordinates[0]-150:coordinates[0]+150, coordinates[1]-150:coordinates[1]+150]
    
    # Global thresholding to build the consensus between 3 out of 4 experts (T = 180)
    ret,image = cv2.threshold(image,180,255,cv2.THRESH_BINARY)
    
    return image
    

for i, fichier in enumerate(oc_files):
    
    # Reading the image
    oc_image = cv2.imread(oc_input_folder + fichier, 0)
    
    width, height = oc_image.shape[:2]
    ratio = width/height
    
    # Resize images
    oc_image = cv2.resize(oc_image, (900, int(900*ratio)), interpolation = cv2.INTER_CUBIC)

    # Processing
    processed_oc_image = process(oc_image, liste_centers[i])
    
    cv2.imwrite(oc_output_folder+fichier, processed_oc_image)
    


for i, fichier in enumerate(od_files):
    
    # Reading the image
    od_image = cv2.imread(od_input_folder + fichier, 0)
    
    width, height = oc_image.shape[:2]
    ratio = width/height
    
    # Resize images
    od_image = cv2.resize(od_image, (900, int(900*ratio)), interpolation = cv2.INTER_CUBIC)
    
    # Processing
    processed_od_image = process(od_image, liste_centers[i])
    
    cv2.imwrite(od_output_folder+fichier, processed_od_image)
    


#plt.imshow(processed_image, 'gray')
#plt.show()
            
