#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Script permettant de faire la segmentation du OC et du OD pour le diagnostic de glaucome grâce au CDR

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from scipy.spatial import distance as dis
from sklearn.metrics import confusion_matrix
from skimage.measure import label, regionprops
from skimage.morphology import convex_hull_image
# import xlsxwriter module 
import xlsxwriter 
import math


""" ---------------------------------------------------------------------------------------------- """

input_folder = '../OD_detection/OD_crops/'
fichiers = os.listdir(input_folder)
fichiers.sort()


segmentation_path = '../OC_OD_segmentation/' 

try:
    os.makedirs(segmentation_path)
except OSError:
    if not os.path.isdir(segmentation_path):
        raise

gt_path = 'Ground_truth/'

oc_folder = segmentation_path + gt_path + 'processed_oc_gt/'
oc_list = os.listdir(oc_folder)
oc_list.sort()

od_folder = segmentation_path + gt_path + 'processed_od_gt/'
od_list = os.listdir(od_folder)
od_list.sort()


outpath_oc = segmentation_path + 'oc_results/' 

try:
    os.makedirs(outpath_oc)
except OSError:
    if not os.path.isdir(outpath_oc):
        raise
        
outpath_od = segmentation_path + 'od_results/' 

try:
    os.makedirs(outpath_od)
except OSError:
    if not os.path.isdir(outpath_od):
        raise
        
outpath_oc_hough = segmentation_path + 'oc_hough/'

try:
    os.makedirs(outpath_oc_hough)
except OSError:
    if not os.path.isdir(outpath_oc_hough):
        raise

outpath_od_hough = segmentation_path + 'od_hough/' 

try:
    os.makedirs(outpath_od_hough)
except OSError:
    if not os.path.isdir(outpath_od_hough):
        raise


""" ----------------------------- Segmentation step ------------------------------------- """

# Histogram equalization

def equalization(image):

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(image)
    
    return cl1.astype(np.uint8)


# K-means class

class Segment:
    def __init__(self,segments=4):
        #define number of segments, with default 3
        self.segments=segments
        
    def kmeans(self,image):
       # Etape de prétraitement : application d'un filtre anisotrope
       #image=mfs.anisotropic_diffusion(image, niter=1, kappa=50, gamma=0.1, voxelspacing=None, option=1)
       
       vectorized=image.flatten()
       vectorized=np.float32(vectorized)
       criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
              10, 1.0)
       ret,label,center=cv2.kmeans(vectorized,self.segments,None,
              criteria,10,cv2.KMEANS_RANDOM_CENTERS)
       res = center[label.flatten()]
       segmented_image = res.reshape((image.shape))
       return label.reshape((image.shape[0],image.shape[1])), segmented_image.astype(np.uint8), center
   
    def extractComponent(self,image,label_image,label):
       component=np.zeros(image.shape,np.uint8)
       component[label_image==label]=image[label_image==label]
       return component
   


# Compute the entire optic disc

def disque_optique_entier(od, oc):
    
    disque_optique = cv2.bitwise_or(oc, od)
    
    return disque_optique


# Compute the distance between two points from the plan
def get_distance(x, y, i, j):
    
    distance = dis.euclidean((i,j), (x, y))
    
    return distance


def in_circle(center_x, center_y, radius, x, y):
    
    dist = math.sqrt((center_x - x) ** 2 + (center_y - y) ** 2)
    
    return dist <= radius



def hough(binary_image, minRad, maxRad):
    
    #image_rgb = cv2.cvtColor(binary_image,cv2.COLOR_GRAY2BGR)
    
    edges = canny(binary_image, sigma=5, low_threshold=10, high_threshold=50)
    
    #hough_radii = np.arange(100, 150, 2)
    hough_radii = np.arange(minRad, maxRad, 2)
    hough_res = hough_circle(edges, hough_radii)
    
    # Select the most prominent 5 circles
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                               total_num_peaks=1)
    
    return cx, cy, radii




# Segmentation of the OC and the OD with method 1: K-means + Convex Hull
    
def segmentation(cropped):
    
    #cropped = equalization(cropped)

    # ------------- k-means algorithm --------------------------
    
    seg = Segment()
    
    # Here, we extract a label map, an image of the resulting clusters, and the table of the cluster centers
    # (here, 4 clusters and 4 centers)
    label, result, center = seg.kmeans(cropped)

    # We extract the pixels belonging to the cluster with highest gray-level
    # (belonging to the OC)
    extracted=seg.extractComponent(cropped,label, np.argmax(center))
    
    # Binarisation 
    ret, thresh = cv2.threshold(extracted, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    # Morphological closing 
    kernel = np.ones((3,3),np.uint8)
    #closing_oc = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    opening_oc = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            
    center[np.argmax(center)] = 0
    
    # We extract the pixels belonging to the cluster with the second highest gray-level
    # (belonging to the OD)
    extracted2 = seg.extractComponent(cropped,label, np.argmax(center))
    
    # Binarisation de l'image résultat d'extraction du DO
    ret2, thresh2 = cv2.threshold(extracted2, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    complete_od = disque_optique_entier(thresh2, thresh)
    
    # Morphological operation
    #closing_od = cv2.morphologyEx(complete_od, cv2.MORPH_CLOSE, kernel)
    opening_od = cv2.morphologyEx(complete_od, cv2.MORPH_OPEN, kernel)
    
    
    """ ------------- Convex hull transform -------------------------- """
    
    OC_chull = convex_hull_image(opening_oc)
    #hullPoints = ConvexHull(OC_chull)
    #print(hullPoints)
    OC_chull = OC_chull*255
    
    OD_chull = convex_hull_image(opening_od)
    OD_chull = OD_chull*255
    
    return result, opening_oc, opening_od, OC_chull, OD_chull



# Segmentation of the OC and the OD with method 2: K-means + Hough + Convex Hull
    
def segmentation2(cropped):
    
    #cropped = equalization(cropped)

    """ ------------- Algorithme de k-means -------------------------- """
    
    seg = Segment()
    
    # Here, we extract a label map, an image of the resulting clusters, and the table of the cluster centers
    # (here, 4 clusters and 4 centers)
    label, result, center = seg.kmeans(cropped)

    # We extract the pixels belonging to the cluster with highest gray-level
    # (belonging to the OC)
    extracted=seg.extractComponent(cropped,label, np.argmax(center))
    
    # Binarisation de l'image résultat d'extraction du CUP
    ret, thresh = cv2.threshold(extracted, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    # Fermeture morphologique de l'image binarisée
    kernel = np.ones((3,3),np.uint8)
    #closing_oc = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    opening_oc = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    center[np.argmax(center)] = 0
    
    # We extract the pixels belonging to the cluster with the second highest gray-level
    # (belonging to the OD)
    extracted2 = seg.extractComponent(cropped,label, np.argmax(center))
    
    # Binarisation de l'image résultat d'extraction du DO
    ret2, thresh2 = cv2.threshold(extracted2, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    complete_od = disque_optique_entier(thresh2, thresh)
    
    # Morphological operation
    #closing_od = cv2.morphologyEx(complete_od, cv2.MORPH_CLOSE, kernel)
    opening_od = cv2.morphologyEx(complete_od, cv2.MORPH_OPEN, kernel)
    
    
    """ ------------- Circular Hough Transform -------------------------- """

    # Convert the oc and od images to RGB to plot the detected circles in blue
    oc_rgb = cv2.cvtColor(opening_oc,cv2.COLOR_GRAY2BGR)
    od_rgb = cv2.cvtColor(opening_od,cv2.COLOR_GRAY2BGR)

    # Canny edge detector
    oc_edges = canny(opening_oc, sigma=5, low_threshold=10, high_threshold=50)
    od_edges = canny(opening_od, sigma=5, low_threshold=10, high_threshold=50)
    
    # Range of radius
    oc_hough_radii = np.arange(80, 110, 5)
    od_hough_radii = np.arange(120, 150, 5)
    
    # Circular Hough Transform
    oc_hough_res = hough_circle(oc_edges, oc_hough_radii)
    od_hough_res = hough_circle(od_edges, od_hough_radii)
    
    # Select the most prominent circles in both OC and OD maps: (cx, cy) the circle center and 
    accums1, oc_cx, oc_cy, oc_radius = hough_circle_peaks(oc_hough_res, oc_hough_radii,
                                               total_num_peaks=1)
    accums2, od_cx, od_cy, od_radius = hough_circle_peaks(od_hough_res, od_hough_radii,
                                               total_num_peaks=1)    
    
    # Plot the circle on the images
    (oc_cx, oc_cy, oc_radius) = (int(oc_cx), int(oc_cy), int(oc_radius))
    (od_cx, od_cy, od_radius) = (int(od_cx), int(od_cy), int(od_radius))
    
    cv2.circle(oc_rgb, (oc_cx, oc_cy), oc_radius, (0, 0, 255), 4)
    cv2.circle(od_rgb, (od_cx, od_cy), od_radius, (0, 0, 255), 4)
    
    
    """ ------------- Convex hull transform for the points inside the detected circle -------------------------- """
    
    # Compute the distance between each point in the image and the center of the circle
    oc_distance = [get_distance(oc_cx, od_cy, i, j) for i in range(opening_oc.shape[0]) for j in range(opening_oc.shape[1])]
    oc_distance = np.reshape(oc_distance, opening_oc.shape)

    od_distance = [get_distance(od_cx, od_cy, i, j) for i in range(opening_od.shape[0]) for j in range(opening_od.shape[1])]
    od_distance = np.reshape(od_distance, opening_od.shape)
    
    # Compute the map of the white points inside each circle
    oc_circle = (opening_oc != 0) & (oc_distance < oc_radius)
    oc_circle*255

    od_circle = (opening_od != 0) & (od_distance < od_radius)
    od_circle*255

    # Convex hull of the points in the circle
    OC_chull = convex_hull_image(oc_circle)
    OC_chull = OC_chull*255
    
    OD_chull = convex_hull_image(od_circle)
    OD_chull = OD_chull*255
    
    
    return result, opening_oc, opening_od, oc_rgb, od_rgb, OC_chull, OD_chull




""" -------------------------------- Evaluation of the segmentation results ----------------------------------- """

# Confusion matrix between two images
def matrice_de_confusion(image1, image2):
    
    # Vectorization of the images
    image1 = image1.flatten()
    image2 = image2.flatten()
    
    # Confusion matrix C_i,j = C_0,0 true negatives, C_1,0 false nagatives, C_0,1 false positives, C_1,1 true positives
    # sur l'axe 0 les vrais classes, sur l'axe 1 les classes prédites
    
    matrice = confusion_matrix(image1, image2)

    TN = matrice[0,0]
    FN = matrice[1,0]
    FP = matrice[0,1]
    TP = matrice[1,1]
    
    return TN, FN, FP, TP


# Computation of the metrics depending on the values of the confusion matrix
def metrics(TN, FN, FP, TP):
    
    #accuracy = (TP+TN)/(TN+FN+FP+TP)
    #specificity = TN/(TN+FP)
    sensitivity = TP/(TP+FN+0.000001)
    precision = TP/(TP+FP+0.000001)
    fscore = 2*((precision*sensitivity)/(precision+sensitivity+0.000001))
    
    return sensitivity, precision, fscore
   
    

""" ---------------------------------------------------------------------------------------------- """

results = [['image', 'sensitivity_oc', 'precision_oc', 'fscore_oc', 
            'sensitivity_od', 'precision_od', 'fscore_od']]

for i, fichier in enumerate(fichiers):
            
    outpath_dossier = segmentation_path + 'Images/' + fichier+'/'
    try:
        os.makedirs(outpath_dossier)
    except OSError:
        if not os.path.isdir(outpath_dossier):
            raise
            
    print("Processing sur l'image %d" % (i+1))

    cropped = cv2.imread(input_folder + fichier, 0)
    
    # Segmentation of the OC and the OD with method 1
    #kmeans, OC, OD, OC_chull, OD_chull = segmentation(cropped)

    # Segmentation of the OC and the OD with method 2
    kmeans, OC, OD, oc_rgb, od_rgb, OC_chull, OD_chull = segmentation2(cropped)
    
    # Saving the computed images
    
    cv2.imwrite(outpath_dossier+'1_k_means.jpg', kmeans)
    
    cv2.imwrite(outpath_dossier+'2_closing_cup.png', OC)
    cv2.imwrite(outpath_dossier+'3_closing_od.png', OD)
    
    # For the segmentation using the circular Hough transform
    cv2.imwrite(outpath_oc_hough+fichier, oc_rgb)
    cv2.imwrite(outpath_od_hough+fichier, od_rgb)
    
    cv2.imwrite(outpath_dossier+"4_oc.png", OC_chull)
    cv2.imwrite(outpath_dossier+"5_od.png", OD_chull)
    
    cv2.imwrite(outpath_oc + fichier, OC_chull)
    cv2.imwrite(outpath_od + fichier, OD_chull)
    
    
    # Reading the "ground-truth" results of the OC and OD segmentation
    oc_gt = cv2.imread(oc_folder + oc_list[i], 0)
    od_gt = cv2.imread(od_folder + od_list[i], 0)
    # Evaluation step
    
    TN_oc, FN_oc, FP_oc, TP_oc = matrice_de_confusion(OC_chull, oc_gt)
    sensitivity_oc, precision_oc, fscore_oc = metrics(TN_oc, FN_oc, FP_oc, TP_oc)
    
    TN_od, FN_od, FP_od, TP_od = matrice_de_confusion(OD_chull, od_gt)
    sensitivity_od, precision_od, fscore_od = metrics(TN_od, FN_od, FP_od, TP_od)

    result = [[fichier, sensitivity_oc, precision_oc, fscore_oc, 
          sensitivity_od, precision_od, fscore_od]]
    
    results = np.concatenate((results, result))
    
"""
moy_sensitivity_oc = np.mean(results[1::,1])
moy_precision_oc = np.mean(results[1::,2])
moy_fscore_oc = np.mean(results[1::,3])
moy_sensitivity_od = np.mean(results[1::,4])
moy_precision_od = np.mean(results[1::,5])
moy_fscore_od = np.mean(results[1::,6])
"""

# Write the evaluation results in a csv file
"""
with open(segmentation_path + 'results.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile)
    spamwriter.writerows(results)
"""

workbook = xlsxwriter.Workbook(segmentation_path + 'results_.xlsx') 
  
# By default worksheet names in the spreadsheet will be Sheet1, Sheet2 etc., but we can also specify a name. 
worksheet = workbook.add_worksheet("My sheet") 
  
# Start from the first cell. Rows and columns are zero indexed. 
row = 0
col = 0
  
# Iterate over the data and write it out row by row. 
for image, sensitivity_oc, precision_oc, fscore_oc, sensitivity_od, precision_od, fscore_od in (results): 
    
    worksheet.write(row, col, image) 
    worksheet.write(row, col + 1, sensitivity_oc)
    worksheet.write(row, col + 2, precision_oc)
    worksheet.write(row, col + 3, fscore_oc)
    worksheet.write(row, col + 4, sensitivity_od)
    worksheet.write(row, col + 5, precision_od)
    worksheet.write(row, col + 6, fscore_od)
    
    row += 1
  
workbook.close() 
    
