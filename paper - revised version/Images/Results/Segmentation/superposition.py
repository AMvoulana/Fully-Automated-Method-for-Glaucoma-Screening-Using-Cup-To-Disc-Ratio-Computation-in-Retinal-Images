# -*- coding: utf-8 -*-


import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

# load the image
#dossier = '../Output/DRISHTI-GS1/drishtiGS_047.png/'
dossier = 'drishti101/'

image_cup = cv2.imread(dossier + "6_hough_cup.png")
image_do = cv2.imread(dossier + "7_hough_do.png")
image_do = cv2.resize(image_do,(600,600))

image_cup_gt = cv2.imread(dossier + "4_cup_gt.png")
image_do_gt = cv2.imread(dossier + "5_do_gt.png")
image_do_gt = cv2.resize(image_do_gt,(600,600))


#image_cup = np.zeros((image_cup.shape), np.uint8)
#cv2.circle(image_cup, (370, 280), 100, (255,255,255), -1)
#cv2.circle(image_cup, (390, 340), 120, (255,255,255), -1) image 42
#cv2.circle(image_do, (200, 350), 205, (255,255,255), -1) image 63

cv2.imwrite(dossier + "6_hough_cup.png", image_cup)
"""
plt.imshow(image_do)
plt.show()
"""

image_cup[np.where((image_cup == [255,255,255]).all(axis = 2))] = [127,127,127]
#new_image_cup[np.where((new_image_cup == [255,255,255]).all(axis = 2))] = [127,127,127]
image_cup_gt[np.where((image_cup_gt == [255,255,255]).all(axis = 2))] = [127,127,127]

# Fonction faisant le calcul de l'aire d'une image binaire
def calcul_aire(binary_image):
    
    binary_image = binary_image[binary_image != 0]
    aire = len(binary_image)
    
    return aire

aire_cup, aire_do = calcul_aire(image_cup), calcul_aire(image_do)
aire_cup_gt, aire_do_gt = calcul_aire(image_cup_gt), calcul_aire(image_do_gt)

ratio = aire_cup/aire_do
ratio_gt = aire_cup_gt/aire_do_gt

overlay = image_cup.copy()
output = image_do.copy()

overlay_gt = image_cup_gt.copy()
output_gt = image_do_gt.copy()

alpha = 0.6

print('1')

cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
cv2.addWeighted(overlay_gt, alpha, output_gt, 1 - alpha, 0, output_gt)

plt.imshow(output)
plt.show()

cv2.imwrite(dossier + "overlay.png", output)
cv2.imwrite(dossier + "overlay_gt.png", output_gt)