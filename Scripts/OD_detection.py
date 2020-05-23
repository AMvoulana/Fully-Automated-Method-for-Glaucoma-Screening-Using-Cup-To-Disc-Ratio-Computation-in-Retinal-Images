#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Script permettant de faire la détection du disque optique grâce au calcul des points brillants et un algorithme de template matching


import numpy as np
import cv2
import os
#import matplotlib.pyplot as plt
#from skimage import exposure
from scipy import ndimage

from skimage.measure import label, regionprops
import scipy.ndimage.measurements as snm

# POur le calcul de la distance euclidienne
from scipy.spatial import distance as dis

# Pour la lecture et l'écriture de fichiers .csv
import csv




""" ---------------------------------------------------------------------------------------------- """

#entree = '../Input/'

input_folder = '../Images/'
fichiers = os.listdir(input_folder)
fichiers.sort()

        
outpath = '../OD_detection/' 

try:
    os.makedirs(outpath)
except OSError:
    if not os.path.isdir(outpath):
        raise


crop_outpath = outpath + 'OD_crops/'

try:
    os.makedirs(crop_outpath)
except OSError:
    if not os.path.isdir(crop_outpath):
        raise
        

""" ---------------------------------------------------------------------------------------------- """


def equalization(image):

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(image)
    cl1 = clahe.apply(cl1)
    
    return cl1.astype(np.uint8)

    
def binarization(image):
    #seuil, image_b = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    #seuil, image_b = cv2.threshold(image,200,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    seuil, image_b = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    if seuil < 82:
        seuil = seuil + 40
        seuil_max = 15
    else:
        seuil = seuil + 60
        seuil_max = 10
    print(seuil)
    seuil, image_b = cv2.threshold(image,seuil,255,cv2.THRESH_BINARY)
    
    return seuil, seuil_max, image_b



""" --------------------------- Step I : Détection des points candidats --------------------------- """    


def bright_points(image):
        
    cv2.imwrite(image_outpath+'0_image.jpg', image)
    
    # Extraction du canal G de l'image
    im = image[:,:,1]
    im_eq = equalization(im)
    
    #plt.imshow(im)
    #plt.show()

    cv2.imwrite(image_outpath+'1_image_egalisee.jpg', im_eq)
    
    # Binarisation de l'image par la méthode d'Otsu
    seuil, seuil_max, image_b = binarization(im_eq)
    cv2.imwrite(image_outpath+'2_image_seuillee.png', image_b)
    
    """
    if base == 'DRIVE/':
    
        # Opération permettant de ne pas considérer le bord dans le calcul de la carte de distances
        centre = (292,282)
        R = 230
        
        for j in range(0, image.shape[0]):
            for i in range(0, image.shape[1]):
                dist = dis.euclidean((i,j), (centre[0], centre[1]))
                if dist > R:
                    image_b[j,i] = 0
    """                               
                    
    # Calcul de la carte de distance euclidienne
    
    dist = ndimage.distance_transform_edt(image_b)
    
    dist[0:200, :] = 0
    dist[:, 0:200] = 0
    dist[dist.shape[0]-200:dist.shape[0], :] = 0
    dist[:, dist.shape[1]-200:dist.shape[1]] = 0
            
    cv2.imwrite(image_outpath + '3_distance_map.jpg', dist)
    #plt.imshow(dist, cmap=plt.cm.gray)
    
           
    # Calcul des maxima de la carte de distances 
    maximum, index = np.amax(dist), np.unravel_index(np.argmax(dist, axis=None), dist.shape)
    
    xd = index[0]
    yd = index[1]
        
    # Affichage du point le plus "brillant" sur l'image d'origine        
    #cv2.circle(image, (yd, xd), 2, (0, 0, 255), thickness=4)
    
    # Carte contenant les maximas    
    #maxima = dist > maximum - 10
    maxima = dist > maximum - seuil_max
    #maxima = dist > moyenne+ecart
    
    image1 = im_eq.copy()
    
    # Image faisant apparaitre les zones maximales 
    for i in range(0, maxima.shape[0]):
        for j in range(0, maxima.shape[1]):
            if maxima[i][j] == True:
                image1[i][j] = 0
    
    #plt.imshow(image1, cmap=plt.cm.gray)
    cv2.imwrite(image_outpath + '4_max_dist_map.jpg', image1)
    
    maxima = maxima*1
    
    # Etiquetage des zones maximales selon leur connexité
    label_points = label(maxima)
    
    # Liste des centres des zones maximales de l'image
    centres = []
    
    for region in regionprops(label_points):
        
        # On ne considère uniquement les zones supérieures à une certaine aire
        if region.area > 200:
            
            i = region.label
        
            centre = snm.center_of_mass(maxima, label_points, i)
            centre = (int(centre[0]), int(centre[1]))
            
            centres.append(centre)
        
    #liste_centres.append(centres)
    
    # Images faisant apparaitre les centres candidats
    
    image2 = im_eq.copy()
        
    for point in centres:
        cv2.circle(image2, (point[1], point[0]), 5, (0, 0, 255), thickness=4)
        
    cv2.imwrite(image_outpath + '5_bright_points.jpg', image2)
    
    return centres
    

    
""" ---------------------- Step II : Détection du disque optique via template matching sur les points brillants détectés --------------------------- """

# A partir de la liste des centres de chaque image, le but est de trouver le centre du disque pour
# chaque image en utilisant le critère de Template Matching

def template_matching(image,centres):
    
    # Application d'un filtre gaussien
    median = cv2.medianBlur(image, 5)

    # Pour chaque point candidat dans une image, on construit une fenêtre centrée en ce point et on 
    # calcule l'histogramme
    
    #similarites = []
    #correlations = []
    c = 0
    
    # Lecture des histogrammes de référence associés à la base générée grâce au 
    # script 'template.py'
    
    hist_ref_r = []
    hist_ref_g = []
    hist_ref_b = []
    
    input_template = '../Template/Output/'
    
    with open(input_template + 'hist_ref_r.csv') as csvfile:
        reader = csv.reader(csvfile)
        for d in reader : 
            hist_ref_r.append(int(d[0]))

    with open(input_template + 'hist_ref_g.csv') as csvfile:
        reader = csv.reader(csvfile)
        for d in reader : 
            hist_ref_g.append(int(d[0]))

    with open(input_template + 'hist_ref_b.csv') as csvfile:
        reader = csv.reader(csvfile)
        for d in reader : 
            hist_ref_b.append(int(d[0]))
    
    
    # Pour chaque centre de chaque image, on compare l'histogramme de sa fenêtre avec 
    # l'histogramme de référence
    
    #for tuplet in liste_centres[i]:
    for tuplet in centres:

        fenetre = median[tuplet[0]-120:tuplet[0]+120, tuplet[1]-120:tuplet[1]+120]

        # Calcul de l'histogramme de chaque canal de la fenêtre courante
        r = fenetre[:,:,2]
        g = fenetre[:,:,1]
        b = fenetre[:,:,0]
        
        hist_r,bins_r = np.histogram(r,256,[0,256])
        hist_g,bins_g = np.histogram(g,256,[0,256])
        hist_b,bins_b = np.histogram(b,256,[0,256])
        
        # Nous considérons uniquement les valeurs inférieures à 200
        
        hist_r = hist_r[0:200]
        hist_g = hist_g[0:200]
        hist_b = hist_b[0:200]
        
        # Calcul de la similarité et de la corrélation entre les deux histogrammes
        # pour chacun des canaux
        
        simi_r = (hist_r-hist_ref_r)**2
        simi_r = sum(simi_r)
        
        simi_g = (hist_g-hist_ref_g)**2
        simi_g = sum(simi_g)

        simi_b = (hist_b-hist_ref_b)**2
        simi_b = sum(simi_b)
        
        #similarites.append(simi)
        
        # Correlation entre les deux histogrammes du canal R
        c_r = 1/(1+simi_r)
        
        # Correlation entre les deux histogrammes du canal G
        c_g = 1/(1+simi_g)

        # Correlation entre les deux histogrammes du canal B
        c_b = 1/(1+simi_b)        
        
        # Paramètres
        t_r, t_g, t_b = 0.5, 2.0, 1.0
        
        # Corrélation finale
        corr = t_r*c_r + t_g*c_g + t_b*c_b
                
        if corr > c:
            centre_disque = tuplet
            #print(centre_disque)
            c = corr

    # Affichage du point detecté sur l'image
    image_cross = image.copy()
    
    cv2.line(image_cross, (centre_disque[1]-10, centre_disque[0]-10), (centre_disque[1]+10, centre_disque[0]+10), (255,0,0), 2)
    cv2.line(image_cross, (centre_disque[1]+10, centre_disque[0]-10), (centre_disque[1]-10, centre_disque[0]+10), (255,0,0), 2)
    
    # Enregistrement des résultats
    cv2.imwrite(image_outpath+'6_Od_point.jpg', image_cross)
    
    return centre_disque


""" ---------------------------------------------------------------------------------------------- """

#image_name = 'drishtiGS_063.png'

for i, fichier in enumerate(fichiers):

    image = cv2.imread(input_folder + fichier)
    
    image_outpath = '../OD_detection/' + fichier + '/'
    
    try:
        os.makedirs(image_outpath)
    except OSError:
        if not os.path.isdir(image_outpath):
            raise
    
    width, height = image.shape[:2]
    ratio = width/height
    
    #if width > 900 :
    image = cv2.resize(image, (900, int(900*ratio)), interpolation = cv2.INTER_CUBIC)
            
    print("Début du traitement sur l'image choisie")
    liste_centres = bright_points(image)
    print("Liste des centres calculée, début du template matching")
    centre_disque = template_matching(image, liste_centres)
    print("Image traitée")
    
    cropped = image[centre_disque[0]-150:centre_disque[0]+150, centre_disque[1]-150:centre_disque[1]+150]
    
    cv2.imwrite(image_outpath+'7_cropped.jpg', cropped)
    cv2.imwrite(crop_outpath+fichier, cropped)
    
    with open(outpath + 'od_coordinates.csv', 'a') as csvfile:
        writer = csv.writer(csvfile)
        if i == 0:
            writer.writerow(['Image name', 'x coordinate', 'y coordinate'])
        writer.writerow([fichier, centre_disque[0], centre_disque[1]])
       
        
    
    
    
    
    
    