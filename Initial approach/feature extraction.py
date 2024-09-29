# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 02:16:07 2021

@author: aishg
"""

import pandas as pd
import cv2
import matplotlib.pyplot as plt
import numpy as np

eyes = cv2.imread("C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/seperate features/eyesmile.jpg")
nose = cv2.imread("C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/seperate features/nosesmile.jpg")
mouth = cv2.imread("C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/seperate features/mouthsmile.jpg")
head = cv2.imread("C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/seperate features/forehead.jpg")


eye2 = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
heightcG, widthcG = eye2.shape

nose2 = cv2.cvtColor(nose, cv2.COLOR_BGR2GRAY)
heightcG, widthcG = nose2.shape


eye = cv2.resize(eye2, dsize=(72, 72), interpolation=cv2.INTER_CUBIC)

plt.imshow(eye, cmap=plt.get_cmap('gray'))

eye = cv2.normalize(eye.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)*255

heightc, widthc = eye.shape

plt.imshow(eye, cmap=plt.get_cmap('gray'))
plt.axis('off')

tmpc = np.zeros((heightc, widthc), np.uint8)
th1 = eye.mean()
for i in range(heightc):
    for j in range(widthc):
        if(eye[i][j]<th1):
            tmpc[i][j] = 0
        else:
            tmpc[i][j] = 255
plt.imshow(tmpc, cmap=plt.get_cmap('gray'))

neweye = np.zeros((heightc, widthc), np.uint8)
for i in range(heightc-8):
    for j in range(widthc-8):
        crop_tmp = eye[i:i+8,j:j+8]
        mn = crop_tmp.min()
        mx = crop_tmp.max()
        neweye[i:i+8,j:j+8] = ((crop_tmp-mn)/(mx-mn)*255)
plt.imshow(neweye, cmap=plt.get_cmap('gray'))

eye_happy = round(((heightc-8)*(widthc-8))/64)
flateye = np.zeros((eye_happy, 66), np.uint8)+2
flateye[:,65] = 1
flateye[:,64] = 0

k = 0
for i in range(0,heightc-8,8):
    for j in range(0,widthc-8,8):
        crop_tmp1 = eye[i:i+8,j:j+8]
        flateye[k,0:64] = crop_tmp1.flatten()
        k = k + 1
fspaceEye = pd.DataFrame(flateye)  

fspaceEye.to_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspaceaish1happy4.csv', index=False)

