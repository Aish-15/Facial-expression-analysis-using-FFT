# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 23:36:13 2021

@author: aishg
"""


import cv2
import dlib
import sys
import matplotlib.pyplot as plt

face_cascade = cv2.CascadeClassifier('C:/Users/aishg/anaconda3/envs/project1/Library/etc/haarcascades/haarcascade_frontalface_default.xml')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:/Users/aishg/anaconda3/envs/project/lib/site-packages/face_recognition_models/models/shape_predictor_81_face_landmarks.dat")
mouth_cascade = cv2.CascadeClassifier('C:/Users/aishg/anaconda3/envs/project1/Library/etc/haarcascades/haarcascade_mcs_mouth.xml')


tmp = cv2.imread('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/happy/happy (1).jpg')
img = cv2.imread('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/happy/happy (1).jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#plt.imshow(tmp)

faces = detector(gray)

for face in faces:
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()


    landmarks = predictor(gray, face)

    x_pts = []
    y_pts = []

    for n in range(49, 65):
         x = landmarks.part(n).x
         y = landmarks.part(n).y
 
         x_pts.append(x)
         y_pts.append(y)
 
         x1 = min(x_pts)
         x2 = max(x_pts)
         y1 = min(y_pts)
         y2 = max(y_pts)
 
         cv2.rectangle(gray, (x1,y1), (x2,y2), (0, 255, 255), 10)
         cv2.rectangle(img, (x1,y1), (x2,y2), (0, 0, 0), -1)
         plt.imshow(img)
         
# =============================================================================
        
        
         for n in range(28, 36):
             x = landmarks.part(n).x
             y = landmarks.part(n).y
            
             x_pts.append(x)
             y_pts.append(y)

             x1 = min(x_pts)
             x2 = max(x_pts)
             y1 = min(y_pts)
             y2 = max(y_pts)

             cv2.rectangle(gray, (x1,y1), (x2,y2), (0, 255, 255), 10)
             cv2.rectangle(img, (x1,y1), (x2,y2), (0, 0, 0), -1)

             plt.imshow(img)
             
             for n in range(1, 16):
                 x = landmarks.part(n).x
                 y = landmarks.part(n).y
                 
                 x_pts.append(x)
                 y_pts.append(y)
                 
                 x1 = min(x_pts)
                 x2 = max(x_pts)
                 y1 = min(y_pts)
                 y2 = max(y_pts)

                 cv2.rectangle(gray, (x1,y1), (x2,y2), (255, 255, 255), 10)
                 #cv2.rectangle(img, (x1,y1), (x2,y2), (0, 255, 255), -1)
                 chin = img[y1:y2, x1:x2]
                 plt.imshow(chin)

                # cv2.imwrite("C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/seperate features/chinH.jpg", chin)


chinH = cv2.imread("C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/seperate features/chinH.jpg")

chinH2 = cv2.cvtColor(chinH, cv2.COLOR_BGR2GRAY)
heightch, widthch = chinH2.shape

chinhH = cv2.resize(chinH2, dsize=(240, 136), interpolation=cv2.INTER_CUBIC)

plt.imshow(chinhH, cmap=plt.get_cmap('gray'))

chinh = cv2.normalize(chinhH.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)*255

heighth, widthh = chinh.shape

plt.imshow(chinh, cmap=plt.get_cmap('gray'))
plt.axis('off')
 
import pandas as pd
import numpy as np

chin_happy = round(((heighth-8)*(widthh-8))/64)
flatchinh = np.zeros((chin_happy, 66), np.uint8)
#flatchinh[:,64] = 5
#flatchinh[:,65] = 1

k = 0
for i in range(0,heighth-8,8):
    for j in range(0,widthh-8,8):
       crop_tmp5 = chinh[i:i+8,j:j+8]
       flatchinh[k,0:64] = crop_tmp5.flatten()
       k = k + 1
fspaceChinh = pd.DataFrame(flatchinh) 

chin2 = fspaceChinh[~np.all(fspaceChinh == 0, axis=1)]
chin2[64]=5
chin2[65]=1

chin2.to_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/fspaceChinH.csv', index=False)



