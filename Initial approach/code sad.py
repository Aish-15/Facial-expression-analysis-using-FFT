# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 03:10:33 2021

@author: aishg
"""

# =============================================================================
# We detect the eyes in the face and extract to a csv file in a sad face
# =============================================================================


import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt


face_cascade = cv2.CascadeClassifier('C:/Users/aishg/anaconda3/envs/project1/Library/etc/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:/Users/aishg/anaconda3/envs/project1/Library/etc/haarcascades/haarcascade_eye.xml')

imgs = cv2.imread('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/Sad/sad.jpg')
gray = cv2.cvtColor(imgs, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)

# =============================================================================
# #Detect the face in the picture
# =============================================================================

for (x,y,w,h) in faces:
    cv2.rectangle(imgs, (x,y), (x+w,y+h), (255,0,0), 2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = imgs[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
   
    
# =============================================================================
# #Detect the eyes after the face has been detected
# =============================================================================

    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(imgs, (ex,ey), (ex+ew, ey+eh), (0,255,0), 2) #draws a rectangle over the detected face
        sad = imgs[ey:ey+eh, ex:ex+ew]
        plt.imshow(sad, cmap=plt.get_cmap('gray'))
        
        
# =============================================================================
# Extract the eyes from the picture and save as another image 
# =============================================================================
        
        #face_file_name = "face/" + str(y) + ".jpg"
       # cv2.imwrite("C:/Users/aishg/Downloads/eye.jpg", sub_face)
#plt.imshow(sub_face)


eyes = cv2.imread("C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/seperate features/eyesmile.jpg")

eye2 = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
heightcG, widthcG = eye2.shape

eye = cv2.resize(eye2, dsize=(72, 72), interpolation=cv2.INTER_CUBIC)

#plt.imshow(eye, cmap=plt.get_cmap('gray'))

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
#plt.imshow(tmpc, cmap=plt.get_cmap('gray'))

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

fspaceEye.to_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspaceAEyesH.csv', index=False)


# ============================================================================= 
# Code for nose identification
# =============================================================================
nose_cascade = cv2.CascadeClassifier('/Users/aishg/anaconda3/envs/project1/Library/etc/haarcascades/haarcascade_mcs_nose.xml')
#nose classifier: https://sourceforge.net/p/emgucv/opencv/ci/d58cd9851fdb592a395488fc5721d11c7436a99c/tree/data/haarcascades/

for (x,y,w,h) in faces:
  #  cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    noseh =  nose_cascade.detectMultiScale(gray, 1.3, 5)


for (nx, ny, nw, nh) in noseh:
    #    cv2.rectangle(img, (nx, ny), (nx + nw, ny + nh), (0, 0, 255), 2)
        nose_faceh = img[ny:ny+nh, nx:nx+nw]
  #      face_file_name = "face/" + str(y) + ".jpg"
   #     cv2.imwrite("C:/Users/aishg/Downloads/nose.jpg", nose_face)
plt.imshow(nose_faceh)

noseh = cv2.imread("C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/seperate features/nosesmile.jpg")

noseh2 = cv2.cvtColor(noseh, cv2.COLOR_BGR2GRAY)
heightcG, widthcG = noseh2.shape

noseh = cv2.resize(noseh2, dsize=(72, 72), interpolation=cv2.INTER_CUBIC)

plt.imshow(noseh, cmap=plt.get_cmap('gray'))

noseh = cv2.normalize(noseh.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)*255

heightc, widthc = noseh.shape

plt.imshow(noseh, cmap=plt.get_cmap('gray'))
plt.axis('off')

tmpc = np.zeros((heightc, widthc), np.uint8)
th1 = noseh.mean()
for i in range(heightc):
    for j in range(widthc):
        if(noseh[i][j]<th1):
            tmpc[i][j] = 0
        else:
            tmpc[i][j] = 255
plt.imshow(tmpc, cmap=plt.get_cmap('gray'))

newnoseh = np.zeros((heightc, widthc), np.uint8)
for i in range(heightc-8):
    for j in range(widthc-8):
        crop_tmp = noseh[i:i+8,j:j+8]
        mn = crop_tmp.min()
        mx = crop_tmp.max()
        newnoseh[i:i+8,j:j+8] = ((crop_tmp-mn)/(mx-mn)*255)
plt.imshow(newnoseh, cmap=plt.get_cmap('gray'))

nose_happy = round(((heightc-8)*(widthc-8))/64)
flatnoseh = np.zeros((nose_happy, 66), np.uint8)+2
flatnoseh[:,64] = 1
flatnoseh[:,65] = 1

k = 0
for i in range(0,heightc-8,8):
    for j in range(0,widthc-8,8):
        crop_tmp1 = noseh[i:i+8,j:j+8]
        flatnoseh[k,0:64] = crop_tmp1.flatten()
        k = k + 1
fspaceNoseh = pd.DataFrame(flatnoseh)  

fspaceNoseh.to_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspaceANoseH.csv', index=False)


# =============================================================================
# Code For MOUTH - HAPPY
# =============================================================================

mouth_cascade = cv2.CascadeClassifier('/Users/aishg/anaconda3/envs/project1/Library/etc/haarcascades//haarcascade_smile.xml')

for (x,y,w,h) in faces:
  #  cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    lips = mouth_cascade.detectMultiScale(gray, 1.35, 9)

    for (mx, my, mw, mh) in lips:
     #   cv2.rectangle(img, (mx, my), (mx + mw, my + mh), (60, 20, 60), 2)
        mouth_face1H = img[my:my+mh, mx:mx+mw]
        face_file_name = "face/" + str(y) + ".jpg"
        cv2.imwrite("C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/seperate features/mouthfaceH.jpg", mouth_face1H)
plt.imshow(mouth_face1H)
    

mouthH = cv2.imread("C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/seperate features/mouthfaceH.jpg")

mouthH2 = cv2.cvtColor(mouthH, cv2.COLOR_BGR2GRAY)
heightcm, widthcm = mouthH2.shape

mouthh = cv2.resize(mouthH2, dsize=(80, 40), interpolation=cv2.INTER_CUBIC)

plt.imshow(mouthh, cmap=plt.get_cmap('gray'))

mouthh = cv2.normalize(mouthh.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)*255

heightm, widthm = mouthh.shape

plt.imshow(mouthh, cmap=plt.get_cmap('gray'))
plt.axis('off')

tmpm = np.zeros((heightm, widthm), np.uint8)
th3 = mouthh.mean()
for i in range(heightm):
    for j in range(widthm):
        if(mouthh[i][j]<th1):
            tmpm[i][j] = 0
        else:
            tmpm[i][j] = 255
plt.imshow(tmpm, cmap=plt.get_cmap('gray'))

newmouthh = np.zeros((heightm, widthm), np.uint8)
for i in range(heightm-8):
    for j in range(widthm-8):
        crop_tmp3 = mouthh[i:i+8,j:j+8]
        mn = crop_tmp3.min()
        mx = crop_tmp3.max()
        newmouthh[i:i+8,j:j+8] = ((crop_tmp3-mn)/(mx-mn)*255)
plt.imshow(newmouthh, cmap=plt.get_cmap('gray'))

mouth_happy = round(((heightm-8)*(widthm-8))/64)
flatmouthh = np.zeros((mouth_happy, 66), np.uint8)+2
flatmouthh[:,64] = 2
flatmouthh[:,65] = 1

k = 0
for i in range(0,heightm-8,8):
    for j in range(0,widthm-8,8):
        crop_tmp3 = mouthh[i:i+8,j:j+8]
        flatmouthh[k,0:64] = crop_tmp3.flatten()
        k = k + 1
fspaceMouthh = pd.DataFrame(flatmouthh)  

fspaceMouthh.to_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspaceMouthH.csv', index=False)
