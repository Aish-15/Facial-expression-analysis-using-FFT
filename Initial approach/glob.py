# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 21:02:30 2021

@author: aishg
"""
# =============================================================================
# 
# import os
# import pandas as pd
# from glob import glob
# 
# os.chdir('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace')
# files = glob('*.csv')
# 
# newFiles = ['C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace' + f for f in files]
# 
# dataframes = [pd.read_csv(f) for f in newFiles]
# =============================================================================
import matplotlib.pyplot as plt
import cv2 
import os 
import glob 

import pandas as pd
import numpy as np


face_cascade = cv2.CascadeClassifier('C:/Users/aishg/anaconda3/envs/project1/Library/etc/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:/Users/aishg/anaconda3/envs/project1/Library/etc/haarcascades/haarcascade_eye.xml')

img_dir = "C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/happy" # Enter Directory of all images  
data_path = os.path.join(img_dir,'*jpg') 
files = glob.glob(data_path) 
data = [] 
for f1 in files: 
    img = cv2.imread(f1) 
    data.append(img)
    plt.imshow(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(gray, 1.2, 10)
        eyer = eye_cascade.detectMultiScale(gray, 1.5, 10)
        for (ex,ey,ew,eh) in eyes:
            sub_face = img[ey:ey+eh, ex:ex+ew]
           # plt.imshow(sub_face, cmap=plt.get_cmap('gray'))
      #  face_file_name = "face/" + str(y) + ".jpg"
       # cv2.imwrite("C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/seperate features/eyeleftfaceH.jpg", sub_face)
            plt.imshow(sub_face)




# =============================================================================
# We detect the eyes in the face and extract to a csv file in a HAPPY face
# =============================================================================


import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt


face_cascade = cv2.CascadeClassifier('C:/Users/aishg/anaconda3/envs/project1/Library/etc/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:/Users/aishg/anaconda3/envs/project1/Library/etc/haarcascades/haarcascade_eye.xml')

img = cv2.imread(img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)

# =============================================================================
# #Detect the face in the picture
# =============================================================================

for (x,y,w,h) in faces:
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(gray, 1.2, 10)
    eyer = eye_cascade.detectMultiScale(gray, 1.5, 10)
    
    
# =============================================================================
# #Detect the eyes after the face has been detected
# =============================================================================

    for (ex,ey,ew,eh) in eyes:
        sub_face = img[ey:ey+eh, ex:ex+ew]
        plt.imshow(sub_face, cmap=plt.get_cmap('gray'))
      #  face_file_name = "face/" + str(y) + ".jpg"
       # cv2.imwrite("C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/seperate features/eyeleftfaceH.jpg", sub_face)
    plt.imshow(sub_face)


    for (ex,ey,ew,eh) in eyer:
        sub_face2 = img[ey:ey+eh, ex:ex+ew]
        plt.imshow(sub_face2, cmap=plt.get_cmap('gray'))
      #  face_file_name = "face/" + str(y) + ".jpg"
       # cv2.imwrite("C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/seperate features/eyerightfaceH.jpg", sub_face2)
plt.imshow(sub_face2)




eyeL = cv2.imread("C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/seperate features/eyeleftfaceH.jpg")

eyeL2 = cv2.cvtColor(eyeL, cv2.COLOR_BGR2GRAY)
heightcL, widthcL = eyeL2.shape

eyeLft = cv2.resize(eyeL2, dsize=(72, 72), interpolation=cv2.INTER_CUBIC)

plt.imshow(eyeLft, cmap=plt.get_cmap('gray'))

eyeLeft = cv2.normalize(eyeLft.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)*255

heightLc, widthLc = eyeLeft.shape

plt.imshow(eyeLeft, cmap=plt.get_cmap('gray'))
plt.axis('off')



eye_happy_left = round(((heightLc-8)*(widthLc-8))/64)
flateyeL = np.zeros((eye_happy_left, 66), np.uint8)+2
flateyeL[:,65] = 1
flateyeL[:,64] = 10

k = 0
for i in range(0,heightLc-8,8):
    for j in range(0,widthLc-8,8):
        crop_tmpL1 = eyeLeft[i:i+8,j:j+8]
        flateyeL[k,0:64] = crop_tmpL1.flatten()
        k = k + 1
fspaceEyeLeft = pd.DataFrame(flateyeL)  

fspaceEyeLeft.to_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/fspaceAEyesLH.csv', index=False)


eyeR = cv2.imread("C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/seperate features/eyerightfaceH.jpg")

eyeR2 = cv2.cvtColor(eyeR, cv2.COLOR_BGR2GRAY)
heightcR, widthcR = eyeR2.shape

eyeRgt = cv2.resize(eyeR2, dsize=(72, 72), interpolation=cv2.INTER_CUBIC)

plt.imshow(eyeRgt, cmap=plt.get_cmap('gray'))

eyeRight = cv2.normalize(eyeRgt.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)*255

heightRc, widthRc = eyeRight.shape

plt.imshow(eyeRight, cmap=plt.get_cmap('gray'))
plt.axis('off')



eye_happy_right = round(((heightRc-8)*(widthRc-8))/64)
flateyeR = np.zeros((eye_happy_right, 66), np.uint8)+2
flateyeR[:,65] = 1
flateyeR[:,64] = 11

k = 0
for i in range(0,heightRc-8,8):
    for j in range(0,widthRc-8,8):
        crop_tmpR1 = eyeRight[i:i+8,j:j+8]
        flateyeR[k,0:64] = crop_tmpR1.flatten()
        k = k + 1
fspaceEyeRight = pd.DataFrame(flateyeR)  

fspaceEyeRight.to_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/fspaceAEyesRH.csv', index=False)


EyeLeft = pd.read_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/fspaceAEyesLH.csv')
Eyeright = pd.read_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/fspaceAEyesRH.csv')


frames = [EyeLeft, Eyeright]
mged = pd.concat(frames)
#indx = np.arange(len(mged))
#fspaceeye = np.random.permutation(indx)
#fspaceeye=mged.sample(frac=1).reset_index(drop=True)
mged.to_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/EyeHappyLRJustMerged.csv', index=False)

# ============================================================================= 
# Code for nose identification
# =============================================================================
nose_cascade = cv2.CascadeClassifier('/Users/aishg/anaconda3/envs/project1/Library/etc/haarcascades/haarcascade_mcs_nose.xml')
#nose classifier: https://sourceforge.net/p/emgucv/opencv/ci/d58cd9851fdb592a395488fc5721d11c7436a99c/tree/data/haarcascades/

for (x,y,w,h) in faces:
  #  cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    noseh =  nose_cascade.detectMultiScale(gray, 1.85, 15)


for (nx, ny, nw, nh) in noseh:
        nose_faceh = img[ny:ny+nh, nx:nx+nw]
      #  face_file_name = "face/" + str(y) + ".jpg"
       # cv2.imwrite("C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/seperate features/nosehappy.jpg", nose_faceh)
plt.imshow(nose_faceh)

noseh = cv2.imread("C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/seperate features/nosehappy.jpg")

noseh2 = cv2.cvtColor(noseh, cv2.COLOR_BGR2GRAY)
heightcG, widthcG = noseh2.shape

noseh = cv2.resize(noseh2, dsize=(72, 72), interpolation=cv2.INTER_CUBIC)

plt.imshow(noseh, cmap=plt.get_cmap('gray'))

noseh = cv2.normalize(noseh.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)*255

heightc, widthc = noseh.shape

plt.imshow(noseh, cmap=plt.get_cmap('gray'))
plt.axis('off')

nose_happy = round(((heightc-8)*(widthc-8))/64)
flatnoseh = np.zeros((nose_happy, 66), np.uint8)+2
flatnoseh[:,64] = 2
flatnoseh[:,65] = 1

k = 0
for i in range(0,heightc-8,8):
    for j in range(0,widthc-8,8):
        crop_tmp1 = noseh[i:i+8,j:j+8]
        flatnoseh[k,0:64] = crop_tmp1.flatten()
        k = k + 1
fspaceNoseh = pd.DataFrame(flatnoseh)  

fspaceNoseh.to_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/fspaceANoseH.csv', index=False)


# =============================================================================
# Code For MOUTH - HAPPY
# =============================================================================

#mouth_cascace: https://github.com/sightmachine/SimpleCV/blob/master/SimpleCV/Features/HaarCascades/mouth.xml
mouth_cascade = cv2.CascadeClassifier('C:/Users/aishg/anaconda3/envs/project1/Library/etc/haarcascades/haarcascade_mcs_mouth.xml')

for (x,y,w,h) in faces:
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    lips = mouth_cascade.detectMultiScale(gray, 1.7, 9)

    for (mx, my, mw, mh) in lips:
        mouth_face1H = img[my:my+mh, mx:mx+mw]
       # face_file_name = "face/" + str(y) + ".jpg"
        #cv2.imwrite("C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/seperate features/mouthfaceH.jpg", mouth_face1H)
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

mouth_happy = round(((heightm-8)*(widthm-8))/64)
flatmouthh = np.zeros((mouth_happy, 66), np.uint8)+2
flatmouthh[:,64] = 3
flatmouthh[:,65] = 1

k = 0
for i in range(0,heightm-8,8):
    for j in range(0,widthm-8,8):
        crop_tmp3 = mouthh[i:i+8,j:j+8]
        flatmouthh[k,0:64] = crop_tmp3.flatten()
        k = k + 1
fspaceMouthh = pd.DataFrame(flatmouthh)  

fspaceMouthh.to_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/fspaceMouthH.csv', index=False)



# =============================================================================
# Forehead detection!!!!!!!
# =============================================================================


import cv2
import dlib
import matplotlib.pyplot as plt

face_cascade = cv2.CascadeClassifier('C:/Users/aishg/anaconda3/envs/project1/Library/etc/haarcascades/haarcascade_frontalface_default.xml')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:/Users/aishg/anaconda3/envs/project/lib/site-packages/face_recognition_models/models/shape_predictor_81_face_landmarks.dat")

img = cv2.imread('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/happy/aish.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
tmp = cv2.imread('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/happy/aish.jpg')


faces = detector(gray)

for face in faces:
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()


    landmarks = predictor(gray, face)

    x_pts = []
    y_pts = []

    for n in range(68, 81):
        x = landmarks.part(n).x
        y = landmarks.part(n).y

        x_pts.append(x)
        y_pts.append(y)

        x1 = min(x_pts)
        x2 = max(x_pts)
        y1 = min(y_pts)
        y2 = max(y_pts)

        cv2.rectangle(gray, (x1,y1), (x2,y2), (0, 255, 255), 10)
        forehead = tmp[y1:y2, x1:x2]
       # plt.imshow(forehead)
       # face_file_name = "face/" + str(y) + ".jpg"
        #cv2.imwrite("C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/seperate features/ForeheadH.jpg", forehead)
    plt.imshow(forehead)

    plt.imshow(gray)

    

headH = cv2.imread("C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/seperate features/foreheadH.jpg")

headH2 = cv2.cvtColor(headH, cv2.COLOR_BGR2GRAY)
heightch, widthch = headH2.shape

headhH = cv2.resize(headH2, dsize=(120, 48), interpolation=cv2.INTER_CUBIC)

plt.imshow(headhH, cmap=plt.get_cmap('gray'))

headh = cv2.normalize(headhH.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)*255

heighth, widthh = headh.shape

plt.imshow(headh, cmap=plt.get_cmap('gray'))
plt.axis('off')
 
import pandas as pd

head_happy = round(((heighth-8)*(widthh-8))/64)
flatheadh = np.zeros((head_happy, 66), np.uint8)+2
flatheadh[:,64] = 4
flatheadh[:,65] = 1

k = 0
for i in range(0,heighth-8,8):
    for j in range(0,widthh-8,8):
       crop_tmp4 = headh[i:i+8,j:j+8]
       flatheadh[k,0:64] = crop_tmp4.flatten()
       k = k + 1
fspaceHeadh = pd.DataFrame(flatheadh)  
fspaceHeadh.to_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/fspaceHeadH.csv', index=False)

# =============================================================================
# 
# =============================================================================

Nose = pd.read_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/fspaceANoseH.csv')
Mouth = pd.read_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/fspaceMouthH.csv')
Eyes = pd.read_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/EyeHappyLRJustMerged.csv')
head = pd.read_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/fspaceHeadH.csv')


framesh = [Eyes, Nose, Mouth, head]
framestogether = pd.concat(framesh)
indx1 = np.arange(len(framestogether))
happyExp = np.random.permutation(indx1)
happyExp = framestogether.sample(frac=1).reset_index(drop=True)
happyExp.to_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/Happyface.csv')
