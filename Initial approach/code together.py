# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 21:24:15 2021

@author: aishg
"""


# =============================================================================
# We detect the eyes in the face and extract to a csv file in a HAPPY face
# =============================================================================


import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt


face_cascade = cv2.CascadeClassifier('C:/Users/aishg/anaconda3/envs/project1/Library/etc/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:/Users/aishg/anaconda3/envs/project1/Library/etc/haarcascades/haarcascade_eye.xml')


img = cv2.imread('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/Dataset/Expression happy/IMG_5177.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)

# =============================================================================
# #Detect the face in the picture
# =============================================================================

for (x,y,w,h) in faces:
    roi_gray = gray[y:y+h, x:x+w]
    cv2.rectangle(gray, (x,y), (x+w,y+h), (255,255,255), 2)

    roi_color = gray[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(gray, 1.5, 25)
    eyer = eye_cascade.detectMultiScale(gray, 1.2, 10)
    
    
# =============================================================================
# #Detect the eyes after the face has been detected
# =============================================================================

    for (ex,ey,ew,eh) in eyes:
        sub_face = gray[ey:ey+eh, ex:ex+ew]
        plt.imshow(sub_face, cmap=plt.get_cmap('gray'))
        #The below line of code helps in saving the cropped image seperately 
        #  face_file_name = "face/" + str(y) + ".jpg"
       # cv2.imwrite("C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/seperate features/eyeleftfaceH.jpg", sub_face)


    for (ex,ey,ew,eh) in eyer:
        sub_face2 = gray[ey:ey+eh, ex:ex+ew]
        plt.imshow(sub_face2, cmap=plt.get_cmap('gray'))



heightcL, widthcL = sub_face.shape

eyeLft = cv2.resize(sub_face, dsize=(72, 72), interpolation=cv2.INTER_CUBIC)

plt.imshow(eyeLft, cmap=plt.get_cmap('gray'))

eyeLeft = cv2.normalize(eyeLft.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)*255

heightLc, widthLc = eyeLeft.shape

plt.imshow(eyeLeft, cmap=plt.get_cmap('gray'))
plt.axis('off')



eye_happy_left = round(((heightLc-8)*(widthLc-8))/64)
flateyeL = np.zeros((eye_happy_left, 64), np.uint8)

k = 0
for i in range(0,heightLc-8,8):
    for j in range(0,widthLc-8,8):
        crop_tmpL1 = eyeLeft[i:i+8,j:j+8]
        flateyeL[k,0:64] = crop_tmpL1.flatten()
        k = k + 1
fspaceEyeLeft = pd.DataFrame(flateyeL)  
fspaceEyeLeft[64] = 'LeftEye'

fspaceEyeLeft[65] = 'Happy'


fspaceEyeLeft.to_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/testing/fspaceAEyesLH3.csv', index=False)

heightcR, widthcR = sub_face2.shape

eyeRgt = cv2.resize(sub_face2, dsize=(72, 72), interpolation=cv2.INTER_CUBIC)

plt.imshow(eyeRgt, cmap=plt.get_cmap('gray'))

eyeRight = cv2.normalize(eyeRgt.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)*255

heightRc, widthRc = eyeRight.shape

plt.imshow(eyeRight, cmap=plt.get_cmap('gray'))
plt.axis('off')



eye_happy_right = round(((heightRc-8)*(widthRc-8))/64)
flateyeR = np.zeros((eye_happy_right, 64), np.uint8)

k = 0
for i in range(0,heightRc-8,8):
    for j in range(0,widthRc-8,8):
        crop_tmpR1 = eyeRight[i:i+8,j:j+8]
        flateyeR[k,0:64] = crop_tmpR1.flatten()
        k = k + 1
fspaceEyeRight = pd.DataFrame(flateyeR)  
fspaceEyeRight[64] = "RightEye"
fspaceEyeRight[65] = 'Happy'

fspaceEyeRight.to_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/testing/fspaceAEyesR3.csv', index=False)


EyeLeft = pd.read_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/fspaceAEyesLH.csv')
Eyeright = pd.read_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/fspaceAEyesRH.csv')


frames = [fspaceEyeLeft, fspaceEyeRight]
mged = pd.concat(frames)
indx = np.arange(len(mged))
fspaceeye = np.random.permutation(indx)
fspaceeye=mged.sample(frac=1).reset_index(drop=True)
fspaceeye.to_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/testing/EyeHappyLRJustMerged3.csv', index=False)


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
        nose_faceh = gray[ny:ny+nh, nx:nx+nw]
      #  face_file_name = "face/" + str(y) + ".jpg"
       # cv2.imwrite("C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/seperate features/nosehappy.jpg", nose_faceh)
plt.imshow(nose_faceh, cmap=plt.get_cmap('gray'))

heightcG, widthcG = nose_faceh.shape

noseh = cv2.resize(nose_faceh, dsize=(72, 72), interpolation=cv2.INTER_CUBIC)

plt.imshow(noseh, cmap=plt.get_cmap('gray'))

noseh = cv2.normalize(noseh.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)*255

heightc, widthc = noseh.shape

plt.imshow(noseh, cmap=plt.get_cmap('gray'))
plt.axis('off')

nose_happy = round(((heightc-8)*(widthc-8))/64)
flatnoseh = np.zeros((nose_happy, 64), np.uint8)

k = 0
for i in range(0,heightc-8,8):
    for j in range(0,widthc-8,8):
        crop_tmp1 = noseh[i:i+8,j:j+8]
        flatnoseh[k,0:64] = crop_tmp1.flatten()
        k = k + 1
fspaceNoseh = pd.DataFrame(flatnoseh)  
fspaceNoseh[64] = "Nose"
fspaceNoseh[65] = "Happy"


fspaceNoseh.to_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/testing/fspaceANoseH3.csv', index=False)



# =============================================================================
# Code For MOUTH - HAPPY
# =============================================================================

#mouth_cascace: https://github.com/sightmachine/SimpleCV/blob/master/SimpleCV/Features/HaarCascades/mouth.xml
mouth_cascade = cv2.CascadeClassifier('C:/Users/aishg/anaconda3/envs/project1/Library/etc/haarcascades/haarcascade_mcs_mouth.xml')

# =============================================================================
# This block of code is used only when the lips are closed together
# =============================================================================
for (x,y,w,h) in faces:
    roi_gray = gray[y:y+h, x:x+w]
    lips = mouth_cascade.detectMultiScale(gray, 1.7, 5)

    for (mx, my, mw, mh) in lips:

        mouth_face1H = gray[my:my+mh, mx:mx+mw]
       # face_file_name = "face/" + str(y) + ".jpg"
        #cv2.imwrite("C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/seperate features/mouthfaceH.jpg", mouth_face1H)
plt.imshow(mouth_face1H)
    
# =============================================================================
# This code help in detecting the lips and cropping the entire region
# =============================================================================
import dlib

mouth_cascade = cv2.CascadeClassifier('C:/Users/aishg/anaconda3/envs/project1/Library/etc/haarcascades/haarcascade_mcs_mouth.xml')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:/Users/aishg/anaconda3/envs/project/lib/site-packages/face_recognition_models/models/shape_predictor_81_face_landmarks.dat")


img = cv2.imread('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/Dataset/Expression happy/IMG_5177.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
tmp = cv2.imread('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/Dataset/Expression happy/IMG_5177.jpg')

faces = detector(gray)

faces = detector(gray)
for face in faces:
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()


    landmarks = predictor(gray, face)

    x_pts = []
    y_pts = []
    
    for n in range(48, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        
        
        x_pts.append(x)
        y_pts.append(y)

        x1 = min(x_pts)
        x2 = max(x_pts)
        y1 = min(y_pts)
        y2 = max(y_pts)

        cv2.rectangle(gray, (x1,y1), (x2,y2), (0, 255, 255), 3)
        happymouth = tmp[y1:y2, x1:x2]
      #  face_file_name = "face/" + str(y) + ".jpg"
       # cv2.imwrite("C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/seperate features/ForeheadD.jpg", disghead)

        plt.imshow(happymouth)
        
# =============================================================================
#         
# =============================================================================


mouth_face1H = cv2.cvtColor(happymouth, cv2.COLOR_BGR2GRAY)

heightcm, widthcm = mouth_face1H.shape

mouthh = cv2.resize(mouth_face1H, dsize=(96, 48), interpolation=cv2.INTER_CUBIC)

plt.imshow(mouthh, cmap=plt.get_cmap('gray'))

mouthh = cv2.normalize(mouthh.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)*255

heightm, widthm = mouthh.shape

plt.imshow(mouthh, cmap=plt.get_cmap('gray'))
plt.axis('off')

mouth_happy = round(((heightm-8)*(widthm-8))/64)
flatmouthh = np.zeros((mouth_happy, 64), np.uint8)

k = 0
for i in range(0,heightm-8,8):
    for j in range(0,widthm-8,8):
        crop_tmp3 = mouthh[i:i+8,j:j+8]
        flatmouthh[k,0:64] = crop_tmp3.flatten()
        k = k + 1
fspaceMouthh = pd.DataFrame(flatmouthh)
fspaceMouthh[64] = "Mouth"
fspaceMouthh[65] = "Happy"
  

fspaceMouthh.to_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/testing/fspaceMouthH3.csv', index=False)




# =============================================================================
# Forehead detection!!!!!!!
# =============================================================================


import cv2
import dlib
import matplotlib.pyplot as plt

face_cascade = cv2.CascadeClassifier('C:/Users/aishg/anaconda3/envs/project1/Library/etc/haarcascades/haarcascade_frontalface_default.xml')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:/Users/aishg/anaconda3/envs/project/lib/site-packages/face_recognition_models/models/shape_predictor_81_face_landmarks.dat")


img = cv2.imread('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/Dataset/Expression happy/IMG_5177.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
tmp = cv2.imread('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/Dataset/Expression happy/IMG_5177.jpg')
tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)

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
    plt.imshow(forehead)
    

heightch, widthch = forehead.shape

headhH = cv2.resize(forehead, dsize=(120, 48), interpolation=cv2.INTER_CUBIC)

plt.imshow(headhH, cmap=plt.get_cmap('gray'))

headh = cv2.normalize(headhH.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)*255

heighth, widthh = headh.shape

plt.imshow(headh, cmap=plt.get_cmap('gray'))
plt.axis('off')
 
head_happy = round(((heighth-8)*(widthh-8))/64)
flatheadh = np.zeros((head_happy, 64), np.uint8)+2

k = 0
for i in range(0,heighth-8,8):
    for j in range(0,widthh-8,8):
       crop_tmp4 = headh[i:i+8,j:j+8]
       flatheadh[k,0:64] = crop_tmp4.flatten()
       k = k + 1
fspaceHeadh = pd.DataFrame(flatheadh)  
fspaceHeadh[64] = "Forehead"
fspaceHeadh[65] = "Happy"


fspaceHeadh.to_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/testing/fspaceHeadH3.csv', index=False)

# =============================================================================
# 
# =============================================================================


framesh = [fspaceeye, fspaceHeadh, fspaceMouthh, fspaceNoseh]
framestogether = pd.concat(framesh)
indx1 = np.arange(len(framestogether))
happyExp = np.random.permutation(indx1)
happyExp = framestogether.sample(frac=1).reset_index(drop=True)
happyExp.to_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/testing/Happyface3.csv', index=False)



# =============================================================================
# Detecting the facial landmarks using haar cascade
#=============================================================================

faces = detector(img)

for face in faces:
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()


    landmarks = predictor(img, face)

    x_pts = []
    y_pts = []
    
    for n in range(0, 81):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        cv2.circle(img, (x, y), 4, (0, 255, 0), 30) 
    

# =============================================================================
# 
# =============================================================================
# =============================================================================
# Finding the distance between points
# =============================================================================

from scipy.spatial import distance as dist
 
def EUC_dist(landmarks):
    A = dist.euclidean(landmarks[0:81], landmarks[34])
    
    return A

# =============================================================================
# Merging the feature spaces
# =============================================================================


h1 = pd.read_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/testing/Happyface.csv')
h2 = pd.read_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/testing/Happyface2.csv')
h3 = pd.read_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/testing/Happyface3.csv')



ha = [h1, h2, h3]
ha1 = pd.concat(ha)
indxh = np.arange(len(ha1))
hExp = np.random.permutation(indxh)
hExp = ha1.sample(frac=1).reset_index(drop=True)
hExp.to_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/testing/Happyface123.csv', index=False)


#Jaw detection

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
             #nose = img[y1:y2, x1:x2]
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

                 cv2.rectangle(gray, (x1,y1), (x2,y2), (0, 255, 255), 10)
                 #cv2.rectangle(img, (x1,y1), (x2,y2), (255, 255, 255), -1)
                 chin = img[y1:y2, x1:x2]
                 plt.imshow(chin)
                 cv2.imwrite("C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/seperate features/chinH.jpg", chin)


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

k = 0
for i in range(0,heighth-8,8):
    for j in range(0,widthh-8,8):
       crop_tmp5 = chinh[i:i+8,j:j+8]
       flatchinh[k,0:64] = crop_tmp5.flatten()
       k = k + 1
fspaceChinh = pd.DataFrame(flatchinh)

chinH = fspaceChinh[~np.all(fspaceChinh == 0, axis=1)]
chinH[64]=5
chinH[65]=1

chinH.to_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/fspaceChinH.csv', index=False)


# =============================================================================
# 
# =============================================================================

