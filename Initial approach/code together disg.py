# -*- coding: utf-8 -*-
"""

@author: aishg
"""

#Code together for disgusted



# =============================================================================
# We detect the eyes in the face and extract to a csv file in a Disgusted face
# =============================================================================


import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import face_recognition
import dlib



face_cascade = cv2.CascadeClassifier('C:/Users/aishg/anaconda3/envs/project1/Library/etc/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:/Users/aishg/anaconda3/envs/project1/Library/etc/haarcascades/haarcascade_eye.xml')


disg = cv2.imread('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/Dataset/Expression Disgusted/IMG_5111.JPG')
gray = cv2.cvtColor(disg, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)

for (x,y,w,h) in faces:
  #  cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
    roi_gray = gray[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(gray, 1.45, 5)
    eyer = eye_cascade.detectMultiScale(gray, 1.2, 10)
    
    
# =============================================================================
# #Detect the eyes after the face has been detected
# =============================================================================

    for (ex,ey,ew,eh) in eyes:
        sub_face = gray[ey:ey+eh, ex:ex+ew]
        plt.imshow(sub_face, cmap=plt.get_cmap('gray'))
    
    
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



eye_disg_left = round(((heightLc-8)*(widthLc-8))/64)
flateyeL = np.zeros((eye_disg_left, 64), np.uint8)

k = 0
for i in range(0,heightLc-8,8):
    for j in range(0,widthLc-8,8):
        crop_tmpL1 = eyeLeft[i:i+8,j:j+8]
        flateyeL[k,0:64] = crop_tmpL1.flatten()
        k = k + 1
fspaceEyeLeft = pd.DataFrame(flateyeL)  
fspaceEyeLeft[64] = 'LeftEye'
fspaceEyeLeft[65] = 'Disgusted'


fspaceEyeLeft.to_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/testing/fspaceAEyesLD3.csv', index=False)


heightcR, widthcR = sub_face2.shape

eyeRgt = cv2.resize(sub_face2, dsize=(72, 72), interpolation=cv2.INTER_CUBIC)

plt.imshow(eyeRgt, cmap=plt.get_cmap('gray'))

eyeRight = cv2.normalize(eyeRgt.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)*255

heightRc, widthRc = eyeRight.shape

plt.imshow(eyeRight, cmap=plt.get_cmap('gray'))
plt.axis('off')



eye_disg_right = round(((heightRc-8)*(widthRc-8))/64)
flateyeR = np.zeros((eye_disg_right, 64), np.uint8)

k = 0
for i in range(0,heightRc-8,8):
    for j in range(0,widthRc-8,8):
        crop_tmpR1 = eyeRight[i:i+8,j:j+8]
        flateyeR[k,0:64] = crop_tmpR1.flatten()
        k = k + 1
fspaceEyeRight = pd.DataFrame(flateyeR)
fspaceEyeRight[64] = 'RightEye'
fspaceEyeRight[65] = 'Disgusted'
  

fspaceEyeRight.to_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/testing/fspaceAEyesRD3.csv', index=False)


frames = [fspaceEyeLeft, fspaceEyeRight]
mged = pd.concat(frames)
indx = np.arange(len(mged))
rndmged = np.random.permutation(indx)
rndmged=mged.sample(frac=1).reset_index(drop=True)
mged.to_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/testing/EyeDJustMerged3.csv', index=False)


# ============================================================================= 
# Code for nose identification
# =============================================================================

nose_cascade = cv2.CascadeClassifier('/Users/aishg/anaconda3/envs/project1/Library/etc/haarcascades/haarcascade_mcs_nose.xml')
#nose classifier: https://sourceforge.net/p/emgucv/opencv/ci/d58cd9851fdb592a395488fc5721d11c7436a99c/tree/data/haarcascades/

for (x,y,w,h) in faces:
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = disg[y:y+h, x:x+w]
    noseD =  nose_cascade.detectMultiScale(gray, 1.7, 10)


    for (nx, ny, nw, nh) in noseD:
        nose_faceD = gray[ny:ny+nh, nx:nx+nw]
      #  face_file_name = "face/" + str(y) + ".jpg"
       # cv2.imwrite("C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/seperate features/nosedisg.jpg", nose_faceD)
        plt.imshow(nose_faceD)

heightcD, widthcD = nose_faceD.shape #cG

noseD = cv2.resize(nose_faceD, dsize=(72, 72), interpolation=cv2.INTER_CUBIC)

plt.imshow(noseD, cmap=plt.get_cmap('gray'))

nosed1 = cv2.normalize(noseD.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)*255
heightd, widthd = nosed1.shape

plt.imshow(nosed1, cmap=plt.get_cmap('gray'))

nose_dist = round(((heightd-8)*(widthd-8))/64)
flatnosed = np.zeros((nose_dist, 64), np.uint8)

k = 0
for i in range(0,heightd-8,8):
    for j in range(0,widthd-8,8):
        crop_tmpd2 = nosed1[i:i+8,j:j+8]
        flatnosed[k,0:64] = crop_tmpd2.flatten()
        k = k + 1
fspaceNoseD = pd.DataFrame(flatnosed)  
fspaceNoseD[64] = 'Nose'
fspaceNoseD[65] = 'Disgusted'


fspaceNoseD.to_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/testing/fspaceANoseD3.csv', index=False)


# =============================================================================
# Code For MOUTH - HAPPY
# =============================================================================

#mouth_cascace: https://github.com/sightmachine/SimpleCV/blob/master/SimpleCV/Features/HaarCascades/mouth.xml
mouth_cascade = cv2.CascadeClassifier('C:/Users/aishg/anaconda3/envs/project1/Library/etc/haarcascades/haarcascade_mcs_mouth.xml')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:/Users/aishg/anaconda3/envs/project/lib/site-packages/face_recognition_models/models/shape_predictor_81_face_landmarks.dat")


disg = cv2.imread('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/Dataset/Expression Disgusted/IMG_5111.JPG')
gray = cv2.cvtColor(disg, cv2.COLOR_BGR2GRAY)
tmp = cv2.imread('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/Dataset/Expression Disgusted/IMG_5111.JPG')

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
        disgmouth = tmp[y1:y2, x1:x2]
      #  face_file_name = "face/" + str(y) + ".jpg"
       # cv2.imwrite("C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/seperate features/ForeheadD.jpg", disghead)

        plt.imshow(disgmouth)
    
disgmouth = cv2.cvtColor(disgmouth, cv2.COLOR_BGR2GRAY)

heightdm, widthdm = disgmouth.shape

mouthd = cv2.resize(disgmouth, dsize=(88, 32), interpolation=cv2.INTER_CUBIC)

plt.imshow(mouthd, cmap=plt.get_cmap('gray'))

mouthd = cv2.normalize(mouthd.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)*255

heightmd, widthmd = mouthd.shape

plt.imshow(mouthd, cmap=plt.get_cmap('gray'))

mouth_disg = round(((heightmd-8)*(widthmd-8))/64)
flatmouthd = np.zeros((mouth_disg, 64), np.uint8)
k = 0
for i in range(0,heightmd-8,8):
    for j in range(0,widthmd-8,8):
        crop_tmpd4 = mouthd[i:i+8,j:j+8]
        flatmouthd[k,0:64] = crop_tmpd4.flatten()
        k = k + 1
fspaceMouthD = pd.DataFrame(flatmouthd)  
fspaceMouthD[64] = 'Mouth'
fspaceMouthD[65] = 'Disgusted'


fspaceMouthD.to_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/testing/fspaceMouthD3.csv', index=False)



# =============================================================================
# Forehead detection!!!!!!!
# =============================================================================

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

        cv2.rectangle(gray, (x1,y1), (x2,y2), (0, 255, 255), 3)
        disghead = tmp[y1:y2, x1:x2]

        plt.imshow(disghead)
    
headD2 = cv2.cvtColor(disghead, cv2.COLOR_BGR2GRAY)
heightcd, widthcd = headD2.shape

headhD = cv2.resize(headD2, dsize=(120, 48), interpolation=cv2.INTER_CUBIC)

plt.imshow(headhD, cmap=plt.get_cmap('gray'))

headd = cv2.normalize(headhD.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)*255

heightd, widthd = headd.shape

plt.imshow(headd, cmap=plt.get_cmap('gray'))
plt.axis('off')

head_disg = round(((heightd-8)*(widthd-8))/64)
flatheadd = np.zeros((head_disg, 64), np.uint8)

k = 0
for i in range(0,heightd-8,8):
    for j in range(0,widthd-8,8):
        crop_tmpd4 = headd[i:i+8,j:j+8]
        flatheadd[k,0:64] = crop_tmpd4.flatten()
        k = k + 1
fspaceHeadd = pd.DataFrame(flatheadd)  
fspaceHeadd[64] = 'Forehead'
fspaceHeadd[65] = 'Disgusted'


fspaceHeadd.to_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/testing/fspaceHeadD3.csv', index=False)

# =============================================================================
# 
# =============================================================================


framesD = [fspaceHeadd, fspaceMouthD, fspaceNoseD, mged]
mgeddisg = pd.concat(framesD)
indxdisg = np.arange(len(mgeddisg))
disgExp = np.random.permutation(indxdisg)
disgExp = mgeddisg.sample(frac=1).reset_index(drop=True)
disgExp.to_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/testing/DisgExpressionsP3.csv', index=False)


# =============================================================================
# 
# =============================================================================
d1 = pd.read_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/testing/DisgExpressionsP.csv')
d2 = pd.read_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/testing/DisgExpressionsP2.csv')
d3 = pd.read_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/testing/DisgExpressionsP3.csv')



dd = [d1, d2, d3]
dd1 = pd.concat(dd)
indxdisg = np.arange(len(dd1))
disgExp = np.random.permutation(indxdisg)
disgExp = dd1.sample(frac=1).reset_index(drop=True)
disgExp.to_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/testing/DisgExpressionsP123.csv', index=False)

