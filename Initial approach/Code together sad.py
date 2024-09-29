# -*- coding: utf-8 -*-
"""

@author: aishg
"""

#Code together



# =============================================================================
# We detect the eyes in the face and extract to a csv file in a HAPPY face
# =============================================================================


import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import face_recognition
import dlib



face_cascade = cv2.CascadeClassifier('C:/Users/aishg/anaconda3/envs/project1/Library/etc/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:/Users/aishg/anaconda3/envs/project1/Library/etc/haarcascades/haarcascade_eye.xml')

sad = cv2.imread('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/Dataset/Expression Sad/sad3.jpg')
gray = cv2.cvtColor(sad, cv2.COLOR_BGR2GRAY)


faces = face_cascade.detectMultiScale(gray, 1.3, 5)

# =============================================================================
# #Detect the face in the picture
# =============================================================================

for (x,y,w,h) in faces:
    roi_gray = gray[y:y+h, x:x+w]
    plt.imshow(roi_gray)

    eyes = eye_cascade.detectMultiScale(gray, 1.05, 15)
    eyer = eye_cascade.detectMultiScale(gray, 1.25, 5)
    
    
# =============================================================================
# #Detect the eyes after the face has been detected
# =============================================================================

    for (ex,ey,ew,eh) in eyes:
        eye_left = gray[ey:ey+eh, ex:ex+ew]
        plt.imshow(eye_left, cmap=plt.get_cmap('gray'))


    for (ex,ey,ew,eh) in eyer:
        sad_face2 = gray[ey:ey+eh, ex:ex+ew]
        plt.imshow(sad_face2, cmap=plt.get_cmap('gray'))
        #face_file_name = "face/" + str(y) + ".jpg"
        #cv2.imwrite("C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/seperate features/eyerightfaceS.jpg", sad_face2)
        


heightcL, widthcL = eye_left.shape

eyeLft = cv2.resize(eye_left, dsize=(72, 72), interpolation=cv2.INTER_CUBIC)

plt.imshow(eyeLft, cmap=plt.get_cmap('gray'))

eyeLeft = cv2.normalize(eyeLft.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)*255

heightLc, widthLc = eyeLeft.shape

plt.imshow(eyeLeft, cmap=plt.get_cmap('gray'))

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
fspaceEyeLeft[65] = 'Sad'
  

fspaceEyeLeft.to_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/testing/fspaceAEyesLS3.csv', index=False)



heightcR, widthcR = sad_face2.shape

eyeRgt = cv2.resize(sad_face2, dsize=(72, 72), interpolation=cv2.INTER_CUBIC)

plt.imshow(eyeRgt, cmap=plt.get_cmap('gray'))

eyeRight = cv2.normalize(eyeRgt.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)*255

heightRc, widthRc = eyeRight.shape

plt.imshow(eyeRight, cmap=plt.get_cmap('gray'))

eye_sad_right = round(((heightRc-8)*(widthRc-8))/64)
flateyeR = np.zeros((eye_sad_right, 64), np.uint8)

k = 0
for i in range(0,heightRc-8,8):
    for j in range(0,widthRc-8,8):
        crop_tmpR1 = eyeRight[i:i+8,j:j+8]
        flateyeR[k,0:64] = crop_tmpR1.flatten()
        k = k + 1
fspaceEyeRight = pd.DataFrame(flateyeR)  
fspaceEyeRight[64] = 'RightEye'
fspaceEyeRight[65] = 'Sad'


fspaceEyeRight.to_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/testing/fspaceAEyesRS3.csv', index=False)


frames = [fspaceEyeLeft, fspaceEyeRight]
merge = pd.concat(frames)
indx = np.arange(len(merge))
fspaceeye = np.random.permutation(indx)
fspaceeye=merge.sample(frac=1).reset_index(drop=True)
fspaceeye.to_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/testing/EyeSadLRjustMerge3.csv', index=False)



# ============================================================================= 
# Code for nose identification
# =============================================================================

nose_cascade = cv2.CascadeClassifier('/Users/aishg/anaconda3/envs/project1/Library/etc/haarcascades/haarcascade_mcs_nose.xml')
#nose classifier: https://sourceforge.net/p/emgucv/opencv/ci/d58cd9851fdb592a395488fc5721d11c7436a99c/tree/data/haarcascades/

for (x,y,w,h) in faces:
    roi_gray = gray[y:y+h, x:x+w]
    noseS =  nose_cascade.detectMultiScale(gray, 1.2, 20)


    for (nx, ny, nw, nh) in noseS:
        nose_faceS = gray[ny:ny+nh, nx:nx+nw]
        plt.imshow(nose_faceS)

heightcS, widthcS = nose_faceS.shape #cG

noseS = cv2.resize(nose_faceS, dsize=(72, 72), interpolation=cv2.INTER_CUBIC)

plt.imshow(noseS, cmap=plt.get_cmap('gray'))

noses1 = cv2.normalize(noseS.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)*255
heights, widths = noses1.shape

plt.imshow(noses1, cmap=plt.get_cmap('gray'))

nose_sad = round(((heights-8)*(widths-8))/64)
flatnoses = np.zeros((nose_sad, 64), np.uint8)

k = 0
for i in range(0,heights-8,8):
    for j in range(0,widths-8,8):
        crop_tmps2 = noses1[i:i+8,j:j+8]
        flatnoses[k,0:64] = crop_tmps2.flatten()
        k = k + 1
fspaceNoseS = pd.DataFrame(flatnoses)  
fspaceNoseS[64] = 'Nose'
fspaceNoseS[65] = 'Sad'


fspaceNoseS.to_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/testing/fspaceANoseS3.csv', index=False)


# =============================================================================
# Code For MOUTH - HAPPY
# =============================================================================

#mouth_cascace: https://github.com/sightmachine/SimpleCV/blob/master/SimpleCV/Features/HaarCascades/mouth.xml
mouth_cascade = cv2.CascadeClassifier('C:/Users/aishg/anaconda3/envs/project1/Library/etc/haarcascades/haarcascade_mcs_mouth.xml')
# =============================================================================
# 
# for (x,y,w,h) in faces:
#   #  cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
#     roi_gray = gray[y:y+h, x:x+w]
#     roi_color = sad[y:y+h, x:x+w]
#     lips = mouth_cascade.detectMultiScale(gray, 1.25, 20)
# 
#     for (mx, my, mw, mh) in lips:
#         my = int(my - 0.15*mh)
#         mouth_face1S = gray[my:my+mh, mx:mx+mw]
# #        face_file_name = "face/" + str(y) + ".jpg"
#  #       cv2.imwrite("C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/seperate features/mouthfaceS.jpg", mouth_face1S)
#         plt.imshow(mouth_face1S)
#     
# =============================================================================
face_cascade = cv2.CascadeClassifier('C:/Users/aishg/anaconda3/envs/project1/Library/etc/haarcascades/haarcascade_frontalface_default.xml')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:/Users/aishg/anaconda3/envs/project/lib/site-packages/face_recognition_models/models/shape_predictor_81_face_landmarks.dat")

sad = cv2.imread('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/Dataset/Expression Sad/sad3.jpg')
gray = cv2.cvtColor(sad, cv2.COLOR_BGR2GRAY)
temp5 = cv2.imread('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/Dataset/Expression Sad/sad3.jpg')

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
        sadmouth = temp5[y1:y2, x1:x2]
      #  face_file_name = "face/" + str(y) + ".jpg"
       # cv2.imwrite("C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/seperate features/ForeheadD.jpg", disghead)

        plt.imshow(sadmouth)    

sadmouth = cv2.cvtColor(sadmouth, cv2.COLOR_BGR2GRAY)

heightsm, widthsm = sadmouth.shape

mouths = cv2.resize(sadmouth, dsize=(80, 48), interpolation=cv2.INTER_CUBIC)

plt.imshow(mouths, cmap=plt.get_cmap('gray'))

mouths = cv2.normalize(mouths.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)*255

heightms, widthms = mouths.shape

plt.imshow(mouths, cmap=plt.get_cmap('gray'))

mouth_sad = round(((heightms-8)*(widthms-8))/64)
flatmouths = np.zeros((mouth_sad, 64), np.uint8)

k = 0
for i in range(0,heightms-8,8):
    for j in range(0,widthms-8,8):
        crop_tmps4 = mouths[i:i+8,j:j+8]
        flatmouths[k,0:64] = crop_tmps4.flatten()
        k = k + 1
fspaceMouths = pd.DataFrame(flatmouths)  
fspaceMouths[64] = 'Mouth'
fspaceMouths[65] = 'Sad'


fspaceMouths.to_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/testing/fspaceMouthS3.csv', index=False)


# =============================================================================
# Forehead detection!!!!!!!
# =============================================================================


# =============================================================================
# 
# =============================================================================



face_cascade = cv2.CascadeClassifier('C:/Users/aishg/anaconda3/envs/project1/Library/etc/haarcascades/haarcascade_frontalface_default.xml')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:/Users/aishg/anaconda3/envs/project/lib/site-packages/face_recognition_models/models/shape_predictor_81_face_landmarks.dat")

sad1 = cv2.imread('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/Dataset/Expression Sad/sad3.jpg')
gray = cv2.cvtColor(sad, cv2.COLOR_BGR2GRAY)
sad = cv2.imread('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/Dataset/Expression Sad/sad3.jpg')

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

        cv2.rectangle(gray, (x1,y1), (x2,y2), (0, 255, 255), 3)
        sadhead = sad[y1:y2, x1:x2]
      #  face_file_name = "face/" + str(y) + ".jpg"
       # cv2.imwrite("C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/seperate features/ForeheadS.jpg", sadhead)
        plt.imshow(sadhead)

sad1 = cv2.cvtColor(sadhead, cv2.COLOR_BGR2GRAY)

heightcs, widthcs = sad1.shape
headhS = cv2.resize(sad1, dsize=(120, 48), interpolation=cv2.INTER_CUBIC)

plt.imshow(headhS, cmap=plt.get_cmap('gray'))

heads = cv2.normalize(headhS.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)*255

heights, widths = heads.shape
 
import pandas as pd

head_sad = round(((heights-8)*(widths-8))/64)
flatheads = np.zeros((head_sad, 64), np.uint8)

k = 0
for i in range(0,heights-8,8):
    for j in range(0,widths-8,8):
       crop_tmp4 = heads[i:i+8,j:j+8]
       flatheads[k,0:64] = crop_tmp4.flatten()
       k = k + 1
fspaceHeads = pd.DataFrame(flatheads)  
fspaceHeads[64] = "Forehead"
fspaceHeads[65] = 'Sad'


fspaceHeads.to_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/testing/fspaceHeadS3.csv', index=False)


# =============================================================================
# 
# tmp = cv2.imread('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/sad/sad (12).jpg')
# img = cv2.imread('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/sad/sad (12).jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# #plt.imshow(tmp)
# 
# faces = detector(gray)
# 
# for face in faces:
#     x1 = face.left()
#     y1 = face.top()
#     x2 = face.right()
#     y2 = face.bottom()
# 
# 
#     landmarks = predictor(gray, face)
# 
#     x_pts = []
#     y_pts = []
# 
#     for n in range(49, 65):
#          x = landmarks.part(n).x
#          y = landmarks.part(n).y
#  
#          x_pts.append(x)
#          y_pts.append(y)
#  
#          x1 = min(x_pts)
#          x2 = max(x_pts)
#          y1 = min(y_pts)
#          y2 = max(y_pts)
#  
#          cv2.rectangle(gray, (x1,y1), (x2,y2), (0, 255, 255), 10)
#          cv2.rectangle(img, (x1,y1), (x2,y2), (0, 0, 0), -1)
#          plt.imshow(img)
#          
# # =============================================================================
#         
#         
#          for n in range(28, 36):
#              x = landmarks.part(n).x
#              y = landmarks.part(n).y
#             
#              x_pts.append(x)
#              y_pts.append(y)
# 
#              x1 = min(x_pts)
#              x2 = max(x_pts)
#              y1 = min(y_pts)
#              y2 = max(y_pts)
# 
#              cv2.rectangle(gray, (x1,y1), (x2,y2), (0, 255, 255), 10)
#              cv2.rectangle(img, (x1,y1), (x2,y2), (0, 0, 0), -1)
#              #nose = img[y1:y2, x1:x2]
#              plt.imshow(img)
#              
#              for n in range(1, 16):
#                  x = landmarks.part(n).x
#                  y = landmarks.part(n).y
#                  
#                  x_pts.append(x)
#                  y_pts.append(y)
#                  
#                  x1 = min(x_pts)
#                  x2 = max(x_pts)
#                  y1 = min(y_pts)
#                  y2 = max(y_pts)
# 
#                  cv2.rectangle(gray, (x1,y1), (x2,y2), (255, 255, 255), 10)
#                  chin = img[y1:y2, x1:x2]
#                  plt.imshow(chin)
#                  cv2.imwrite("C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/seperate features/chinS.jpg", chin)
# 
# 
# chinS = cv2.imread("C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/seperate features/chinS.jpg")
# 
# chinS2 = cv2.cvtColor(chinS, cv2.COLOR_BGR2GRAY)
# heightcs, widthcs = chinS2.shape
# 
# chinsS = cv2.resize(chinS2, dsize=(240, 152), interpolation=cv2.INTER_CUBIC)
# 
# plt.imshow(chinsS, cmap=plt.get_cmap('gray'))
# 
# chins = cv2.normalize(chinsS.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)*255
# 
# heights, widths = chins.shape
# 
# plt.imshow(chins, cmap=plt.get_cmap('gray'))
# plt.axis('off')
#  
# import pandas as pd
# import numpy as np
# 
# chin_sad = round(((heights-8)*(widths-8))/64)
# flatchins = np.zeros((chin_sad, 66), np.uint8)
# 
# k = 0
# for i in range(0,heights-8,8):
#     for j in range(0,widths-8,8):
#        crop_tmp5 = chins[i:i+8,j:j+8]
#        flatchins[k,0:64] = crop_tmp5.flatten()
#        k = k + 1
# fspaceChins = pd.DataFrame(flatchins)  
# 
# chinS = fspaceChins[~np.all(fspaceChins == 0, axis=1)]
# chinS[64]=5
# chinS[65]=2
# 
# chinS.to_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/fspaceChinS.csv', index=False)
# =============================================================================


framesS = [fspaceeye, fspaceNoseS, fspaceMouths, fspaceHeads]
mgedsad = pd.concat(framesS)
indxsad = np.arange(len(mgedsad))
sadExp = np.random.permutation(indxsad)
sadExp = mgedsad.sample(frac=1).reset_index(drop=True)
sadExp.to_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/testing/SadExpressionsP3.csv', index=False)


# =============================================================================
# 
# =============================================================================


sa1 = pd.read_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/testing/SadExpressionsP.csv')
sa2 = pd.read_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/testing/SadExpressionsP2.csv')
sa3 = pd.read_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/testing/SadExpressionsP3.csv')



sasa = [sa1, sa2, sa3]
sasa1 = pd.concat(sasa)
indxsad = np.arange(len(sasa1))
sadExp = np.random.permutation(indxsad)
sadExp = sasa1.sample(frac=1).reset_index(drop=True)
sadExp.to_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/testing/SadExpressionsP123.csv', index=False)
