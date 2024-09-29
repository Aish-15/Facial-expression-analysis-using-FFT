# -*- coding: utf-8 -*-
"""

@author: aishg
"""

#Code together for disgusted



# =============================================================================
# We detect the eyes in the face and extract to a csv file in a Fear face
# =============================================================================


import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import dlib



face_cascade = cv2.CascadeClassifier('C:/Users/aishg/anaconda3/envs/project1/Library/etc/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:/Users/aishg/anaconda3/envs/project1/Library/etc/haarcascades/haarcascade_eye.xml')

img = cv2.imread("C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/Dataset/Expression Fear/IMG_5207.JPG")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)

# =============================================================================
# 
for (x,y,w,h) in faces:
     cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,0), 12)
     plt.imshow(img)
     roi_gray = gray[y:y+h, x:x+w]
     eyes = eye_cascade.detectMultiScale(gray, 1.6, 10)
     eyer = eye_cascade.detectMultiScale(gray, 1.9, 10)
     

# =============================================================================
# #Detect the eyes after the face has been detected
# =============================================================================

    for (ex,ey,ew,eh) in eyes:
cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),22)
        print(ex,ey)

        sub_face = gray[ey:ey+eh, ex:ex+ew]
        plt.imshow(sub_face, cmap=plt.get_cmap('gray'))
        #The below line of code helps in saving the cropped image seperately 
    #    face_file_name = "face/" + str(y) + ".jpg"
     #   cv2.imwrite("C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/seperate features/eyeleftfaceF.jpg", sub_face)


    for (ex,ey,ew,eh) in eyer:
        sub_face2 = gray[ey:ey+eh, ex:ex+ew]
        plt.imshow(sub_face2, cmap=plt.get_cmap('gray'))
     #   face_file_name = "face/" + str(y) + ".jpg"
      #  cv2.imwrite("C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/seperate features/eyerightfaceF.jpg", sub_face2)

heightcL, widthcL = sub_face.shape

eyeLft = cv2.resize(sub_face, dsize=(72, 72), interpolation=cv2.INTER_CUBIC)

plt.imshow(eyeLft, cmap=plt.get_cmap('gray'))

eyeLeft = cv2.normalize(eyeLft.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)*255

heightLc, widthLc = eyeLeft.shape

plt.imshow(eyeLeft, cmap=plt.get_cmap('gray'))
plt.axis('off')



eye_fear_left = round(((heightLc-8)*(widthLc-8))/64)
flateyeL = np.zeros((eye_fear_left, 64), np.uint8)

k = 0
for i in range(0,heightLc-8,8):
    for j in range(0,widthLc-8,8):
        crop_tmpL1 = eyeLeft[i:i+8,j:j+8]
        flateyeL[k,0:64] = crop_tmpL1.flatten()
        k = k + 1
fspaceEyeLeft = pd.DataFrame(flateyeL)  
fspaceEyeLeft[64] = 'LeftEye'
fspaceEyeLeft[65] = 'Fear'


fspaceEyeLeft.to_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/testing/fspaceAEyesLF3.csv', index=False)


heightcR, widthcR = sub_face2.shape

eyeRgt = cv2.resize(sub_face2, dsize=(80, 72), interpolation=cv2.INTER_CUBIC)

plt.imshow(eyeRgt, cmap=plt.get_cmap('gray'))

eyeRight = cv2.normalize(eyeRgt.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)*255

heightRc, widthRc = eyeRight.shape

plt.imshow(eyeRight, cmap=plt.get_cmap('gray'))
plt.axis('off')



eye_fear_right = round(((heightRc-8)*(widthRc-8))/64)
flateyeR = np.zeros((eye_fear_right, 64), np.uint8)

k = 0
for i in range(0,heightRc-8,8):
    for j in range(0,widthRc-8,8):
        crop_tmpR1 = eyeRight[i:i+8,j:j+8]
        flateyeR[k,0:64] = crop_tmpR1.flatten()
        k = k + 1
fspaceEyeRight = pd.DataFrame(flateyeR)  
fspaceEyeRight[64] = 'RightEye'
fspaceEyeRight[65] = 'Fear'


fspaceEyeRight.to_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/testing/fspaceAEyesRF3.csv', index=False)


frames = [fspaceEyeLeft, fspaceEyeRight]
mged = pd.concat(frames)
indx = np.arange(len(mged))
rndmged = np.random.permutation(indx)
rndmged=mged.sample(frac=1).reset_index(drop=True)
rndmged.to_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/testing/EyeFLR3.csv', index=False)



framesfear = [fspaceEyeLeft, fspaceEyeRight]
justmged = pd.concat(framesfear)
justmged.to_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/Eyefearjustmerged.csv', index=False)


# ============================================================================= 
# Code for nose identification
# =============================================================================

nose_cascade = cv2.CascadeClassifier('/Users/aishg/anaconda3/envs/project1/Library/etc/haarcascades/haarcascade_mcs_nose.xml')
#nose classifier: https://sourceforge.net/p/emgucv/opencv/ci/d58cd9851fdb592a395488fc5721d11c7436a99c/tree/data/haarcascades/

for (x,y,w,h) in faces:
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = fear[y:y+h, x:x+w]
    noseF =  nose_cascade.detectMultiScale(gray, 2.1, 15)


    for (nx, ny, nw, nh) in noseF:
        nose_faceF = gray[ny:ny+nh, nx:nx+nw]
       # face_file_name = "face/" + str(y) + ".jpg"
        #cv2.imwrite("C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/seperate features/nosefear.jpg", nose_faceF)
        plt.imshow(nose_faceF)

heightcF, widthcF = nose_faceF.shape #cG

noseF = cv2.resize(nose_faceF, dsize=(72, 72), interpolation=cv2.INTER_CUBIC)

plt.imshow(noseF, cmap=plt.get_cmap('gray'))

nosef1 = cv2.normalize(noseF.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)*255
heightf, widthf = nosef1.shape

plt.imshow(nosef1, cmap=plt.get_cmap('gray'))

nose_fear = round(((heightf-8)*(widthf-8))/64)
flatnosef = np.zeros((nose_fear, 64), np.uint8)

k = 0
for i in range(0,heightf-8,8):
    for j in range(0,widthf-8,8):
        crop_tmpf3 = nosef1[i:i+8, j:j+8]
        flatnosef[k,0:64] = crop_tmpf3.flatten()
        k = k + 1
fspaceNoseF = pd.DataFrame(flatnosef)  
fspaceNoseF[64] = 'Nose'
fspaceNoseF[65] = 'Fear'


fspaceNoseF.to_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/testing/fspaceANoseF3.csv', index=False)



# =============================================================================
# Code For MOUTH - HAPPY
# =============================================================================

mouth_cascade = cv2.CascadeClassifier('C:/Users/aishg/anaconda3/envs/project1/Library/etc/haarcascades/haarcascade_mcs_mouth.xml')

face_cascade = cv2.CascadeClassifier('C:/Users/aishg/anaconda3/envs/project1/Library/etc/haarcascades/haarcascade_frontalface_default.xml')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:/Users/aishg/anaconda3/envs/project/lib/site-packages/face_recognition_models/models/shape_predictor_81_face_landmarks.dat")


fear = cv2.imread("C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/Dataset/Expression Fear/IMG_5207.JPG")
gray = cv2.cvtColor(fear, cv2.COLOR_BGR2GRAY)

temp5 = cv2.imread("C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/Dataset/Expression Fear/IMG_5207.JPG")

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
        fearmouth = temp5[y1:y2, x1:x2]
      #  face_file_name = "face/" + str(y) + ".jpg"
       # cv2.imwrite("C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/seperate features/ForeheadD.jpg", disghead)

        plt.imshow(fearmouth)    

fearmouth = cv2.cvtColor(fearmouth, cv2.COLOR_BGR2GRAY)

heightfm, widthfm = fearmouth.shape

mouthf = cv2.resize(fearmouth, dsize=(80, 48), interpolation=cv2.INTER_CUBIC)

plt.imshow(mouthf, cmap=plt.get_cmap('gray'))

mouthf = cv2.normalize(mouthf.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)*255

heightmf, widthmf = mouthf.shape

plt.imshow(mouthf, cmap=plt.get_cmap('gray'))

mouth_fear = round(((heightmf-8)*(widthmf-8))/64)
flatmouthf = np.zeros((mouth_fear, 64), np.uint8)
k = 0
for i in range(0,heightmf-8,8):
    for j in range(0,widthmf-8,8):
        crop_tmpf4 = mouthf[i:i+8,j:j+8]
        flatmouthf[k,0:64] = crop_tmpf4.flatten()
        k = k + 1
fspaceMouthF = pd.DataFrame(flatmouthf)  
fspaceMouthF[64] = 'Mouth'
fspaceMouthF[65] = 'Fear'


fspaceMouthF.to_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/testing/fspaceMouthF3.csv', index=False)


# =============================================================================
# Forehead detection!!!!!!!
# =============================================================================


face_cascade = cv2.CascadeClassifier('C:/Users/aishg/anaconda3/envs/project1/Library/etc/haarcascades/haarcascade_frontalface_default.xml')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:/Users/aishg/anaconda3/envs/project/lib/site-packages/face_recognition_models/models/shape_predictor_81_face_landmarks.dat")


fear = cv2.imread("C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/Dataset/Expression Fear/IMG_5207.JPG")
gray = cv2.cvtColor(fear, cv2.COLOR_BGR2GRAY)

temp5 = cv2.imread("C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/Dataset/Expression Fear/IMG_5207.JPG")



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
        fear = temp5[y1:y2, x1:x2]
        plt.imshow(fear)
     
headF2 = cv2.cvtColor(fear, cv2.COLOR_BGR2GRAY)
heightcf, widthcf = headF2.shape

headhF = cv2.resize(headF2, dsize=(152, 64), interpolation=cv2.INTER_CUBIC)

headf = cv2.normalize(headhF.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)*255

heightf, widthf = headf.shape

plt.imshow(headf, cmap=plt.get_cmap('gray'))
plt.axis('off')

head_fear = round(((heightf-8)*(widthf-8))/64)
flatheadfear = np.zeros((head_fear, 64), np.uint8)

k = 0
for i in range(0,heightf-8,8):
    for j in range(0,widthf-8,8):
        crop_tmpf4 = headf[i:i+8,j:j+8]
        flatheadfear[k,0:64] = crop_tmpf4.flatten()
        k = k + 1
fspaceHeadfear = pd.DataFrame(flatheadfear)  
fspaceHeadfear[64] = 'Forehead'
fspaceHeadfear[65] = 'Fear'


fspaceHeadfear.to_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/testing/fspaceHeadFear3.csv', index=False)

# =============================================================================
# 
# =============================================================================


tmp = cv2.imread('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fear/fear (5).jpg')
img = cv2.imread('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fear/fear (5).jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#plt.imshow(tmp)
# =============================================================================
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
#                  cv2.imwrite("C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/seperate features/chinF.jpg", chin)
# 
# 
# chinF = cv2.imread("C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/seperate features/chinF.jpg")
# 
# chinF2 = cv2.cvtColor(chinF, cv2.COLOR_BGR2GRAY)
# heightcf, widthcf = chinF2.shape
# 
# chinfF = cv2.resize(chinF2, dsize=(240, 152), interpolation=cv2.INTER_CUBIC)
# 
# plt.imshow(chinfF, cmap=plt.get_cmap('gray'))
# 
# chinf = cv2.normalize(chinfF.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)*255
# 
# heightf, widthf = chinf.shape
# 
# plt.imshow(chinf, cmap=plt.get_cmap('gray'))
# plt.axis('off')
#  
# import pandas as pd
# import numpy as np
# 
# chin_fear = round(((heightf-8)*(widthf-8))/64)
# flatchinf = np.zeros((chin_fear, 66), np.uint8)
# 
# k = 0
# for i in range(0,heightf-8,8):
#     for j in range(0,widthf-8,8):
#        crop_tmp5 = chinf[i:i+8,j:j+8]
#        flatchinf[k,0:64] = crop_tmp5.flatten()
#        k = k + 1
# fspaceChinf = pd.DataFrame(flatchinf) 
# 
# chinF = fspaceChinf[~np.all(fspaceChinf == 0, axis=1)]
# chinF[64]=5
# chinF[65]=4
#  
# chinF.to_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/fspaceChinF.csv', index=False)
# 
# =============================================================================

framesfear = [fspaceHeadfear, fspaceMouthF, fspaceNoseF, rndmged]
framestogetherf = pd.concat(framesfear)
indx4 = np.arange(len(framestogetherf))
fearExp = np.random.permutation(indx4)
fearExp = framestogetherf.sample(frac=1).reset_index(drop=True)
fearExp.to_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/testing/fearface3.csv', index=False)

# =============================================================================

f1 = pd.read_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/testing/fearface.csv')
f2 = pd.read_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/testing/fearface2.csv')
f3 = pd.read_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/testing/fearface3.csv')



ff = [f1, f2, f3]
ff1 = pd.concat(ff)
indxfear = np.arange(len(ff1))
fearExp = np.random.permutation(indxfear)
fearExp = ff1.sample(frac=1).reset_index(drop=True)
fearExp.to_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/testing/FearExpressionsP123.csv', index=False)
