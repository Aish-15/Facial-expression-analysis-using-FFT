# -*- coding: utf-8 -*-
"""

@author: aishg
"""

#Code together for surprised



# =============================================================================
# We detect the eyes in the face and extract to a csv file in a surprised face
# =============================================================================


import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import face_recognition
import dlib



face_cascade = cv2.CascadeClassifier('C:/Users/aishg/anaconda3/envs/project1/Library/etc/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:/Users/aishg/anaconda3/envs/project1/Library/etc/haarcascades/haarcascade_eye.xml')

surp = cv2.imread("C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/Dataset/Expression Surprised/IMG_5206.JPG")
gray = cv2.cvtColor(surp, cv2.COLOR_BGR2GRAY)
plt.imshow(gray, cmap=plt.get_cmap('gray'))

faces = face_cascade.detectMultiScale(gray, 1.3, 5)

for (x,y,w,h) in faces:
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = surp[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(gray, 1.33, 10)
    eyer = eye_cascade.detectMultiScale(gray, 1.4, 5)
    
    
# =============================================================================
# #Detect the eyes after the face has been detected
# =============================================================================

    for (ex,ey,ew,eh) in eyes:
        sub_face = gray[ey:ey+eh, ex:ex+ew]
        plt.imshow(sub_face, cmap=plt.get_cmap('gray'))
       # face_file_name = "face/" + str(y) + ".jpg"
        #cv2.imwrite("C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/seperate features/eyeleftfaceSurp.jpg", sub_face)


    for (ex,ey,ew,eh) in eyer:
        sub_face2 = gray[ey:ey+eh, ex:ex+ew]
        plt.imshow(sub_face2, cmap=plt.get_cmap('gray'))
 #       face_file_name = "face/" + str(y) + ".jpg"
  #      cv2.imwrite("C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/seperate features/eyerightfaceSurp.jpg", sub_face2)


heightcL, widthcL = sub_face.shape

eyeLft = cv2.resize(sub_face, dsize=(72, 72), interpolation=cv2.INTER_CUBIC)

plt.imshow(eyeLft, cmap=plt.get_cmap('gray'))

eyeLeft = cv2.normalize(eyeLft.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)*255

heightLc, widthLc = eyeLeft.shape

plt.imshow(eyeLeft, cmap=plt.get_cmap('gray'))
plt.axis('off')



eye_surp_left = round(((heightLc-8)*(widthLc-8))/64)
flateyeL = np.zeros((eye_surp_left, 64), np.uint8)

k = 0
for i in range(0,heightLc-8,8):
    for j in range(0,widthLc-8,8):
        crop_tmpL1 = eyeLeft[i:i+8,j:j+8]
        flateyeL[k,0:64] = crop_tmpL1.flatten()
        k = k + 1
fspaceEyeLeft = pd.DataFrame(flateyeL)  
fspaceEyeLeft[64] = 'LeftEye'
fspaceEyeLeft[65] = 'Surprised'


fspaceEyeLeft.to_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/testing/fspaceAEyesLSurp3.csv', index=False)


heightcR, widthcR = sub_face2.shape

eyeRgt = cv2.resize(sub_face2, dsize=(80, 72), interpolation=cv2.INTER_CUBIC)

plt.imshow(eyeRgt, cmap=plt.get_cmap('gray'))

eyeRight = cv2.normalize(eyeRgt.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)*255

heightRc, widthRc = eyeRight.shape

plt.imshow(eyeRight, cmap=plt.get_cmap('gray'))
plt.axis('off')



eye_surp_right = round(((heightRc-8)*(widthRc-8))/64)
flateyeR = np.zeros((eye_surp_right, 64), np.uint8)

k = 0
for i in range(0,heightRc-8,8):
    for j in range(0,widthRc-8,8):
        crop_tmpR1 = eyeRight[i:i+8,j:j+8]
        flateyeR[k,0:64] = crop_tmpR1.flatten()
        k = k + 1
fspaceEyeRight = pd.DataFrame(flateyeR)  
fspaceEyeRight[64] = 'RightEye'
fspaceEyeRight[65] = 'Surprised'


fspaceEyeRight.to_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/testing/fspaceAEyesRSurp3.csv', index=False)


frames = [fspaceEyeLeft, fspaceEyeRight]
mged = pd.concat(frames)
indx = np.arange(len(mged))
rndmged = np.random.permutation(indx)
rndmged=mged.sample(frac=1).reset_index(drop=True)
rndmged.to_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/testing/EyeSurpLR3.csv', index=False)


# ============================================================================= 
# Code for nose identification
# =============================================================================

nose_cascade = cv2.CascadeClassifier('/Users/aishg/anaconda3/envs/project1/Library/etc/haarcascades/haarcascade_mcs_nose.xml')
#nose classifier: https://sourceforge.net/p/emgucv/opencv/ci/d58cd9851fdb592a395488fc5721d11c7436a99c/tree/data/haarcascades/

for (x,y,w,h) in faces:
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = surp[y:y+h, x:x+w]
    noseSurp =  nose_cascade.detectMultiScale(gray, 1.2, 12)


    for (nx, ny, nw, nh) in noseSurp:
        nose_faceSur = gray[ny:ny+nh, nx:nx+nw]
        #face_file_name = "face/" + str(y) + ".jpg"
        #cv2.imwrite("C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/seperate features/noseSurp.jpg", nose_faceSur)
        plt.imshow(nose_faceSur, cmap=plt.get_cmap('gray'))

heightcS, widthcS = nose_faceSur.shape #cG

noseSurp = cv2.resize(nose_faceSur, dsize=(72, 72), interpolation=cv2.INTER_CUBIC)

plt.imshow(noseSurp, cmap=plt.get_cmap('gray'))

nosesurp1 = cv2.normalize(noseSurp.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)*255
heightsurp, widthsurp = nosesurp1.shape

plt.imshow(nosesurp1, cmap=plt.get_cmap('gray'))

nose_surp = round(((heightsurp-8)*(widthsurp-8))/64)
flatnosesurp = np.zeros((nose_surp, 64), np.uint8)

k = 0
for i in range(0,heightsurp-8,8):
    for j in range(0,widthsurp-8,8):
        crop_tmps3 = nosesurp1[i:i+8, j:j+8]
        flatnosesurp[k,0:64] = crop_tmps3.flatten()
        k = k + 1
fspaceNoseSurp = pd.DataFrame(flatnosesurp)  
fspaceNoseSurp[64] = 'Nose'
fspaceNoseSurp[65] = 'Surprised'


fspaceNoseSurp.to_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/testing/fspaceANoseSurp3.csv', index=False)


# =============================================================================
# Code For MOUTH - HAPPY
# =============================================================================

#mouth_cascace: https://github.com/sightmachine/SimpleCV/blob/master/SimpleCV/Features/HaarCascades/mouth.xml
mouth_cascade = cv2.CascadeClassifier('C:/Users/aishg/anaconda3/envs/project1/Library/etc/haarcascades/haarcascade_mcs_mouth.xml')

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:/Users/aishg/anaconda3/envs/project/lib/site-packages/face_recognition_models/models/shape_predictor_81_face_landmarks.dat")


surp = cv2.imread("C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/Dataset/Expression Surprised/IMG_5206.JPG")
gray = cv2.cvtColor(surp, cv2.COLOR_BGR2GRAY)
temp5 = cv2.imread("C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/Dataset/Expression Surprised/IMG_5206.JPG")

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
        surpmouth = temp5[y1:y2, x1:x2]
      #  face_file_name = "face/" + str(y) + ".jpg"
       # cv2.imwrite("C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/seperate features/ForeheadD.jpg", disghead)

        plt.imshow(surpmouth)    


surpmouth = cv2.cvtColor(surpmouth, cv2.COLOR_BGR2GRAY)
heightsurpm, widthsurpm = surpmouth.shape

mouthsurp = cv2.resize(surpmouth, dsize=(80, 48), interpolation=cv2.INTER_CUBIC)

plt.imshow(mouthsurp, cmap=plt.get_cmap('gray'))

mouthsurp = cv2.normalize(mouthsurp.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)*255

heightmsurp, widthmsurp = mouthsurp.shape

plt.imshow(mouthsurp, cmap=plt.get_cmap('gray'))

mouth_surp = round(((heightmsurp-8)*(widthmsurp-8))/64)
flatmouthsurp = np.zeros((mouth_surp, 64), np.uint8)

k = 0
for i in range(0,heightmsurp-8,8):
    for j in range(0,widthmsurp-8,8):
        crop_tmpsurp4 = mouthsurp[i:i+8,j:j+8]
        flatmouthsurp[k,0:64] = crop_tmpsurp4.flatten()
        k = k + 1
fspaceMouthSurp = pd.DataFrame(flatmouthsurp)  
fspaceMouthSurp[64] = 'Mouth'
fspaceMouthSurp[65] = 'Surprised'

fspaceMouthSurp.to_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/testing/fspaceMouthSurp3.csv', index=False)



# =============================================================================
# Forehead detection!!!!!!!
# =============================================================================


face_cascade = cv2.CascadeClassifier('C:/Users/aishg/anaconda3/envs/project1/Library/etc/haarcascades/haarcascade_frontalface_default.xml')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:/Users/aishg/anaconda3/envs/project/lib/site-packages/face_recognition_models/models/shape_predictor_81_face_landmarks.dat")

surp = cv2.imread("C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/Dataset/Expression Surprised/IMG_5206.JPG")
gray = cv2.cvtColor(surp, cv2.COLOR_BGR2GRAY)
temp5 = cv2.imread("C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/Dataset/Expression Surprised/IMG_5206.JPG")

faces = detector(gray)
plt.imshow(gray)

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

        surphead = temp5[y1:y2, x1:x2]
        plt.imshow(surphead)
        #face_file_name = "face/" + str(y) + ".jpg"
        #cv2.imwrite("C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/seperate features/ForeheadSurp.jpg", surphead)

headS2 = cv2.cvtColor(surphead, cv2.COLOR_BGR2GRAY)
heightcs, widthcs = headS2.shape

headhS = cv2.resize(headS2, dsize=(136, 56), interpolation=cv2.INTER_CUBIC)

plt.imshow(headhS, cmap=plt.get_cmap('gray'))

heads = cv2.normalize(headhS.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)*255

heights, widths = heads.shape

plt.imshow(heads, cmap=plt.get_cmap('gray'))
plt.axis('off')

head_surp = round(((heights-8)*(widths-8))/64)
flatheadsurp = np.zeros((head_surp, 64), np.uint8)

k = 0
for i in range(0,heights-8,8):
    for j in range(0,widths-8,8):
        crop_tmps4 = heads[i:i+8,j:j+8]
        flatheadsurp[k,0:64] = crop_tmps4.flatten()
        k = k + 1
fspaceHeadsurp = pd.DataFrame(flatheadsurp)  
fspaceHeadsurp[64] = 'Forehead'
fspaceHeadsurp[65] = 'Surprised'


fspaceHeadsurp.to_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/testing/fspaceHeadsurp3.csv', index=False)

# =============================================================================
# 
# =============================================================================


tmp = cv2.imread('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/Surprised/surprised (1).jpg')
img = cv2.imread('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/Surprised/surprised (1).jpg')
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
             
             for n in range(2, 15):
                 x = landmarks.part(n).x
                 y = landmarks.part(n).y
                 
                 x_pts.append(x)
                 y_pts.append(y)
                 
                 x1 = min(x_pts)
                 x2 = max(x_pts)
                 y1 = min(y_pts)
                 y2 = max(y_pts)

                 cv2.rectangle(gray, (x1,y1), (x2,y2), (255, 255, 255), 10)
                 chin = img[y1:y2, x1:x2]
                 plt.imshow(chin)
                 cv2.imwrite("C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/seperate features/chinSurp.jpg", chin)


chinSurp = cv2.imread("C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/seperate features/chinSurp.jpg")

chinSurp2 = cv2.cvtColor(chinSurp, cv2.COLOR_BGR2GRAY)
heightcsurp, widthcsurp = chinSurp2.shape

chinsSurp = cv2.resize(chinSurp2, dsize=(240, 152), interpolation=cv2.INTER_CUBIC)

#plt.imshow(chinsSurp, cmap=plt.get_cmap('gray'))

chinsurp = cv2.normalize(chinsSurp.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)*255

heightsurp, widthsurp = chinsurp.shape

#plt.imshow(chinsurp, cmap=plt.get_cmap('gray'))
#plt.axis('off')
 
import pandas as pd
import numpy as np

chin_surp = round(((heightsurp-8)*(widthsurp-8))/64)
flatchinsurp = np.zeros((chin_surp, 66), np.uint8)

k = 0
for i in range(0,heightsurp-8,8):
    for j in range(0,widthsurp-8,8):
       crop_tmp5 = chinsurp[i:i+8,j:j+8]
       flatchinsurp[k,0:64] = crop_tmp5.flatten()
       k = k + 1
fspaceChinsurp = pd.DataFrame(flatchinsurp)  
chinSurp = fspaceChinsurp[~np.all(fspaceChinsurp == 0, axis=1)]
chinSurp[64]=5
chinSurp[65]=5

chinSurp.to_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/fspaceChinSurp.csv', index=False)


# =============================================================================
# 
# =============================================================================

framesurp = [rndmged, fspaceNoseSurp, fspaceMouthSurp, fspaceHeadsurp]
framestogethers = pd.concat(framesurp)
indx5 = np.arange(len(framestogethers))
surpExp = np.random.permutation(indx5)
surpExp = framestogethers.sample(frac=1).reset_index(drop=True)
surpExp.to_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/testing/Surpface3.csv', index=False)


s1 = pd.read_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/testing/Surpface.csv')
s2 = pd.read_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/testing/Surpface3.csv')
s3 = pd.read_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/testing/Surpface3.csv')



ss = [s1, s2, s3]
ss1 = pd.concat(ss)
indxsur = np.arange(len(ss1))
surExp = np.random.permutation(indxsur)
surExp = ss1.sample(frac=1).reset_index(drop=True)
surExp.to_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/testing/Surpface123.csv', index=False)


