# -*- coding: utf-8 -*-
"""

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

img = cv2.imread('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/original features/happy/happy (1).jpg')
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
        sub_face = gray[ey:ey+eh, ex:ex+ew]
        sub_face = cv2.Sobel(sub_face,cv2.CV_64F,0,1,ksize=5)  
        plt.imshow(sub_face, cmap=plt.get_cmap('gray'))

      #  face_file_name = "face/" + str(y) + ".jpg"
       # cv2.imwrite("C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/seperate features/eyeleftfaceH.jpg", sub_face)


    for (ex,ey,ew,eh) in eyer:
        sub_face2 = gray[ey:ey+eh, ex:ex+ew]
        sub_face2 = cv2.Sobel(sub_face2,cv2.CV_64F,0,1,ksize=5)  

        plt.imshow(sub_face2, cmap=plt.get_cmap('gray'))
      #  face_file_name = "face/" + str(y) + ".jpg"
       # cv2.imwrite("C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/seperate features/eyerightfaceH.jpg", sub_face2)




#eyeL = cv2.imread("C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/seperate features/eyeleftfaceH.jpg")


eyeLft = cv2.resize(sub_face, dsize=(72, 72), interpolation=cv2.INTER_CUBIC)

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

fspaceEyeLeft.to_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/original features/happy/fspaceAEyesLH.csv', index=False)



eyeRgt = cv2.resize(sub_face2, dsize=(72, 72), interpolation=cv2.INTER_CUBIC)

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

fspaceEyeRight.to_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/original features/happy/fspaceAEyesRH.csv', index=False)


EyeLeft = pd.read_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/fspaceAEyesLH.csv')
Eyeright = pd.read_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/fspaceAEyesRH.csv')


frames = [fspaceEyeLeft, fspaceEyeRight]
mged = pd.concat(frames)
#indx = np.arange(len(mged))
#fspaceeye = np.random.permutation(indx)
#fspaceeye=mged.sample(frac=1).reset_index(drop=True)
mged.to_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/original features/happy/EyeHappyLRJustMerged.csv', index=False)

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

# =============================================================================
# 
# 
# =============================================================================
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

Nose = pd.read_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/fspaceANoseH.csv')
Mouth = pd.read_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/fspaceMouthH.csv')
Eyes = pd.read_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/original features/happy/EyeHappyLRJustMerged.csv')
head = pd.read_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/fspaceHeadH.csv')
#chin = pd.read_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/fspaceChinH.csv')

framesh = [Eyes, Nose, Mouth, head]
framestogether = pd.concat(framesh)
indx1 = np.arange(len(framestogether))
happyExp = np.random.permutation(indx1)
happyExp = framestogether.sample(frac=1).reset_index(drop=True)
happyExp.to_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/original features/happy/Happyface.csv', index=False)


# -*- coding: utf-8 -*-
"""

@author: aishg
"""

#Code together for disgusted



# =============================================================================
# We detect the eyes in the face and extract to a csv file in a Disgusted face
# =============================================================================


disg = cv2.imread('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/Dataset/Expression Disgusted/disgusted.JPG')
gray = cv2.cvtColor(disg, cv2.COLOR_BGR2GRAY)
plt.imshow(gray, cmap=plt.get_cmap('gray'))

faces = face_cascade.detectMultiScale(gray, 1.3, 5)

for (x,y,w,h) in faces:
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = disg[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(gray, 1.2, 10)
    eyer = eye_cascade.detectMultiScale(gray, 1.7, 10)
    
    
# =============================================================================
# #Detect the eyes after the face has been detected
# =============================================================================

    for (ex,ey,ew,eh) in eyes:
        sub_face = gray[ey:ey+eh, ex:ex+ew]
        sub_face = cv2.Sobel(sub_face,cv2.CV_64F,0,1,ksize=5)  
       # face_file_name = "face/" + str(y) + ".jpg"
        #cv2.imwrite("C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/seperate features/eyeleftfaceD.jpg", sub_face)


    for (ex,ey,ew,eh) in eyer:
        sub_face2 = gray[ey:ey+eh, ex:ex+ew]
        sub_face2 = cv2.Sobel(sub_face2,cv2.CV_64F,0,1,ksize=5)  

       # face_file_name = "face/" + str(y) + ".jpg"
        #cv2.imwrite("C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/seperate features/eyerightfaceD.jpg", sub_face2)

eyeLft = cv2.resize(sub_face, dsize=(72, 72), interpolation=cv2.INTER_CUBIC)

eyeLeft = cv2.normalize(eyeLft.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)*255

heightLc, widthLc = eyeLeft.shape

eye_disg_left = round(((heightLc-8)*(widthLc-8))/64)
flateyeL = np.zeros((eye_disg_left, 66), np.uint8)+2
flateyeL[:,65] = 3
flateyeL[:,64] = 10

k = 0
for i in range(0,heightLc-8,8):
    for j in range(0,widthLc-8,8):
        crop_tmpL1 = eyeLeft[i:i+8,j:j+8]
        flateyeL[k,0:64] = crop_tmpL1.flatten()
        k = k + 1
fspaceEyeLeft = pd.DataFrame(flateyeL)  

#fspaceEyeLeft.to_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/fspaceAEyesLD.csv', index=False)

eyeRgt = cv2.resize(sub_face2, dsize=(72, 72), interpolation=cv2.INTER_CUBIC)
eyeRight = cv2.normalize(eyeRgt.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)*255
heightRc, widthRc = eyeRight.shape

eye_disg_right = round(((heightRc-8)*(widthRc-8))/64)
flateyeR = np.zeros((eye_disg_right, 66), np.uint8)+2
flateyeR[:,65] = 3
flateyeR[:,64] = 11

k = 0
for i in range(0,heightRc-8,8):
    for j in range(0,widthRc-8,8):
        crop_tmpR1 = eyeRight[i:i+8,j:j+8]
        flateyeR[k,0:64] = crop_tmpR1.flatten()
        k = k + 1
fspaceEyeRight = pd.DataFrame(flateyeR)  

#fspaceEyeRight.to_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/fspaceAEyesRD.csv', index=False)


frames = [fspaceEyeLeft, fspaceEyeRight]
mged = pd.concat(frames)
indx = np.arange(len(mged))
rndmged = np.random.permutation(indx)
rndmged_disg=mged.sample(frac=1).reset_index(drop=True)
#mged.to_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/EyeDJustMerged.csv', index=False)

# =============================================================================
# 
# =============================================================================

Nosedisg = pd.read_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/fspaceANoseD.csv')
Mouthdisg = pd.read_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/fspaceMouthD.csv')
Eyedisg = pd.read_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/EyeDJustMerged.csv')
headdisg = pd.read_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/fspaceHeadD.csv')
chin = pd.read_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/fspaceChinD.csv')


framesD = [rndmged_disg, Nosedisg, Mouthdisg,  headdisg]
mgeddisg = pd.concat(framesD)
indxdisg = np.arange(len(mgeddisg))
disgExp = np.random.permutation(indxdisg)
disgExp = mgeddisg.sample(frac=1).reset_index(drop=True)

# =============================================================================
# 
# =============================================================================

sad = cv2.imread('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/Dataset/Expression Sad/IMG_5135.jpg')
gray = cv2.cvtColor(sad, cv2.COLOR_BGR2GRAY)


faces = face_cascade.detectMultiScale(gray, 1.3, 5)

# =============================================================================
# #Detect the face in the picture
# =============================================================================

for (x,y,w,h) in faces:
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = sad[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(gray, 1.2, 10)
    eyer = eye_cascade.detectMultiScale(gray, 1.5, 10)
    
    
# =============================================================================
# #Detect the eyes after the face has been detected
# =============================================================================

    for (ex,ey,ew,eh) in eyes:
        sad_face = gray[ey:ey+eh, ex:ex+ew]
        sad_face = cv2.Sobel(sad_face,cv2.CV_64F,0,1,ksize=5)  

        plt.imshow(sad_face, cmap=plt.get_cmap('gray'))
#        face_file_name = "face/" + str(y) + ".jpg"
#        cv2.imwrite("C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/seperate features/eyeleftfaceS.jpg", sad_face)
#    plt.imshow(sad_face)


    for (ex,ey,ew,eh) in eyer:
        sad_face2 = gray[ey:ey+eh, ex:ex+ew]
        sad_face2 = cv2.Sobel(sad_face2,cv2.CV_64F,0,1,ksize=5)  

#        plt.imshow(sad_face2, cmap=plt.get_cmap('gray'))
 #       face_file_name = "face/" + str(y) + ".jpg"
 #      cv2.imwrite("C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/seperate features/eyerightfaceS.jpg", sad_face2)
#plt.imshow(sad_face2)
        

eyeLft = cv2.resize(sad_face, dsize=(72, 72), interpolation=cv2.INTER_CUBIC)

eyeLeft = cv2.normalize(eyeLft.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)*255

heightLc, widthLc = eyeLeft.shape

eye_happy_left = round(((heightLc-8)*(widthLc-8))/64)
flateyeL = np.zeros((eye_happy_left, 66), np.uint8)+2
flateyeL[:,65] = 2
flateyeL[:,64] = 10

k = 0
for i in range(0,heightLc-8,8):
    for j in range(0,widthLc-8,8):
        crop_tmpL1 = eyeLeft[i:i+8,j:j+8]
        flateyeL[k,0:64] = crop_tmpL1.flatten()
        k = k + 1
fspaceEyeLeft = pd.DataFrame(flateyeL)  

#fspaceEyeLeft.to_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/fspaceAEyesLS.csv', index=False)


eyeRgt = cv2.resize(sad_face2, dsize=(72, 72), interpolation=cv2.INTER_CUBIC)

eyeRight = cv2.normalize(eyeRgt.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)*255

heightRc, widthRc = eyeRight.shape
eye_sad_right = round(((heightRc-8)*(widthRc-8))/64)
flateyeR = np.zeros((eye_sad_right, 66), np.uint8)+2
flateyeR[:,65] = 2
flateyeR[:,64] = 11

k = 0
for i in range(0,heightRc-8,8):
    for j in range(0,widthRc-8,8):
        crop_tmpR1 = eyeRight[i:i+8,j:j+8]
        flateyeR[k,0:64] = crop_tmpR1.flatten()
        k = k + 1
fspaceEyeRight = pd.DataFrame(flateyeR)  

#fspaceEyeRight.to_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/fspaceAEyesRS.csv', index=False)

#EyeLeft = pd.read_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/fspaceAEyesLS.csv')
#Eyeright = pd.read_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/fspaceAEyesRS.csv')


frames = [fspaceEyeLeft, fspaceEyeRight]
merge = pd.concat(frames)
indx = np.arange(len(mged))
fspaceeye = np.random.permutation(indx)
fspaceeye_sad=mged.sample(frac=1).reset_index(drop=True)
#merge.to_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/EyeSadLRjustMerge.csv', index=False)

Nosesad = pd.read_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/fspaceANoseS.csv')
Mouthsad = pd.read_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/fspaceMouthS.csv')
#Eyessad = pd.read_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/EyeSadLRjustMerge.csv')
headsad = pd.read_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/fspaceHeadS.csv')
chin = pd.read_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/fspaceChinS.csv')


framesS = [fspaceeye_sad, Nosesad, Mouthsad,  headsad, chin]
mgedsad = pd.concat(framesS)
indxsad = np.arange(len(mgedsad))
sadExp = np.random.permutation(indxsad)
sadExp = mgedsad.sample(frac=1).reset_index(drop=True)
#sadExp.to_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/SadExpressionsP.csv', index=False)

# =============================================================================
# 
# =============================================================================


fear = cv2.imread('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/Dataset/Expression fear/IMG_5149.jpg')
gray = cv2.cvtColor(fear, cv2.COLOR_BGR2GRAY)
#plt.imshow(gray, cmap=plt.get_cmap('gray'))

faces = face_cascade.detectMultiScale(gray, 1.3, 5)

for (x,y,w,h) in faces:
  #  cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = fear[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(gray, 1.2, 10)
    eyer = eye_cascade.detectMultiScale(gray, 1.9, 10)
    
    
# =============================================================================
# #Detect the eyes after the face has been detected
# =============================================================================

    for (ex,ey,ew,eh) in eyes:
        sub_face = gray[ey:ey+eh, ex:ex+ew]
        sub_face = cv2.Sobel(sub_face,cv2.CV_64F,0,1,ksize=5)  
        plt.imshow(sub_face, cmap=plt.get_cmap('gray'))
    #    face_file_name = "face/" + str(y) + ".jpg"
     #   cv2.imwrite("C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/seperate features/eyeleftfaceF.jpg", sub_face)


    for (ex,ey,ew,eh) in eyer:
        sub_face2 = gray[ey:ey+eh, ex:ex+ew]
        sub_face2 = cv2.Sobel(sub_face2,cv2.CV_64F,0,1,ksize=5)  

eyeLft = cv2.resize(roi_gray, dsize=(32, 32), interpolation=cv2.INTER_CUBIC)
plt.imshow(eyeLft, cmap=plt.get_cmap('gray'))
plt.axis('off')

eyeLeft = cv2.normalize(eyeLft.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)*255

heightLc, widthLc = eyeLeft.shape

eye_fear_left = round(((heightLc-8)*(widthLc-8))/64)
flateyeL = np.zeros((eye_fear_left, 66), np.uint8)+2
flateyeL[:,65] = 4
flateyeL[:,64] = 10

k = 0
for i in range(0,heightLc-8,8):
    for j in range(0,widthLc-8,8):
        crop_tmpL1 = eyeLeft[i:i+8,j:j+8]
        flateyeL[k,0:64] = crop_tmpL1.flatten()
        k = k + 1
fspaceEyeLeft = pd.DataFrame(flateyeL)  


eyeRgt = cv2.resize(sub_face2, dsize=(80, 72), interpolation=cv2.INTER_CUBIC)

eyeRight = cv2.normalize(eyeRgt.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)*255

heightRc, widthRc = eyeRight.shape

eye_fear_right = round(((heightRc-8)*(widthRc-8))/64)
flateyeR = np.zeros((eye_fear_right, 66), np.uint8)+2
flateyeR[:,65] = 4
flateyeR[:,64] = 11

k = 0
for i in range(0,heightRc-8,8):
    for j in range(0,widthRc-8,8):
        crop_tmpR1 = eyeRight[i:i+8,j:j+8]
        flateyeR[k,0:64] = crop_tmpR1.flatten()
        k = k + 1
fspaceEyeRight = pd.DataFrame(flateyeR)  

#fspaceEyeRight.to_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/fspaceAEyesRF.csv', index=False)


frames = [fspaceEyeLeft, fspaceEyeRight]
mged = pd.concat(frames)
indx = np.arange(len(mged))
rndmged = np.random.permutation(indx)
rndmged_fear=mged.sample(frac=1).reset_index(drop=True)


Nose = pd.read_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/fspaceANoseF.csv')
Mouth = pd.read_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/fspaceMouthF.csv')
head = pd.read_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/fspaceHeadFear.csv')
chin = pd.read_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/fspaceChinF.csv')

framesfear = [rndmged_fear, Nose, Mouth, head]
framestogetherf = pd.concat(framesfear)
indx4 = np.arange(len(framestogetherf))
fearExp = np.random.permutation(indx4)
fearExp = framestogetherf.sample(frac=1).reset_index(drop=True)


# =============================================================================
# 
# =============================================================================


surp = cv2.imread('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/Dataset/Expression Surprised/IMG_5165.jpg')
gray = cv2.cvtColor(surp, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = surp[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(gray, 1.2, 10)
    eyer = eye_cascade.detectMultiScale(gray, 1.9, 10)    
# =============================================================================
# #Detect the eyes after the face has been detected
# =============================================================================

    for (ex,ey,ew,eh) in eyes:
        sub_face = gray[ey:ey+eh, ex:ex+ew]
        sub_face = cv2.Sobel(sub_face,cv2.CV_64F,0,1,ksize=5)  

        plt.imshow(sub_face, cmap=plt.get_cmap('gray'))
       # face_file_name = "face/" + str(y) + ".jpg"
        #cv2.imwrite("C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/seperate features/eyeleftfaceSurp.jpg", sub_face)
    for (ex,ey,ew,eh) in eyer:
        sub_face2 = gray[ey:ey+eh, ex:ex+ew]
        sub_face2 = cv2.Sobel(sub_face2,cv2.CV_64F,0,1,ksize=5)  

        plt.imshow(sub_face2, cmap=plt.get_cmap('gray'))
 #       face_file_name = "face/" + str(y) + ".jpg"
  #      cv2.imwrite("C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/seperate features/eyerightfaceSurp.jpg", sub_face2)
eyeLft = cv2.resize(sub_face, dsize=(72, 72), interpolation=cv2.INTER_CUBIC)
eyeLeft = cv2.normalize(eyeLft.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)*255

heightLc, widthLc = eyeLeft.shape

eye_surp_left = round(((heightLc-8)*(widthLc-8))/64)
flateyeL = np.zeros((eye_surp_left, 66), np.uint8)+2
flateyeL[:,65] = 5
flateyeL[:,64] = 10

k = 0
for i in range(0,heightLc-8,8):
    for j in range(0,widthLc-8,8):
        crop_tmpL1 = eyeLeft[i:i+8,j:j+8]
        flateyeL[k,0:64] = crop_tmpL1.flatten()
        k = k + 1
fspaceEyeLeft = pd.DataFrame(flateyeL)  

#fspaceEyeLeft.to_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/fspaceAEyesLSurp.csv', index=False)


eyeRgt = cv2.resize(sub_face2, dsize=(80, 72), interpolation=cv2.INTER_CUBIC)
eyeRight = cv2.normalize(eyeRgt.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)*255
heightRc, widthRc = eyeRight.shape
eye_surp_right = round(((heightRc-8)*(widthRc-8))/64)
flateyeR = np.zeros((eye_surp_right, 66), np.uint8)+2
flateyeR[:,65] = 5
flateyeR[:,64] = 11

k = 0
for i in range(0,heightRc-8,8):
    for j in range(0,widthRc-8,8):
        crop_tmpR1 = eyeRight[i:i+8,j:j+8]
        flateyeR[k,0:64] = crop_tmpR1.flatten()
        k = k + 1
fspaceEyeRight = pd.DataFrame(flateyeR)  

#fspaceEyeRight.to_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/fspaceAEyesRSurp.csv', index=False)


frames = [fspaceEyeLeft, fspaceEyeRight]
mged = pd.concat(frames)
indx = np.arange(len(mged))
rndmged = np.random.permutation(indx)
rndmged=mged.sample(frac=1).reset_index(drop=True)

Nose = pd.read_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/fspaceANoseSurp.csv')
Mouth = pd.read_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/fspaceMouthSurp.csv')
head = pd.read_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/fspaceHeadsurp.csv')

framesurp = [rndmged, Nose, Mouth, head]
framestogethers = pd.concat(framesurp)
indx5 = np.arange(len(framestogethers))
surpExp = np.random.permutation(indx5)
surpExp = framestogethers.sample(frac=1).reset_index(drop=True)
# =============================================================================
# 
# =============================================================================

allexp = [happyExp, sadExp, disgExp, fearExp, surpExp]
framestogetherf = pd.concat(allexp)
indxn = np.arange(len(framestogetherf))
allExp = np.random.permutation(indxn)
allExp = framestogetherf.sample(frac=1).reset_index(drop=True)
allExp.to_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/original features/happy/allExp.csv', index=False)



y = allExp[65]
allExp.drop(65,axis=1,inplace=True)
X = allExp

tmp = np.array(X)
X1 = tmp[:,0:65]

Y1 = np.array(y)

row, col = X.shape

TR = round(row*0.8)# Training with 80% data
X1_train = X1[0:TR-1,:]
Y1_train = Y1[0:TR-1]

from sklearn import svm

obj = svm.SVC()
obj.fit(X1_train,Y1_train)


X1_test = X1[TR:row,:]
y_test = y[TR:row]

yhat_test = obj.predict(X1_test)



from sklearn import metrics
from sklearn.metrics import classification_report

print("BuiltIn_Accuracy:",metrics.accuracy_score(y_test, yhat_test))
print("BuiltIn_Sensitivity (recall):",metrics.recall_score(y_test, yhat_test, average = 'weighted'))
print('\nClassification Report of MCR dataset\n')
print(classification_report(y_test, yhat_test, target_names=['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5']))

clf = svm.SVC(kernel='linear', C=1).fit(X1_train, Y1_train)
clf.score(X1_test, y_test)

#Overall accuracy score is 25.538

from sklearn.model_selection import cross_val_score
clf = svm.SVC(kernel='linear', C=1, random_state=42)
scores = cross_val_score(clf, X, y, cv=5)
