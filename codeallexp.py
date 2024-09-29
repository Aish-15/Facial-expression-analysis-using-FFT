# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 22:05:39 2021

@author: aishg
"""

import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
#import face_recognition
import dlib


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

fear = cv2.imread("C:/Users/aishg/OneDrive/Desktop/Project/Face recognition/faces/leftlight/subject09/subject09.jpg")
ff = fear.copy()
gray = cv2.cvtColor(fear, cv2.COLOR_BGR2GRAY)

plt.imshow(gray, cmap=plt.get_cmap('gray'))
plt.axis('off')

faces = face_cascade.detectMultiScale(gray, 1.3, 5)

for (x,y,w,h) in faces:
    x1 = x
    y1 = y
    w1 = w
    h1 = h
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = fear[y:y+h, x:x+w]

plt.imshow(roi_gray, cmap=plt.get_cmap('gray'))
plt.axis('off')

predictor_path = 'C:/Users/aishg/anaconda3/envs/project/lib/site-packages/face_recognition_models/models/shape_predictor_81_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

#landmarks = predictor(gray, faces)

gray=gray-1
faces = detector(gray)
for face in faces:
    shape = predictor(gray, face)
    landmarks = np.matrix([[p.x, p.y] for p in shape.parts()])
    for num in range(shape.num_parts):
            cv2.circle(gray, (shape.parts()[num].x, shape.parts()[num].y), 1, (255,0,0), -1)
            #cv2.circle(gray, (shape.parts()[num].x, shape.parts()[num].y), 8, (255,0,0), -1)


new_gray = gray[y1:y1+h1, x1:x1+w1]
    
plt.imshow(new_gray, cmap=plt.get_cmap('gray'))
plt.axis('off')

cv2.imwrite("C:/Users/aishg/OneDrive/Desktop/Project/Face recognition/faces/leftlight/subject09/new_gray_subject09.png", new_gray)

old_gray = ff[y1:y1+h1, x1:x1+w1]
#cv2.imwrite("C:/Users/s_suthah/Desktop/Aishwarya/old_gray_AG.png", old_gray)
cv2.imwrite("C:/Users/aishg/OneDrive/Desktop/Project/Face recognition/faces/leftlight/subject09/old_gray_subject09.png", old_gray)

heightc, widthc = new_gray.shape
bin_gray = np.zeros((heightc, widthc), np.uint8)
th1 = 255
for i in range(heightc):
    for j in range(widthc):
        if(new_gray[i][j]<th1):
            bin_gray[i][j] = 0
        else:
            bin_gray[i][j] = 255  

plt.imshow(bin_gray, cmap=plt.get_cmap('gray'))
#cv2.imwrite("C:/Users/s_suthah/Desktop/Aishwarya/bin_gray_AG.png", bin_gray)
cv2.imwrite("C:/Users/aishg/OneDrive/Desktop/Project/Face recognition/faces/leftlight/subject09/bin_gray_subject09.png", bin_gray)

f = np.fft.fft2(bin_gray)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))
plt.subplot(121),plt.imshow(bin_gray, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()
#cv2.imwrite("C:/Users/s_suthah/Desktop/Aishwarya/spec_gray_AG.png", magnitude_spectrum)
cv2.imwrite("C:/Users/aishg/OneDrive/Desktop/Project/Face recognition/faces/leftlight/subject09/spec_gray_subject09.png", magnitude_spectrum)


#############################################################################################################3



I1 = cv2.imread('C:/Users/aishg/OneDrive/Desktop/Project/Face recognition/faces/happy/subject01/old_gray_subject01.png')
I2 = cv2.imread('C:/Users/aishg/OneDrive/Desktop/Project/Face recognition/faces/sad/subject01/old_gray_subject01.png')
I3 = cv2.imread('C:/Users/aishg/OneDrive/Desktop/Project/Face recognition/faces/sleepy/subject01/old_gray_subject01.png')


I1 = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
I2 = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)
I3 = cv2.cvtColor(I3, cv2.COLOR_BGR2GRAY)

happy = cv2.resize(I1, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
sad = cv2.resize(I2, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
sleepy = cv2.resize(I3, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)

plt.imshow(happy)
plt.imshow(sad)
plt.imshow(sleepy)


plt.imshow(abs(happy-sad))


ss1, ss2 = sleepy.shape

sleepy1 = round(((ss1-8)*(ss2-8))/64)
sleepy2 = np.zeros((sleepy1, 64), np.uint8)

k = 0
for i in range(0,ss1-8,8):
    for j in range(0,ss2-8,8):
        crop_tmpL1 = sleepy[i:i+8,j:j+8]
        sleepy2[k,0:64] = crop_tmpL1.flatten()
        k = k + 1
sleepyy = pd.DataFrame(sleepy2)
sleepyy[64] = "1"




ss1, ss2 = happy.shape

happy_face = round(((ss1-8)*(ss2-8))/64)
happy_face2 = np.zeros((happy_face, 64), np.uint8)

k = 0
for i in range(0,ss1-8,8):
    for j in range(0,ss2-8,8):
        crop_tmpL1 = happy[i:i+8,j:j+8]
        happy_face2[k,0:64] = crop_tmpL1.flatten()
        k = k + 1
happyy = pd.DataFrame(happy_face2)
happyy[64] = "2"



ss1, ss2 = sad.shape

sad_face = round(((ss1-8)*(ss2-8))/64)
sad_face2 = np.zeros((sad_face, 64), np.uint8)

k = 0
for i in range(0,ss1-8,8):
    for j in range(0,ss2-8,8):
        crop_tmpL1 = sad[i:i+8,j:j+8]
        sad_face2[k,0:64] = crop_tmpL1.flatten()
        k = k + 1
sadd = pd.DataFrame(sad_face2)
sadd[64] = "2"

framesh = [sleepyy, happyy, sadd]
framestogether = pd.concat(framesh)


#% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


I1 = cv2.imread('C:/Users/aishg/OneDrive/Desktop/Project/Face recognition/faces/happy/subject01/old_gray_subject01.png')

I1 = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
happy = cv2.resize(I1, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
plt.imshow(happy)


mm1 = []

#print(range(64, 15, -2))

for i in range(63, 14, -2):
    
    f = np.fft.fft2(happy)
    fshift = np.fft.fftshift(f)
    crow = 64; ccol = 64; ww = i; uu = 12;
    tt = fshift-fshift
    tt[crow-ww-1:crow+ww-1, ccol-ww-1:ccol+ww-1] = 1
    tt[crow-ww+uu-1:crow+ww-uu-1, ccol-ww+uu-1:ccol+ww-uu-1] = 0
    
    tshift = tt*fshift
    

    ttshift = tshift.astype(np.uint8)
    
    #plt.imshow(ttshift, cmap='gray')
    #plt.show()
    

    
    f_ishift= np.fft.ifft2(np.fft.ifftshift(tshift))
    dd = np.real(f_ishift)
    dd = 255*(dd-min(dd.min(0)))/(max(dd.max(0))-min(dd.min(0)))
    
    

    plt.imshow(dd, cmap='gray', interpolation = None)
    plt.show()
    dd2 = dd.flatten()
    dd3 = dd2.T
    if i==63:
        mm1 = np.append(mm1,dd3)
    else:
        mm1 = np.vstack((mm1,dd3))
    
    

dd_= (3*((1+dd)/(1+dd))).flatten()   
mm1 = np.vstack((mm1, dd_))
mm1= mm1.T
df = pd.DataFrame(mm1)   

df.to_csv('C:/Users/aishg/OneDrive/Desktop/python codes/code2/3.csv', index=False)



I1 = cv2.imread('C:/Users/aishg/OneDrive/Desktop/Project/Face recognition/faces/sad/subject01/old_gray_subject01.png')

I1 = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
happy = cv2.resize(I1, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
plt.imshow(happy)


mm1 = []

#print(range(64, 15, -2))

for i in range(63, 14, -2):
    
    f = np.fft.fft2(happy)
    fshift = np.fft.fftshift(f)
    crow = 64; ccol = 64; ww = i; uu = 12;
    tt = fshift-fshift
    tt[crow-ww-1:crow+ww-1, ccol-ww-1:ccol+ww-1] = 1
    tt[crow-ww+uu-1:crow+ww-uu-1, ccol-ww+uu-1:ccol+ww-uu-1] = 0
    
    tshift = tt*fshift
    

    ttshift = tshift.astype(np.uint8)
    
    plt.imshow(ttshift, cmap='gray')
    plt.show()
    

    
    f_ishift= np.fft.ifft2(np.fft.ifftshift(tshift))
    dd = np.real(f_ishift)
    dd = 255*(dd-min(dd.min(0)))/(max(dd.max(0))-min(dd.min(0)))
    
    

    #imgplot = plt.imshow(dd, cmap='gray',  interpolation = None)
    #plt.show()  
    dd2 = dd.flatten()
    dd3 = dd2.T
    if i==63:
        mm1 = np.append(mm1,dd3)
    else:
        mm1 = np.vstack((mm1,dd3))
    
    

dd_= (2*((1+dd)/(1+dd))).flatten()   
mm1 = np.vstack((mm1, dd_))
mm1= mm1.T
df = pd.DataFrame(mm1)   

df.to_csv('C:/Users/aishg/OneDrive/Desktop/python codes/code2/2.csv', index=False)



I1 = cv2.imread('C:/Users/aishg/OneDrive/Desktop/Project/Face recognition/faces/sleepy/subject01/old_gray_subject01.png')

I1 = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
happy = cv2.resize(I1, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
plt.imshow(happy)


mm1 = []

#print(range(64, 15, -2))

for i in range(63, 14, -2):
    
    f = np.fft.fft2(happy)
    fshift = np.fft.fftshift(f)
    crow = 64; ccol = 64; ww = i; uu = 12;
    tt = fshift-fshift
    tt[crow-ww-1:crow+ww-1, ccol-ww-1:ccol+ww-1] = 1
    tt[crow-ww+uu-1:crow+ww-uu-1, ccol-ww+uu-1:ccol+ww-uu-1] = 0
    
    tshift = tt*fshift
    

    ttshift = tshift.astype(np.uint8)
    
    #plt.imshow(ttshift, cmap='gray')
    #plt.show()
    

    
    f_ishift= np.fft.ifft2(np.fft.ifftshift(tshift))
    dd = np.real(f_ishift)
    dd = 255*(dd-min(dd.min(0)))/(max(dd.max(0))-min(dd.min(0)))
    
    

    imgplot = plt.imshow(dd, cmap='gray')
    plt.show()  
    dd2 = dd.flatten()
    dd3 = dd2.T
    if i==63:
        mm1 = np.append(mm1,dd3)
    else:
        mm1 = np.vstack((mm1,dd3))
    
    

dd_= (1*((1+dd)/(1+dd))).flatten()   
mm1 = np.vstack((mm1, dd_))
mm1= mm1.T
df = pd.DataFrame(mm1, index = None)

df.to_csv('C:/Users/aishg/OneDrive/Desktop/python codes/code2/1.csv', index=False)

a=pd.read_csv("C:/Users/aishg/OneDrive/Desktop/python codes/code2/1.csv", header = None)
b=pd.read_csv("C:/Users/aishg/OneDrive/Desktop/python codes/code2/2.csv", header = None)
c=pd.read_csv("C:/Users/aishg/OneDrive/Desktop/python codes/code2/3.csv", header = None)

abc = [a,b,c]
dfff = pd.concat(abc)


#########################################################################################

# =============================================================================
#USING GLOB FUNCTION 
# =============================================================================


import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import os


MM = []
im_path = "C:/Users/aishg/OneDrive/Desktop/Project/Face recognition/train/happy/"
k = 1
for image in os.listdir(im_path):
    print(image)
    df = pd.DataFrame()
    
    input_img = cv2.imread(im_path + image)
    if input_img.ndim == 3 and input_img.shape[-1] == 3:
        img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    elif input_img.ndim == 2:
        img = input_img
    happy = cv2.resize(img, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
    mm1 = []
    for i in range(63, 14, -2):
    
        f = np.fft.fft2(happy)
        fshift = np.fft.fftshift(f)
        crow = 64; ccol = 64; ww = i; uu = 12;
        tt = fshift-fshift
        tt[crow-ww-1:crow+ww-1, ccol-ww-1:ccol+ww-1] = 1
        tt[crow-ww+uu-1:crow+ww-uu-1, ccol-ww+uu-1:ccol+ww-uu-1] = 0
        
        tshift = tt*fshift
    

        ttshift = tshift.astype(np.uint8)
    
        #plt.imshow(ttshift, cmap='gray')
        #plt.show()
        f_ishift= np.fft.ifft2(np.fft.ifftshift(tshift))
        dd = np.real(f_ishift)
        dd = 255*(dd-min(dd.min(0)))/(max(dd.max(0))-min(dd.min(0)))
        #imgplot = plt.imshow(dd, cmap='gray')
        #plt.show()  
        dd2 = dd.flatten()
        dd3 = dd2.T
        if i==63:
             mm1 = np.append(mm1,dd3, axis = None)
        else:
             mm1 = np.vstack((mm1,dd3))
    if k==1:
        MM = mm1.T
    else:
        MM =  np.vstack((MM,mm1.T))    
    k = k +1
 
df = pd.DataFrame(MM)
df[25] = 1

df.to_csv('C:/Users/aishg/OneDrive/Desktop/python codes/code2/1.csv', index=False)

a = pd.read_csv("C:/Users/aishg/OneDrive/Desktop/python codes/code2/1.csv", header = None)
a[25] = 1



MM = []
im_path = "C:/Users/aishg/OneDrive/Desktop/Project/Face recognition/train/sad/"
k = 1
for image in os.listdir(im_path):
    print(image)
    df = pd.DataFrame()
    
    input_img = cv2.imread(im_path + image)
    if input_img.ndim == 3 and input_img.shape[-1] == 3:
        img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    elif input_img.ndim == 2:
        img = input_img
    happy = cv2.resize(img, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
    mm1 = []
    for i in range(63, 14, -2):
    
        f = np.fft.fft2(happy)
        fshift = np.fft.fftshift(f)
        crow = 64; ccol = 64; ww = i; uu = 12;
        tt = fshift-fshift
        tt[crow-ww-1:crow+ww-1, ccol-ww-1:ccol+ww-1] = 1
        tt[crow-ww+uu-1:crow+ww-uu-1, ccol-ww+uu-1:ccol+ww-uu-1] = 0
        
        tshift = tt*fshift
    

        ttshift = tshift.astype(np.uint8)
    
       # plt.imshow(ttshift, cmap='gray')
        #plt.show()
        f_ishift= np.fft.ifft2(np.fft.ifftshift(tshift))
        dd = np.real(f_ishift)
        dd = 255*(dd-min(dd.min(0)))/(max(dd.max(0))-min(dd.min(0)))
        #imgplot = plt.imshow(dd, cmap='gray')
        #plt.show()  
        dd2 = dd.flatten()
        dd3 = dd2.T
        if i==63:
            mm1 = np.append(mm1,dd3, axis = None)
        else:
            mm1 = np.vstack((mm1,dd3))
    if k==1:
        MM = mm1.T
    else:
        MM =  np.vstack((MM,mm1.T))    
    k = k +1
 
df = pd.DataFrame(MM)
df[25] = 2
df.to_csv('C:/Users/aishg/OneDrive/Desktop/python codes/code2/2.csv', index=False)



MM = []
im_path = "C:/Users/aishg/OneDrive/Desktop/Project/Face recognition/train/sleepy/"
k = 1
for image in os.listdir(im_path):
    print(image)
    df = pd.DataFrame()
    
    input_img = cv2.imread(im_path + image)
    if input_img.ndim == 3 and input_img.shape[-1] == 3:
        img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    elif input_img.ndim == 2:
        img = input_img
    happy = cv2.resize(img, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
    mm1 = []
    for i in range(63, 14, -2):
    
        f = np.fft.fft2(happy)
        fshift = np.fft.fftshift(f)
        crow = 64; ccol = 64; ww = i; uu = 12;
        tt = fshift-fshift
        tt[crow-ww-1:crow+ww-1, ccol-ww-1:ccol+ww-1] = 1
        tt[crow-ww+uu-1:crow+ww-uu-1, ccol-ww+uu-1:ccol+ww-uu-1] = 0
        
        tshift = tt*fshift
    

        ttshift = tshift.astype(np.uint8)
    
       # plt.imshow(ttshift, cmap='gray')
        #plt.show()
        f_ishift= np.fft.ifft2(np.fft.ifftshift(tshift))
        dd = np.real(f_ishift)
        dd = 255*(dd-min(dd.min(0)))/(max(dd.max(0))-min(dd.min(0)))
        #imgplot = plt.imshow(dd, cmap='gray')
        #plt.show()  
        dd2 = dd.flatten()
        dd3 = dd2.T
        if i==63:
            mm1 = np.append(mm1,dd3, axis = None)
        else:
            mm1 = np.vstack((mm1,dd3))
    if k==1:
        MM = mm1.T
    else:
        MM =  np.vstack((MM,mm1.T))    
    k = k +1
 
df = pd.DataFrame(MM)
df[25] = 3
df.to_csv('C:/Users/aishg/OneDrive/Desktop/python codes/code2/3.csv', index=False)



MM = []
im_path = "C:/Users/aishg/OneDrive/Desktop/Project/Face recognition/train/surprised/"
k = 1
for image in os.listdir(im_path):
    print(image)
    df = pd.DataFrame()
    
    input_img = cv2.imread(im_path + image)
    if input_img.ndim == 3 and input_img.shape[-1] == 3:
        img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    elif input_img.ndim == 2:
        img = input_img
    happy = cv2.resize(img, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
    mm1 = []
    for i in range(63, 14, -2):
    
        f = np.fft.fft2(happy)
        fshift = np.fft.fftshift(f)
        crow = 64; ccol = 64; ww = i; uu = 12;
        tt = fshift-fshift
        tt[crow-ww-1:crow+ww-1, ccol-ww-1:ccol+ww-1] = 1
        tt[crow-ww+uu-1:crow+ww-uu-1, ccol-ww+uu-1:ccol+ww-uu-1] = 0
        
        tshift = tt*fshift
    

        ttshift = tshift.astype(np.uint8)
    
       # plt.imshow(ttshift, cmap='gray')
        #plt.show()
        f_ishift= np.fft.ifft2(np.fft.ifftshift(tshift))
        dd = np.real(f_ishift)
        dd = 255*(dd-min(dd.min(0)))/(max(dd.max(0))-min(dd.min(0)))
        #imgplot = plt.imshow(dd, cmap='gray')
        #plt.show()  
        dd2 = dd.flatten()
        dd3 = dd2.T
        if i==63:
            mm1 = np.append(mm1,dd3, axis = None)
        else:
            mm1 = np.vstack((mm1,dd3))
    if k==1:
        MM = mm1.T
    else:
        MM =  np.vstack((MM,mm1.T))    
    k = k +1
 
df = pd.DataFrame(MM)
df[25] = 4
df.to_csv('C:/Users/aishg/OneDrive/Desktop/python codes/code2/4.csv', index=False)


MM = []
im_path = "C:/Users/aishg/OneDrive/Desktop/Project/Face recognition/train/wink/"
k = 1
for image in os.listdir(im_path):
    print(image)
    df = pd.DataFrame()
    
    input_img = cv2.imread(im_path + image)
    if input_img.ndim == 3 and input_img.shape[-1] == 3:
        img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    elif input_img.ndim == 2:
        img = input_img
    happy = cv2.resize(img, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
    mm1 = []
    for i in range(63, 14, -2):
    
        f = np.fft.fft2(happy)
        fshift = np.fft.fftshift(f)
        crow = 64; ccol = 64; ww = i; uu = 12;
        tt = fshift-fshift
        tt[crow-ww-1:crow+ww-1, ccol-ww-1:ccol+ww-1] = 1
        tt[crow-ww+uu-1:crow+ww-uu-1, ccol-ww+uu-1:ccol+ww-uu-1] = 0
        
        tshift = tt*fshift
    

        ttshift = tshift.astype(np.uint8)
    
       # plt.imshow(ttshift, cmap='gray')
        #plt.show()
        f_ishift= np.fft.ifft2(np.fft.ifftshift(tshift))
        dd = np.real(f_ishift)
        dd = 255*(dd-min(dd.min(0)))/(max(dd.max(0))-min(dd.min(0)))
        imgplot = plt.imshow(dd, cmap='gray')
        plt.show()  
        dd2 = dd.flatten()
        dd3 = dd2.T
        if i==63:
            mm1 = np.append(mm1,dd3, axis = None)
        else:
            mm1 = np.vstack((mm1,dd3))
    if k==1:
        MM = mm1.T
    else:
        MM =  np.vstack((MM,mm1.T))    
    k = k +1
 
df = pd.DataFrame(MM)
df[25] = 5
df.to_csv('C:/Users/aishg/OneDrive/Desktop/python codes/code2/5.csv', index=False)

a = pd.read_csv("C:/Users/aishg/OneDrive/Desktop/python codes/code2/1.csv", index_col = None)
b = pd.read_csv("C:/Users/aishg/OneDrive/Desktop/python codes/code2/2.csv", index_col = None)
c = pd.read_csv("C:/Users/aishg/OneDrive/Desktop/python codes/code2/3.csv", index_col = None)
d = pd.read_csv("C:/Users/aishg/OneDrive/Desktop/python codes/code2/4.csv", index_col = None)
e = pd.read_csv("C:/Users/aishg/OneDrive/Desktop/python codes/code2/5.csv", index_col = None)



allpds = [a,b,c,d,e]
dffff = pd.concat(allpds)