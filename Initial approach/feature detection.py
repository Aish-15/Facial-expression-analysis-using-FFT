# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 01:22:38 2021

@author: aishg
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt


face_cascade = cv2.CascadeClassifier('C:/Users/aishg/anaconda3/envs/project1/Library/etc/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:/Users/aishg/anaconda3/envs/project1/Library/etc/haarcascades/haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier('/Users/aishg/anaconda3/envs/project1/Library/etc/haarcascades//haarcascade_smile.xml')
nose_cascade = cv2.CascadeClassifier('/Users/aishg/anaconda3/envs/project1/Library/etc/haarcascades/haarcascade_mcs_nose.xml')
#nose classifier: https://sourceforge.net/p/emgucv/opencv/ci/d58cd9851fdb592a395488fc5721d11c7436a99c/tree/data/haarcascades/

image = cv2.imread('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/Sad/sad.jpg')
grayimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(grayimg, 1.3, 5)


for (x,y,w,h) in faces:
  #  cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
    roi_gray = grayimg[y:y+h, x:x+w]
    roi_color = image[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(grayimg, 1.5, 5)
    nose =  nose_cascade.detectMultiScale(grayimg, 1.5, 5)
    mouth = mouth_cascade.detectMultiScale(grayimg, 1.7, 5)
    

    for (ex,ey,ew,eh) in eyes:
    #    cv2.rectangle(img, (ex,ey), (ex+ew, ey+eh), (0,255,0), 2)
        sub_face = image[ey:ey+eh, ex:ex+ew]
        
        
        face_file_name = "face/" + str(y) + ".jpg"
        cv2.imwrite("C:/Users/aishg/Downloads/eye.jpg", sub_face)
        plt.imshow(sub_face)
   # for (nx, ny, nw, nh) in nose:
   #    cv2.rectangle(img, (nx, ny), (nx + nw, ny + nh), (0, 0, 255), 2)
    for (mx, my, mw, mh) in mouth:
     #   cv2.rectangle(img, (mx, my), (mx + mw, my + mh), (60, 20, 60), 2)
        mouth_face = image[my:my+mh, mx:mx+mw]
#        face_file_name = "face/" + str(y) + ".jpg"
#        cv2.imwrite("C:/Users/aishg/Downloads/mouth.jpg", mouth_face)
    plt.imshow(mouth_face)

    for (nx, ny, nw, nh) in nose:
    #    cv2.rectangle(img, (nx, ny), (nx + nw, ny + nh), (0, 0, 255), 2)
        nose_face = image[ny:ny+nh, nx:nx+nw]
  #      face_file_name = "face/" + str(y) + ".jpg"
   #     cv2.imwrite("C:/Users/aishg/Downloads/nose.jpg", nose_face)
plt.imshow(nose_face)