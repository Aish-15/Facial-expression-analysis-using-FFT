# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 20:37:10 2021

@author: aishg
"""

# =============================================================================
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# import dlib
# 
# face_cascade = cv2.CascadeClassifier('C:/Users/aishg/anaconda3/envs/project1/Library/etc/haarcascades/haarcascade_frontalface_default.xml')
# 
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("C:/Users/aishg/anaconda3/envs/project/lib/site-packages/face_recognition_models/models/shape_predictor_68_face_landmarks.dat")
# 
# 
# img = cv2.imread('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/happy/aish.jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 
# faces = face_cascade.detectMultiScale(gray, 1.3, 5)
# #faces = detector(gray)
# 
# for (x,y,w,h) in faces:
#     cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
# #    roi_gray = gray[y:y+h, x:x+w]
#   #  roi_color = img[y:y+h, x:x+w]
#  #   cv2.circle(img, (x, y), 4, (0, 255, 0), -1)
#    # plt.imshow(img)
#     
#     landmarks = predictor(gray, img)
# 
#     x_pts = []
#     y_pts = []
# 
#     for n in range(68, 81):
#         x = landmarks.part(n).x
#         y = landmarks.part(n).y
# 
#         x_pts.append(x)
#         y_pts.append(y)
#     
#     x1 = min(x_pts)
#     x2 = max(x_pts)
#     y1 = min(y_pts)
#     y2 = max(y_pts)
# 
#     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
#    # plt.imshow(img)
#     sunface = img[y2:y1, x2:x1]
# plt.imshow(sunface)
#     
# for face in faces:
#         x1 = face.left()
#         y1 = face.top()
#         x2 = face.right()
#         y2 = face.bottom()
#         
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
#         
#         landmarks = predictor(gray, face)
# 
#         for n in range(68, 81):
#             x = landmarks.part(n).x
#             y = landmarks.part(n).y
# 
#             cv2.circle(frame, (x, y), 4, (0, 255, 0), -1) 
#             
# 
# import numpy as np
# import cv2
# 
# face_cascade = cv2.CascadeClassifier('C:/Users/aishg/anaconda3/envs/project1/Library/etc/haarcascades/haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier('C:/Users/aishg/anaconda3/envs/project1/Library/etc/haarcascades/haarcascade_eye.xml')
# 
# img = cv2.imread('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/happy/aish.jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 
# faces = face_cascade.detectMultiScale(gray, 1.3, 5)
# for (x,y,w,h) in faces:
#     img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
#     roi_gray = gray[y:y+h, x:x+w]
#     roi_color = img[y:y+h, x:x+w]
#     eyes = eye_cascade.detectMultiScale(roi_gray)
#     for (ex,ey,ew,eh) in eyes:
#         cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
#         plt.imshow(img)
# 
# #https://stackoverflow.com/questions/63770831/is-there-a-way-to-get-the-area-of-the-forehead-bounding-box-by-using-opencv-dl
# 
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# import dlib
# 
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("C:/Users/aishg/anaconda3/envs/project/lib/site-packages/face_recognition_models/models/shape_predictor_81_face_landmarks.dat")
# 
# 
# img = cv2.imread('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/happy/aish.jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 
# faces = face_cascade.detectMultiScale(gray, 1.3, 5)
# #faces = detector(gray)
# 
# for (x,y,w,h) in faces:
#     cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
#     roi_gray = gray[y:y+h, x:x+w]
#     roi_color = img[y:y+h, x:x+w]
#  #   cv2.circle(img, (x, y), 4, (0, 255, 0), -1)
#     plt.imshow(img)
#     
#     landmarks = predictor(img)
# 
#     x_pts = []
#     y_pts = []
# 
#     for n in range(68, 81):
#         x = landmarks.part(n).x
#         y = landmarks.part(n).y
# 
#         x_pts.append(x)
#         y_pts.append(y)
#     
#     x1 = min(x_pts)
#     x2 = max(x_pts)
#     y1 = min(y_pts)
#     y2 = max(y_pts)
# 
#     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
#    # plt.imshow(img)
#     sunface = img[y2:y1, x2:x1]
#     plt.imshow(sunface)
# 
# 
# 
# sad = cv2.imread('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/Sad/IMG_5135.jpg')
# grey = cv2.cvtColor(sad, cv2.COLOR_BGR2GRAY)
# plt.imshow(grey, cmap=plt.get_cmap('gray'))
# 
# faces = face_cascade.detectMultiScale(grey, 1.3, 11)
# 
# =============================================================================

import cv2
import dlib
import matplotlib.pyplot as plt

face_cascade = cv2.CascadeClassifier('C:/Users/aishg/anaconda3/envs/project1/Library/etc/haarcascades/haarcascade_frontalface_default.xml')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:/Users/aishg/anaconda3/envs/project/lib/site-packages/face_recognition_models/models/shape_predictor_81_face_landmarks.dat")

img = cv2.imread('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/Sad/IMG_5135.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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

      #  cv2.circle(img, (x, y), 4, (0, 255, 0), -1)

        x1 = min(x_pts)
        x2 = max(x_pts)
        y1 = min(y_pts)
        y2 = max(y_pts)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
        roi_gray = gray[y2:y1, x2:x1]
    plt.imshow(img)

    gray = cv2.cvtColor(roi_gray, cv2.COLOR_BGR2GRAY)

    for (x1, y1, x2, y2) in roi_gray:
    #    cv2.rectangle(img, (nx, ny), (nx + nw, ny + nh), (0, 0, 255), 2)
        dude = roi_gray[y2:y1, x2:x1]
        plt.imshow(dude, cmap= plt.get_cmap(gray))


        plt.imshow(roi_gray, cmap=plt.get_cmap(gray))
