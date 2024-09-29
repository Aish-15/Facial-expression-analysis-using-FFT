# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 16:26:41 2021

@author: aishg
"""

# =============================================================================
# Detection and extraction of forehead in the surprised face
# =============================================================================


import face_recognition
import cv2
import PIL.Image
import PIL.ImageDraw
import os

image =face_recognition.load_image_file('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/happy/shalu.jpg')
#image = Image.open('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/happy/aish.jpg')
#image.show()

#image = face_recognition.load_image_file(image1)
face_locations = face_recognition.face_locations(image) # detects all the faces in image
t = len(face_locations)
print(len(face_locations))
print(face_locations)
face_landmarks_list = face_recognition.face_landmarks(image)
pil_image = PIL.Image.fromarray(image)
for face_location in face_locations:
    top,right,bottom,left =face_location
    draw_shape = PIL.ImageDraw.Draw(pil_image)
    surprise1 = PIL.Image.open("C:/Users/aishg/Downloads/2.png")
    k = face_landmarks_list[0]['right_eyebrow']
    bottom= face_landmarks_list[0]['right_eyebrow'][0][1]
    for k1 in k :   
        if(bottom>k1[1]):
            bottom=k1[1]
    k = face_landmarks_list[0]['left_eyebrow']
    lbottom= face_landmarks_list[0]['left_eyebrow'][0][1]
    for k1 in k :   
        if(lbottom>k1[1]):
            lbottom=k1[1]
    bottom=min(bottom,lbottom)
    print(bottom)
    surprise = surprise1.crop((left, top, right, bottom))
    surprise.save("headdddddddd.jpg")    
 #   draw_shape.rectangle([left, top, right, bottom],outline="blue")
