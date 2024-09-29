# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 16:29:09 2021

@author: aishg
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import glob
import cv2
from sklearn.model_selection import train_test_split
from keras.layers import Dropout, Dense
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, load_model
from keras.applications import VGG16
from sklearn.metrics import accuracy_score, confusion_matrix

Happy = glob.glob("C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/happy/*")
print("Number of images in Happy emotion = "+str(len(Happy)))

happy_folderName = [str(i.split("\\")[0])+"/" for i in Happy]
happy_imageName = [str(i.split("\\")[1]) for i in Happy]
happy_emotion = [["Happy"]*len(Happy)][0]
happy_label = [1]*len(Happy)

#len(human_angry_folderName), len(human_angry_imageName), len(human_angry_emotion), len(human_angry_label)

df_happy = pd.DataFrame()
df_happy["folderName"] = happy_folderName
df_happy["imageName"] = happy_imageName
df_happy["Emotion"] = happy_emotion
df_happy["Labels"] = happy_label
df_happy.head()


human_happy = glob.glob("C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/happy/*")
human_happy.remove('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/happy\\Thumbs.db')
print("Number of images in Happy emotion = "+str(len(human_happy)))


Sad = Happy = glob.glob("C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/sad/*")
print("Number of images in sad emotion = "+str(len(Sad)))

sad_folderName = [str(i.split("\\")[0])+"/" for i in Sad]
sad_imageName = [str(i.split("\\")[1]) for i in Sad]
sad_emotion = [["Sad"]*len(Sad)][0]
sad_label = [2]*len(Sad)

#len(human_angry_folderName), len(human_angry_imageName), len(human_angry_emotion), len(human_angry_label)

df_sad = pd.DataFrame()
df_sad["folderName"] = sad_folderName
df_sad["imageName"] = sad_imageName
df_sad["Emotion"] = sad_emotion
df_sad["Labels"] = sad_label
df_sad.head()


fear = glob.glob("C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fear/*")
print("Number of images in fear emotion = "+str(len(fear)))

Disgusted = glob.glob("C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/disgusted/*")
print("Number of images in Disgusted emotion = "+str(len(Disgusted)))

surprised = glob.glob("C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/surprised/*")
print("Number of images in surprised emotion = "+str(len(surprised)))



