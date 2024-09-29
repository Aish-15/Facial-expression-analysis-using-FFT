# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 02:20:49 2021

@author: aishg
"""
import cv2
import pandas as pd
import numpy as np

happy = pd.read_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/Happyface.csv')
sad = pd.read_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/SadExpressionsP.csv')
disgust = pd.read_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/DisgExpressionsP.csv')
fear = pd.read_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/fearface.csv')
surprised = pd.read_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/Surpface.csv')


frames = [happy, sad, disgust,  fear, surprised]
mgedexp = pd.concat(frames)
indxexp = np.arange(len(mgedexp))
Exp = np.random.permutation(indxexp)
Exp = mgedexp.sample(frac=1).reset_index(drop=True)
Exp.to_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/allExpP.csv', index=False)
mgedexp.to_csv('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/fspace/AllExpMerged.csv', index=False)

