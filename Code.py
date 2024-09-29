# -*- coding: utf-8 -*-
"""
@author: aishg
"""

# =============================================================================
# Updated final code:
# =============================================================================
# =============================================================================
# For a single image
# =============================================================================

import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt


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
    
    plt.imshow(ttshift, cmap='gray')
    plt.show()
    
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

dd_= (3*((1+dd)/(1+dd))).flatten()   
mm1 = np.vstack((mm1, dd_))
mm1= mm1.T
df = pd.DataFrame(mm1)   

df.to_csv('C:/Users/aishg/OneDrive/Desktop/python codes/code1/3.csv', index=False)


# =============================================================================
# Code for multiple images in a folder 
# =============================================================================


import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import os


im_path = "C:/Users/aishg/OneDrive/Desktop/Project/Face recognition/train/happy/"
MM = []
k = 1
for image in os.listdir(im_path):
    
    input_img = cv2.imread(im_path + image)
    img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
 
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
        #plt.imshow(dd, cmap='gray')
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

df.to_csv('C:/Users/aishg/OneDrive/Desktop/Project/Face recognition/train/happy/1.csv', index=False)
