# -*- coding: utf-8 -*-
"""
@author: aishg
"""

import cv2 as cv
import matplotlib.pyplot as plt

ddepth = cv.CV_16S
kernel_size = 3
img = cv.imread('C:/Users/aishg/OneDrive/Desktop/Project/Facial expression detection/seperate features/eyeleftfaceF.jpg')

src_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
src = cv.GaussianBlur(src_gray, (3, 3), 0)

dst = cv.Laplacian(src, ddepth, ksize=5)
sobelx = cv.Sobel(src,cv.CV_64F,1,0,ksize=5)  # x
sobely = cv.Sobel(src,cv.CV_64F,0,1,ksize=5)  # y
sobel = cv.Sobel(src,cv.CV_64F,1,1,ksize=5)  
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure

fd, hog_image = hog(img, orientations=9, multichannel=True, pixels_per_cell=(18, 18), cells_per_block=(2, 2), visualize=True)

edges = cv.Canny(src,100,200)

plt.subplot(2,4,1),plt.imshow(img)
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,4,2),plt.imshow(dst,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2,4,3),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,4,4),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
plt.subplot(2,4,5),plt.imshow(edges,cmap = 'gray')
plt.title('canny'), plt.xticks([]), plt.yticks([])
plt.subplot(2,4,6),plt.imshow(sobel,cmap = 'gray')
plt.title('Gradient magnitude'), plt.xticks([]), plt.yticks([])
plt.subplot(2,4,8),plt.imshow(src,cmap = 'gray')
plt.title('Gaussian'), plt.xticks([]), plt.yticks([])
plt.subplot(2,4,7),plt.imshow(hog_image,cmap = 'gray')
plt.title('HOG'), plt.xticks([]), plt.yticks([])

