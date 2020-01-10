#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 20:47:47 2019

@author: avinashshanker
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image = cv2.imread('Lena.png')
#gray scale normalized image
image_gray = cv2.imread('Lena.png',0)
cv2.imwrite('Lena_gray.png', image_gray)
#downsample image once
First_down_sample=image_gray[0::2,0::2]
cv2.imwrite('First_down_sample.png', First_down_sample)
#downsample image twice
Second_down_sample=First_down_sample[0::2,0::2]
cv2.imwrite('Second_down_sample.png', Second_down_sample)

#upsample 128x128 to 256x256
rows,cols = Second_down_sample.shape
upsample_128 = np.zeros((rows*2,cols*2))
i,j = 0,0
while j < rows:
    a,b = 0,0
    while b < cols:
        upsample_128[i,a] = Second_down_sample[j,b]
        a += 2
        b +=1
    i += 2
    j += 1

cv2.imwrite('upsample_256.png', upsample_128)

#upsample 256x256 to 512x512
rows2,cols2 = upsample_128.shape
upsample_256 = np.zeros((rows2*2,cols2*2))

i,j = 0,0
while j < rows2:
    a,b = 0,0
    while b < cols2:
        upsample_256[i,a] = upsample_128[j,b]
        a += 2
        b +=1
    i += 2
    j += 1

cv2.imwrite('upsample_512.png', upsample_256)

##############################################
#This part of code only for Displaying  images
##############################################

Original = mpimg.imread('Lena.png')
Grap_Scale = mpimg.imread('Lena_gray.png')
fig1 = plt.figure(figsize = (10,10))
a = fig1.add_subplot(1, 2, 1)
imgplot = plt.imshow(Original)
a.set_title('Original Lena Image')
a = fig1.add_subplot(1, 2, 2)
imgplot = plt.imshow(Grap_Scale,cmap='gray')
a.set_title('Gray Scale Converted')

down_img1 = mpimg.imread('First_down_sample.png')
down_img2 = mpimg.imread('Second_down_sample.png')
fig2 = plt.figure(figsize = (10,10))
a = fig2.add_subplot(1, 2, 1)
imgplot = plt.imshow(down_img1,cmap='gray')
a.set_title('1st Down Sample 256x256')
a = fig2.add_subplot(1, 2, 2)
imgplot = plt.imshow(down_img2,cmap='gray')
a.set_title('2nd Down Sample 128x128')

up_img1 = mpimg.imread('upsample_256.png')
up_img2 = mpimg.imread('upsample_512.png')
fig3 = plt.figure(figsize = (16,16))
a = fig3.add_subplot(2, 1, 1)
imgplot = plt.imshow(up_img1,cmap='gray')
a.set_title('1st Up Sample 256x256')
a = fig3.add_subplot(2, 1, 2)
imgplot = plt.imshow(up_img2,cmap='gray')
a.set_title('2nd Up Sample 512x512')




