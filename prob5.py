#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 20:47:47 2019

@author: avinashshanker
"""
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 



def mySobelFilter(I):
	mag = np.copy(I)
	ori  =np.zeros(I.shape)
	size = mag.shape
	for i in range(1, size[0] - 1):
	    for j in range(1, size[1] - 1):
	        gx = (I[i - 1][j - 1] + 2*I[i][j - 1] + I[i + 1][j - 1]) - (I[i - 1][j + 1] + 2*I[i][j + 1] + I[i + 1][j + 1])
	        gy = (I[i - 1][j - 1] + 2*I[i - 1][j] + I[i - 1][j + 1]) - (I[i + 1][j - 1] + 2*I[i + 1][j] + I[i + 1][j + 1])
	        mag[i][j] = min(255, np.sqrt(gx**2 + gy**2))
	        ori[i][j] = math.degrees(math.atan(float(gy/gx)))
	return mag,ori

def main():
    original = mpimg.imread('Lena.png')
    gray_img = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    mag,ori = mySobelFilter(gray_img)
    fig1 = plt.figure(figsize = (12,12))
    fig1.suptitle('Magnitude and Orientation', fontsize=14)
    a = fig1.add_subplot(2, 2, 1)
    imgplot = plt.imshow(mag,cmap='gray')
    a.set_title('Magnitude')
    a = fig1.add_subplot(2, 2, 2)
    imgplot = plt.imshow(ori,cmap='gray')
    a.set_title('Orientation')
    
    HSV = mag + ori
    
    a = fig1.add_subplot(2, 2, 3)
    imgplot = plt.imshow(HSV,cmap='hsv')
    a.set_title('Conversion to Color')

     
    
    
    
    
if __name__ == "__main__":
    main()