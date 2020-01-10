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

#gray scale normalized image
image_gray = cv2.imread('Lena.png',0)
cv2.imwrite('Lena_gray.png', image_gray)

downsample256 = image_gray[::2,::2]
downsample128 = downsample256[::2,::2]

cv2.imwrite('downsample128.png', downsample128)

def myGaussianSmoothing(I, k, s):
    ax = np.linspace(-(k - 1) / 2., (k - 1) / 2., k)
    x, y = np.meshgrid(ax, ax)
    exp = np.exp(-0.5 * (np.square(x) + np.square(y)) / np.square(s))
    divide = 2*np.pi*s*s
    kernel = exp/divide
    img_row, img_col = I.shape[0],I.shape[1]
    kernel_row,kernel_col = kernel.shape[0],kernel.shape[1]
    output = np.zeros(I.shape)
    pad_height = int((kernel_row - 1) / 2)
    pad_width = int((kernel_col - 1) / 2)
    padded_image = np.zeros((img_row + (2 * pad_height), img_col + (2 * pad_width)))
    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = I
    for row in range(img_row):
        for col in range(img_col):
            output[row, col] = np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col])
    return output

def medianFiltering(image,kernel):
    image1=np.copy(image)
    indexer = kernel//2
    dims = image1.shape
    rows = image1.shape[0]
    cols = image1.shape[1]
    med_filter = np.zeros(dims)
    for i in range(rows):
        hold = []
        for j in range(cols):
            for z in range(kernel):
                    if (i + z - indexer < 0 or i + z - indexer > len(image1) - 1):
                        for c in range(kernel):
                            hold.append(0)
                    else:
                        if (j + z - indexer < 0 or j + indexer > len(image1) - 1):
                            hold.append(0)
                        else:
                            for k in range(kernel):
                                hold.append(image1[i + z - indexer][j + k - indexer])
            hold.sort()
            med_filter[i][j] = hold[len(hold) // 2]
    return med_filter

#main function
def main():

    upsample256 = np.zeros((2*downsample128.shape[0],2*downsample128.shape[1]),np.uint8)
    i,j = 0,0
    while j < downsample128.shape[0]:
        m,n = 0,0
        while n < downsample128.shape[1]:
            upsample256[i,m] = downsample128[j,n]
            m += 2
            n += 1
        i += 2
        j += 1
    
    cv2.imwrite('upsample256.png',upsample256)
    
    I_smooth1 = myGaussianSmoothing(upsample256,11, 1)
    cv2.imwrite('Smooth_up256.png',I_smooth1)
    
    upsample512 = np.zeros((2*I_smooth1.shape[0],2*I_smooth1.shape[1]),np.uint8)
    i,j = 0,0
    while j < I_smooth1.shape[0]:
        m,n = 0,0
        while n < I_smooth1.shape[1]:
            upsample512[i,m] = I_smooth1[j,n]
            m += 2
            n += 1
        i += 2
        j += 1


    I_smooth2 = myGaussianSmoothing(upsample512,11, 1)
    cv2.imwrite('Smooth_up512.png',I_smooth2)

    Median_Filtering = medianFiltering(upsample256,11)
    cv2.imwrite('Median_up256.png',Median_Filtering)


    up256 = mpimg.imread('upsample256.png')
    gaus_smooth256 = mpimg.imread('Smooth_up256.png')
    gaus_smooth512 = mpimg.imread('Smooth_up512.png')
    median_smooth = mpimg.imread('Median_up256.png')
    fig1 = plt.figure(figsize = (10,10))
    fig1.suptitle('Gaussian Vs Median Filtering', fontsize=14)
    a = fig1.add_subplot(2, 2, 1)
    imgplot = plt.imshow(up256,cmap='gray')
    a.set_title('Upsampled from 128 to 256')
    a = fig1.add_subplot(2, 2, 2)
    imgplot = plt.imshow(gaus_smooth256,cmap='gray')
    a.set_title('Gaussian Smoothing 256x256')
    a = fig1.add_subplot(2, 2, 3)
    imgplot = plt.imshow(gaus_smooth512,cmap='gray')
    a.set_title('Gaussian Smoothing 512x512')
    a = fig1.add_subplot(2, 2, 4)
    imgplot = plt.imshow(median_smooth,cmap='gray')
    a.set_title('Median smoothing 256x256')
    
    

if __name__ == "__main__":
    main()
