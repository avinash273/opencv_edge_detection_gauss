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

##############################################
#This part of code only for Displaying  images
##############################################
def main():

    #changing k={3, 5, 7, 11, 51} with fixed s = 1.
    kernel = [3, 5, 7, 11, 51]
    for i  in range(len(kernel)):
        k = kernel[i] 
        I_smooth = myGaussianSmoothing(image_gray,k, 1)
        filename = "kernel_change_%d.png"%k
        cv2.imwrite(filename,I_smooth)

    #changing the s = {0.1, 1, 2, 3, 5} with fixed kernel size k = 11.
    sigma = [0.1, 1, 2, 3, 5]
    for i  in range(len(sigma)):
        s = sigma[i] 
        I_smooth = myGaussianSmoothing(image_gray,11, s)
        filename = "sigma_change%d.png"%s
        cv2.imwrite(filename,I_smooth)
   

    kernel3 = mpimg.imread('kernel_change_3.png')
    kernel5 = mpimg.imread('kernel_change_5.png')
    kernel7 = mpimg.imread('kernel_change_7.png')
    kernel11 = mpimg.imread('kernel_change_11.png')
    kernel51 = mpimg.imread('kernel_change_51.png')
    fig1 = plt.figure(figsize = (10,10))
    fig1.suptitle('changing k={3, 5, 7, 11, 51} with fixed s = 1', fontsize=16)
    a = fig1.add_subplot(2, 2, 1)
    imgplot = plt.imshow(kernel3,cmap='gray')
    a.set_title('kernel_change_3')
    a = fig1.add_subplot(2, 2, 2)
    imgplot = plt.imshow(kernel5,cmap='gray')
    a.set_title('kernel_change_5')
    a = fig1.add_subplot(2, 2, 3)
    imgplot = plt.imshow(kernel7,cmap='gray')
    a.set_title('kernel_change_7')
    a = fig1.add_subplot(2, 2, 4)
    imgplot = plt.imshow(kernel11,cmap='gray')
    a.set_title('kernel_change_11')
    
    fig1 = plt.figure(figsize = (10,10))
    a = fig1.add_subplot(1, 2, 1)
    imgplot = plt.imshow(kernel51,cmap='gray')
    a.set_title('kernel_change51')
    
    sigma0 = mpimg.imread('sigma_change1.png')
    sigma1 = mpimg.imread('sigma_change1.png')
    sigma2 = mpimg.imread('sigma_change2.png')
    sigma3 = mpimg.imread('sigma_change3.png')
    sigma5 = mpimg.imread('sigma_change5.png')
    fig1 = plt.figure(figsize = (10,10))
    fig1.suptitle('changing the s = {0.1, 1, 2, 3, 5} with fixed kernel size k = 11', fontsize=16)
    a = fig1.add_subplot(2, 2, 1)
    imgplot = plt.imshow(sigma0,cmap='gray')
    a.set_title('sigma_change0.1')
    a = fig1.add_subplot(2, 2, 2)
    imgplot = plt.imshow(sigma1,cmap='gray')
    a.set_title('sigma_change1')
    a = fig1.add_subplot(2, 2, 3)
    imgplot = plt.imshow(sigma2,cmap='gray')
    a.set_title('sigma_change2')
    a = fig1.add_subplot(2, 2, 4)
    imgplot = plt.imshow(sigma3,cmap='gray')
    a.set_title('sigma_change3')
    
    fig1 = plt.figure(figsize = (10,10))
    a = fig1.add_subplot(1, 2, 1)
    imgplot = plt.imshow(sigma5,cmap='gray')
    a.set_title('sigma_change5')


if __name__ == "__main__":
    main()
