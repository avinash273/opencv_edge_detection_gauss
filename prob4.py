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

def Noise(gray_img):
    noise =  np.random.normal(loc=0, scale=0.1, size=gray_img.shape)
    for row in range(noise.shape[0]):
        for col in range(noise.shape[1]):
            if noise[row,col] > 0.2:
                noise[row,col] = 1
            else:
                noise[row,col] = 0
    noisy = gray_img + noise
    return noisy



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
    size = 5
    image1=np.copy(image)
    for i in np.arange(int(size//2), image1.shape[0]-int(size//2)):
        for j in np.arange(int(size//2), image1.shape[1]-int(size//2)):
            neighbors = []
            for k in np.arange(-2, 3):
                for l in np.arange(-2, 3):
                    a = image1.item(i+k, j+l)
                    neighbors.append(a)
            neighbors.sort()
            median = neighbors[12]
            b = median
            image1.itemset((i,j), b)
    return image1

#main function
def main():
    original = mpimg.imread('Lena.png')
    gray_img = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    nosiy_new =  Noise(gray_img)
    gray_img1 = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    noise1 =  np.random.normal(loc=0, scale=0.1, size=gray_img.shape)
    only_noisenimage = noise1 + gray_img1
    gauss_smoothed_noise = myGaussianSmoothing(only_noisenimage, 11, 3)
    median_smoothed_noise = medianFiltering(only_noisenimage,3)
    gauss_smoothed_noise_new = myGaussianSmoothing(nosiy_new, 11, 3)
    median_smoothed_noise_new = medianFiltering(nosiy_new,3)
    
    
    fig1 = plt.figure(figsize = (12,12))
    fig1.suptitle('Gaussian and Median Filtering with Noise', fontsize=14)
    a = fig1.add_subplot(3, 2, 1)
    imgplot = plt.imshow(only_noisenimage,cmap='gray')
    a.set_title('Gauss Noise Added')
    
    a = fig1.add_subplot(3, 2, 3)
    imgplot = plt.imshow(gauss_smoothed_noise,cmap='gray')
    a.set_title('Gauss Noise Gauss filtered')
    
    a = fig1.add_subplot(3, 2, 5)
    imgplot = plt.imshow(median_smoothed_noise,cmap='gray')
    a.set_title('Gauss Noise Median filtered')
    
    a = fig1.add_subplot(3, 2, 2)
    imgplot = plt.imshow(nosiy_new,cmap='gray')
    a.set_title('New noice>0.2 [0,1]')
    
    a = fig1.add_subplot(3, 2, 4)
    imgplot = plt.imshow(gauss_smoothed_noise_new,cmap='gray')
    a.set_title('New noice Gauss Filtered')
    
    a = fig1.add_subplot(3, 2, 6)
    imgplot = plt.imshow(median_smoothed_noise_new,cmap='gray')
    a.set_title('New noice Median Filtered')
    
    

if __name__ == "__main__":
    main()
