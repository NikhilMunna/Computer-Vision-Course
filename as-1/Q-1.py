import numpy as np
import cv2
img1 = cv2.imread('HW1_Q1/cat.bmp') / 255.0
img2 = cv2.imread('HW1_Q1/dog.bmp') / 255.0
cv2.imshow('image1', img1) 
cv2.imshow('image2', img2)
alpha=8
beta=8
def Kernel_guassian(sigma):
    #window size 
    n = np.ceil(sigma*6)
    y,x = np.ogrid[-n//2:n//2+1,-n//2:n//2+1]
    y_filter = np.exp(-(y*y/(2.*sigma*sigma)))
    x_filter = np.exp(-(x*x/(2.*sigma*sigma)))
    final_filter = (x_filter*y_filter) * (1/(2*np.pi*sigma**2))
    return final_filter
def low_freq(image,kernel):
    return cv2.filter2D(image,-1,kernel)
def high_freq(image,kernel):
    return image-cv2.filter2D(image,-1,kernel)
def hybridimage(image1,image2):
    return image1+image2
low_frequencies=low_freq(img1,Kernel_guassian(alpha))
cv2.imshow('Low_frequency',low_frequencies)
high_frequencies=high_freq(img2,Kernel_guassian(beta));
cv2.imshow('high_frequency',high_frequencies)
hybrid=hybridimage(low_frequencies,high_frequencies)
cv2.imshow('hybrid_image',hybrid)
