import numpy as np
import cv2
import os
import time

imfolder  = "images_original/"
outfolder = "images/"
outfolder_mask = "mask/"

h = 500
w = 500
hmin = (720 - h)/2
wmin = (1280 - w)/2

for file in os.listdir(imfolder):
    A = cv2.imread("images_original/{}".format(file))
    w, h, _ = A.shape
    D = np.zeros((w,h,4))
    lab = cv2.cvtColor(A, cv2.COLOR_RGB2LAB)
    ab = lab[:,:,1:3]
    numColors = 2
    
    data = ab.reshape((-1,2))
    data = np.float32(data)
    
    criteria = (cv2.TERM_CRITERIA_EPS +cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_PP_CENTERS

    compactness, L2, centers = cv2.kmeans(data,numColors,None,criteria, 10, flags)
    print("L2", L2.shape)
    
    L2 = np.reshape(L2, (w, h))
    ma = np.sum(ab[:,:,0]*L2)/np.sum(L2)
    mb = np.sum(ab[:,:,1]*L2)/np.sum(L2)
    print(L2[0,0])
    if L2[0,0]==1:
        L2[L2==1] = 2
        L2[L2==0] = 1
        L2[L2==2] = 0
    D[:,:,0] = A[:,:,0]*L2
    D[:,:,1] = A[:,:,1]*L2
    D[:,:,2] = A[:,:,2]*L2
    D[:,:,3] = L2*255
    alpha = np.float32(L2)
    mask = np.uint8(255*L2)
    mask3 = np.zeros((w,h,4))
    mask3[:,:,0] = mask
    mask3[:,:,1] = mask
    mask3[:,:,2] = mask
    mask3[:,:,3] = L2*255
    #cv2.imshow('imshow',D)
    #key = cv2.waitKey(0)
    cv2.imwrite("images/{}".format(file), D)
    cv2.imwrite("mask/{}".format(file), mask3)