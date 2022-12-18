import numpy as np
import cv2
import os
import argparse
import sys

def parse_args():
    parser = argparse.ArgumentParser(description=" use two image to get a criteria judgement standard")
    parser.add_argument("--p1", default=" ", help="photo1 ")
    parser.add_argument("--p2", default=" ", help=" photo2 ")

    args = parser.parse_args()
    return args

##输入的为白色的主体图片
##返回的是 iou
def img_reshape(filename1, filename2):
    img1 = cv2.imread(filename1)
    img2 = cv2.imread(filename2)
    w1,h1,_ = img1.shape
    w2,h2,_ = img2.shape
    alpha1 = img1[:,:,1]
    alpha2 = img2[:,:,1]
    alpha1[alpha1 != 0] = 1
    alpha2[alpha2 != 0] = 1
    area1 = np.sum(alpha1)
    area2 = np.sum(alpha2)
    ##print(type(img1))
    ##print(area1,area2)
    ##第一个图片大的情况，最好默认第一个大
    if area1 > area2:
        scale = int(np.sqrt(area1/area2))
        img2new = cv2.resize(img2, dsize=None, dst=None, fx=scale,fy=scale,interpolation=None)
        '''cv2.imshow("img2new", img2new)
        cv2.imshow("img2", img2)
        cv2.imshow("img1", img1)'''
        w2new, h2new,_ = img2new.shape
        ##img1的图形主体的中心
        center = center_find(img1,w1,h1)
        ##图形裁剪 左上与右下坐标
        left1  = np.uint64(center[0]-w2new/2)
        right1 = np.uint64(left1 + w2new)
        left2  = np.uint64(center[1]-h2new/2)
        right2 = np.uint64(left2 + h2new)
        ##print(w2new,h2new)
        ##裁剪图片
        img1new = img1[left1:right1, left2:right2, 1]
        ##逻辑判断 转 int
        u = np.logical_or(img1new,img2new[:,:,1])+0
        i = np.logical_and(img1new,img2new[:,:,1])+0
        usum =np.sum(u)
        isum =np.sum(i)
        iou = isum/usum
        print("iou",iou)
    if area1 <= area2:
        scale = int(np.sqrt(area2/area1))
        img1new = cv2.resize(img1, dsize=None, dst=None, fx=scale,fy=scale,interpolation=None)
        '''cv2.imshow("img2new", img2new)
        cv2.imshow("img2", img2)
        cv2.imshow("img1", img1)'''
        w1new, h1new,_ = img1new.shape
        ##img1的图形主体的中心
        center = center_find(img2,w2,h2)
        ##图形裁剪 左上与右下坐标
        left1  = np.uint64(center[0]-w1new/2)
        right1 = np.uint64(left1 + w1new)
        left2  = np.uint64(center[1]-h1new/2)
        right2 = np.uint64(left2 + h1new)
        ##print(w2new,h2new)
        ##裁剪图片
        img2new = img2[left1:right1, left2:right2, 1]
        ##逻辑判断 转 int
        u = np.logical_or(img2new,img1new[:,:,1])+0
        i = np.logical_and(img2new,img1new[:,:,1])+0
        usum =np.sum(u)
        isum =np.sum(i)
        iou = isum/usum
        print("iou",iou)
    ##key = cv2.waitKey(0)
    ##print(scale)
    return iou
    ''' 未完待续    '''

def center_find(g1,w1,h1):
    sumi = 0
    sumj = 0
    num = 0
    for i in range(0,720):
        for j in range(0,1280):
            if g1[i,j,1]==1:
                sumi = sumi + i
                sumj = sumj + j
                num  = num + 1
    ##print(sumi,sumj,num)
    center = np.zeros(2)
    center[0] = sumi / num
    center[1] = sumj / num
    center = np.uint64(center)
    ##print(center)
    return center
    ''' 未完待续    '''

def main(args):
    if args.p1 == " ":
        print("  Error: please input image1")
        sys.exit(1)
    if args.p2 == " ":
        print("  Error: please input image2")
        sys.exit(1)
    ##第一个是大图片
    ##第二个是小的说明书图片
    fill_name1 = args.p1
    fill_name2 = args.p2
    img_reshape(fill_name1,fill_name2)

if __name__ == '__main__':
    args = parse_args()
    score = main(args)