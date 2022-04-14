import numpy as np
import cv2
import imutils
from PIL import Image
from kraken import binarization
from util import *
from collections import Counter

class Barcode:
    def __init__(self,box,center_point,valid = True):
        self.box = box
        self.center_point = center_point
        self.valid = valid

class DetectBarcode:
    def __init__(self,margin = 0) :
        self.MARGIN = margin
        self.result = []
        self.barcodes = []
        pass
    def get_image(self):
        '''Take image_path as path to image 
        output: binary output and origin image'''
        image = Image.open(self.image_path)
        bw_im = binarization.nlbin(image)
        self.image = np.array(image)
        self.binary = np.array(bw_im)
        self.orgin = self.image.copy()

    def get_barcode(self,image_path):
        # take input path
        self.image_path = image_path
        self.get_image()

        #calculate gradient
        gradX = cv2.Sobel(self.binary, ddepth = cv2.CV_32F, dx = 1, dy = 0, ksize = 5)
        gradY = cv2.Sobel(self.binary, ddepth = cv2.CV_32F, dx = 0, dy = 1, ksize = 5)
        gradient = cv2.subtract(gradX, gradY)
        gradient = cv2.convertScaleAbs(gradient)
        
        #process the gradient to get barcode region
        blur = cv2.GaussianBlur(gradient,(11,11),0)
        (_, thresh) = cv2.threshold(blur, 170, 255, cv2.THRESH_BINARY)
        kernel = np.ones((11,11),np.uint8)
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        closed = cv2.erode(closed, kernel, iterations = 4)
        closed = cv2.dilate(closed, kernel, iterations = 4)

        #get candidate contours
        cnts = cv2.findContours(closed.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        self.cnts = imutils.grab_contours(cnts)

        #filter contours
        self.filtered_cnts = self.filter_cnts()

        #decode filter contours and 
        self.decode()
        count =dict(Counter(self.result))

        return self.barcodes,count,self.orgin

    def filter_cnts(self):
        """Filter the given contours"""
        filtered_cnts =[]
        count = 0
        for cnt in self.cnts:
            (x, y, w, h) = cv2.boundingRect(cnt)
            crop = self.orgin[y - self.MARGIN:y+h+self.MARGIN, x-self.MARGIN:x+w+self.MARGIN]
            lines = count_lines(crop)
            count +=1
            
            if lines > 20 :
                # print(lines)
                filtered_cnts.append(cnt)
        return filtered_cnts

    def decode(self):
        for cnt in self.filtered_cnts:
            (x, y, w, h) = cv2.boundingRect(cnt)
            crop = self.image[y:y+h, x:x+w]
            data = decode_barcode(crop)
            box = [x,y,x+w,y+h]
            center_x = x + w/2
            center_y = y+h/2
            center_point = [center_x,center_y]
            if data == '':
                barcode = Barcode(box,center_point,False)
                self.barcodes.append(barcode)
            else:
                self.result.append(data)
                barcode = Barcode(box,center_point)
                self.barcodes.append(barcode)
    