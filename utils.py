from pyzbar import pyzbar
import cv2
import numpy as np
def decode_barcode(barcode_img):
    decoded = pyzbar.decode(barcode_img)
    data = ", ".join([d.data.decode("utf-8") for d in decoded])
    return data

def count_lines(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(11,11),0)


    # kernel = np.ones((7, 7), np.uint8)
    
    edges = cv2.Canny(blur,49,155,apertureSize = 3)
    # dilate = cv2.dilate(edges,kernel,iterations = 1)
    
    minLineLength = 100
    maxLineGap = 10
    lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
    if lines is not None:
        return len(lines)
    return 0
    