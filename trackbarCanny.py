import cv2
import numpy as np 
import matplotlib.pyplot as plt 

def callback(x):
    print(x)

img = cv2.imread('candidate/16.jpg')
gray = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(11,11),0)

canny = cv2.Canny(blur, 50, 150,apertureSize = 3) 

cv2.namedWindow('image') # make a window with name 'image'
cv2.createTrackbar('L', 'image', 0, 255, callback) #lower threshold trackbar for window 'image
cv2.createTrackbar('U', 'image', 0, 255, callback) #upper threshold trackbar for window 'image
kernel = np.ones((7, 7), np.uint8)
opening = cv2.morphologyEx(canny, cv2.MORPH_OPEN, kernel)
dilate = cv2.dilate(canny,kernel,iterations = 1)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
while(1):  
    l = cv2.getTrackbarPos('L', 'image')
    u = cv2.getTrackbarPos('U', 'image')

    canny = cv2.Canny(blur, l, u,apertureSize = 3)
    dilate = cv2.dilate(canny,kernel,iterations = 1)
    closing = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel)
    numpy_horizontal_concat = np.concatenate((canny, closing), axis=1) # to display image side by side
    numpy_horizontal_concat = cv2.resize(numpy_horizontal_concat,(1000,600))
    cv2.imshow('image', numpy_horizontal_concat)
    k = cv2.waitKey(1) & 0xFF
    if k == 27: #escape key
        break
    

cv2.destroyAllWindows()