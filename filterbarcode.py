import cv2
import numpy as np

img = cv2.imread('candidate/17.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# blur = cv2.GaussianBlur(gray,(11,11),0)

# (_, thresh) = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)

kernel = np.ones((3, 3), np.uint8)
# opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

# erosion = cv2.erode(opening, kernel, iterations=1)

# cv2.imshow("edges",erosion)
# cv2.waitKey()
edges = cv2.Canny(gray,45,155,apertureSize = 3)
closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
# erosion = cv2.erode(edges, kernel, iterations=1)
dilate = cv2.dilate(edges,kernel,iterations = 1)

minLineLength = 100
maxLineGap = 50
lines = cv2.HoughLinesP(closing,1,np.pi/180,100,minLineLength,maxLineGap)
if lines is not None:
    print(len(lines))
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

cv2.imshow('houghlines3.jpg',img)
cv2.waitKey(0)