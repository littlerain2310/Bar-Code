import numpy as np
import cv2
import imutils
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(20, 20))
rows = 2
columns = 2


image = cv2.imread("all barcode/IMG_20220303_175539.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# blur = cv2.GaussianBlur(gray,(3,3),0)
# equalize lighting
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
gray = clahe.apply(gray)
# edge enhancement
edge_enh = cv2.Laplacian(gray, ddepth = cv2.CV_8U, 
                         ksize = 3, scale = 1, delta = 0)
# edge_enh = cv2.Canny(blur,168,234)

retval = cv2.imwrite("edge_enh.jpg", edge_enh)
# edge image
fig.add_subplot(rows, columns, 1)
plt.imshow(edge_enh,cmap='gray')
plt.axis('off')
plt.title('edge')
# bilateral blur, which keeps edges
blurred = cv2.bilateralFilter(edge_enh, 13, 50, 50)

# use simple thresholding. adaptive thresholding might be more robust
thresh = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
thresh = cv2.bitwise_not(thresh)
retval = cv2.imwrite("thresh.jpg", thresh)

fig.add_subplot(rows, columns, 2)
plt.imshow(thresh,cmap='gray')
plt.axis('off')
plt.title('thresh')
# do some morphology to isolate just the barcode blob
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
closed = cv2.erode(closed, None, iterations = 4)
closed = cv2.dilate(closed, None, iterations = 4)


retval = cv2.imwrite("closed.jpg", closed)
# close image
fig.add_subplot(rows, columns, 3)
plt.imshow(closed,cmap='gray')
plt.axis('off')
plt.title('close')
# find contours left in the image
cnts = cv2.findContours(closed.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

cv2.drawContours(image, cnts, -1, (0,255,0), 3)

retval = cv2.imwrite("found.jpg", image)
# found image
fig.add_subplot(rows, columns, 4)
plt.imshow(image)
plt.axis('off')
plt.title('found')

plt.show()