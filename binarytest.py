import numpy as np
import cv2
import imutils
import matplotlib.pyplot as plt
from PIL import Image
from kraken import binarization
import argparse
from utils import *
from collections import Counter

MARGIN =0

fig = plt.figure(figsize=(50, 35))
rows = 2
columns = 3
result = []

parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str)
args = parser.parse_args()
image_path = args.image

image = Image.open(image_path)
bw_im = binarization.nlbin(image)
image = np.array(image)
gray = np.array(bw_im)


# gray = cv2.cvtColor(binary, cv2.COLOR_BGR2GRAY)
gradX = cv2.Sobel(gray, ddepth = cv2.CV_32F, dx = 1, dy = 0, ksize = 5)
gradY = cv2.Sobel(gray, ddepth = cv2.CV_32F, dx = 0, dy = 1, ksize = 5)
gradient = cv2.subtract(gradX, gradY)
gradient = cv2.convertScaleAbs(gradient)
fig.add_subplot(rows, columns, 1)
plt.imshow(gradient,cmap='gray')
plt.axis('off')
plt.title('gradient')
blur = cv2.GaussianBlur(gradient,(11,11),0)
# blur = cv2.bitwise_not(blur)

fig.add_subplot(rows, columns, 2)
plt.imshow(blur,cmap='gray')
plt.axis('off')
plt.title('blur')

(_, thresh) = cv2.threshold(blur, 170, 255, cv2.THRESH_BINARY)

fig.add_subplot(rows, columns, 3)
plt.imshow(thresh,cmap='gray')
plt.axis('off')
plt.title('thresh')

kernel = np.ones((11,11),np.uint8)
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
closed = cv2.erode(closed, kernel, iterations = 4)
closed = cv2.dilate(closed, kernel, iterations = 4)




fig.add_subplot(rows, columns, 4)
plt.imshow(closed,cmap='gray')
plt.axis('off')
plt.title('closed')

cnts = cv2.findContours(closed.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
filtered_cnts =[]
count = 0
for cnt in cnts:
    
    (x, y, w, h) = cv2.boundingRect(cnt)
    crop = image[y - MARGIN:y+h+MARGIN, x-MARGIN:x+w+MARGIN]
    gray = cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
    lines = count_lines(crop)
    count +=1
    candidate = cv2.imwrite(f"candidate/{count}.jpg", crop) 
    if lines >0 :
               
        filtered_cnts.append(cnt)
    

for cnt in filtered_cnts:
    (x, y, w, h) = cv2.boundingRect(cnt)
    crop = image[y:y+h, x:x+w]
    data = decode_barcode(crop)
    if data == '':
        cv2.rectangle(image, (x-MARGIN, y-MARGIN), (x + w+MARGIN, y + h + MARGIN), (0,255,255), 3)
    else:
        result.append(data)
        cv2.rectangle(image, (x-MARGIN, y-MARGIN), (x + w+MARGIN, y + h + MARGIN), (0,255,0), 3)

count =dict(Counter(result))

fig.add_subplot(rows, columns, 5)
plt.imshow(image,cmap='gray')
plt.axis('off')
plt.title('draw image')
retval = cv2.imwrite("output.jpg", image)
print(count)

# plt.show()

