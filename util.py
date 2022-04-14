from pyzbar import pyzbar
import cv2
import numpy as np


def decode_barcode(barcode_img):
    decoded = pyzbar.decode(barcode_img)
    data = ", ".join([d.data.decode("utf-8") for d in decoded])
    return data

def count_lines(img):
    
    # Initialize output
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(9,9),0)

    invert = 255 -blur
    (_, thresh) = cv2.threshold(invert, 150, 255, cv2.THRESH_BINARY)
    # Median blurring to get rid of the noise; invert image
    binary = cv2.bitwise_not(blur)
    # Detect and draw lines

    lines = cv2.HoughLinesP(thresh, 1, np.pi/180, 10, minLineLength=50, maxLineGap=5)
    if lines is not None:
        return len(lines)
    else:
        return 0
            

def item_contain_barcode(box_item,center_barcode):
    item_x1,item_y1,item_x2,item_y2 = box_item
    center_x,center_y = center_barcode
    if (center_x > item_x1) and (center_x < item_x2) and  (center_y > item_y1) and (center_y < item_y2) :
        return True
    else:
        return False
