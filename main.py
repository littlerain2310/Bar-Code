
import argparse
from detectBarcode import DetectBarcode
from detectItem import DetectObject
import cv2
from util import item_contain_barcode


parser = argparse.ArgumentParser()
parser.add_argument('--dir',default='' ,type=str)

parser.add_argument('--image',default='', type=str)

args = parser.parse_args()

item_detector = DetectObject()
barcodes_detector = DetectBarcode()
dir_path = args.dir
image_path = args.image
    
barcode_list,result,image_origin = barcodes_detector.get_barcode(image_path)

image_draw = image_origin.copy()
_,items_bbs = item_detector.get_bb(image_origin)

for barcode in barcode_list:
    x1,y1,x2,y2 = barcode.box
    valid = barcode.valid
    if valid:
        cv2.rectangle(image_draw, (x1,y1 ), (x2 , y2), (255,0,0), 3)
    else:
        cv2.rectangle(image_draw, (x1,y1 ), (x2 , y2), (0,255,255), 3)

for item_bb in items_bbs:
    x1,y1,x2,y2,conf = item_bb
    box = [x1,y1,x2,y2]
    contain = False
    for barcode in barcode_list:
        if item_contain_barcode(box,barcode.center_point):
            contain = True
    if not contain:
        cv2.rectangle(image_draw, (x1,y1),(x2,y2),(0,0,255), 3)

print(result)

image_draw = cv2.resize(image_draw,(800,800))
cv2.imshow('output.jpg',image_draw)
cv2.waitKey(0)

