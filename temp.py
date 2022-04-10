from kraken import binarization
from PIL import Image
from pyzbar.pyzbar import decode
from pyzbar.pyzbar import ZBarSymbol
import numpy as np
import cv2
import matplotlib.pyplot as plt




image_path = "all_barcode/IMG_20220303_173611.jpg"
# binarization using kraken
im = Image.open(image_path)


bw_im = binarization.nlbin(im)
open_cv_image = np.array(bw_im) 
retval = cv2.imwrite("binary.jpg", open_cv_image)
