<h1><b>Barcode</b></h1>
Program use image processing technique (binary,gradient,morphological,threshold,etc.) to find barcode candidate region.
Then I use Hough transform to detect line to sort for real barcode.
Barcode then is fed into pyzbar to decode barcode.If it cant be decoded then it will be marked yellow.Otherwise,region is marked with Blue.
I use yolov5 to detect object in Image.Any item without barcode inside box will be marked red. 


<h2>Key results</h2>

![IMG_20220303_175324](https://user-images.githubusercontent.com/56443812/163321470-68a692b2-0c22-4285-bbc2-2ed2b4a06aa9.jpg)

![IMG_20220303_175451](https://user-images.githubusercontent.com/56443812/163321480-4882cfd9-7ec4-4aa9-9326-207377c31844.jpg)

![IMG_20220303_173611](https://user-images.githubusercontent.com/56443812/163321599-3a6cf7b7-0470-4bcb-b39c-8bdaa4677510.jpg)

<h2>Remainning Results</h2>
In "results" folder.

Decoded results for barcodes is saved in a dict format("Value": "repeated time")

<h2>Run</h2>

<b>Install Requirements</b><br>

pip install -r requirements.txt<br>

<b>Run the test</b><br>

python main.py -- image {image_path}
where images_path is path to image

<i>for example</i>:
python3 main.py --image partial_barcode/IMG_20220303_174238.jpg
<h1>Conclusion</h1>
It works with some objects.

The program show a lot of false positive region of barcode,which is not good.The reason is normal image technique cannot solve every cases due to variety of lightning and color of object.This problem can be solved by using Deep Learning techinique Object Detection like Yolo.

Some barcode is correctly detected but not decoded properly.It might be the limit of pyzbar
