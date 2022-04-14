
import torch
import cv2




class DetectObject:
    """
    The class is to detect items in image
    """
    def __init__(self,classes=[0]):
       
        self.model = self.load_model()
        self.model.conf = 0.3 # set inference threshold at 0.3
        self.model.iou = 0.3 # set inference IOU threshold at 0.3
        self.model.classes = classes # set model to only detect "Human" class
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def load_model(self):
        """
        Function loads the yolo5 model from PyTorch Hub.
        """
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='weights/best.pt')  # default

        return model

    def score_frame(self, frame):
        """
        function scores each frame of the video and returns results.
        :param frame: frame to be infered.
        :return: labels and coordinates of objects found.
        """
        self.model.to(self.device)
        results = self.model([frame])
        labels, cord = results.xyxyn[0][:, -1].to('cpu').numpy(), results.xyxyn[0][:, :-1].to('cpu').numpy()
        return labels, cord

    def plot_boxes(self, results, frame):
        """
        plots boxes and labels on frame.
        :param results: inferences made by model
        :param frame: frame on which to  make the plots
        :return: new frame with boxes and labels plotted.
        """
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        items =[]
        for i in range(n):
            row = cord[i]
            x1, y1, x2, y2,confidence = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape),row[4]
            # w = x2-x1
            # h = y2-y1
            item = [x1,y1,x2,y2,confidence]
            items.append(item)
            bgr = (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 1)
            label = f"{int(row[4]*100)}"
            cv2.putText(frame, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            cv2.putText(frame, f"Total Targets: {n}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return frame,items
    def get_bb(self,image):
        # image = cv2.imread(image)
        results = self.score_frame(image)
        image,items = self.plot_boxes(results, image)
        return image,items
    


    
# a = DetectObject()#class item
# image,items = a.get_bb('missing_barcode/IMG_20220303_174028.jpg')
# # print(items)
# cv2.imwrite("output.jpg", image)
# cv2.imshow('test',image)
# cv2.waitKey(0)