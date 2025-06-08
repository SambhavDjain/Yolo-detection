import os
import cv2 as cv
import numpy as np 
import util
from  util import get_outputs 
import matplotlib.pyplot as plt



#DEFINE THE CONSTANT  
model_cfg_path = os.path.join('.', 'models','cfg','yolov3.cfg')
model_weights_path = os.path.join('.', 'models' , 'weights' , 'yolov3.weights')

class_names_path = os.path.join('.' , 'models' , 'class.names')
#image_path =  './venv/pexels-diana-huggins-615369.jpg'
image_path = './venv/cat.png'

#LOAD CLASS NAMES

class_names = []
with open(class_names_path, 'r') as f:
    class_names =[j[:-1] for j in f.readlines() if len(j) > 2 ]#j[:1]-takes 1st character f.readlines():Reads all lines from a file object f into a list.
    f.close()#j[:-1] uses -1 becoz for the last character like its giving the first character of cat as c and to complte useingg -1 can help if leave blank then ?appers



#LOADING THE MODEL
net = cv.dnn.readNetFromDarknet(model_cfg_path,model_weights_path)

#LOAD IMAGE

img = cv.imread(image_path)
H,W,_ = img.shape 


#CONVERT IMAGE 

blob = cv.dnn.blobFromImage(img, 1/255, (320,320),(0,0,0),True)#1/255 is scale  320,320 is size

#GET DETECTIONS 

net.setInput(blob)

detections = util.get_outputs(net)

#BBOXES , CLASS_IDS, CONFIDENCE

bboxes = []
class_ids = []
scores = []

for detection in detections:
    #[x1,x2,x3,x4,x5,x6... , x85]
    #print(detections)
    
    bbox = detection[:4] #for up to 4th index 


    xc,yc,w,h = bbox #x,y are cordinates and w,h are height and width
    bbox = [int(xc*W), int(yc*H), int(w*W),int(h*H)] 

    bbox_confidence = detection[4] #for index value 

    class_id = np.argmax(detection[5:]) #Returns the index of the max value 

    score = np.amax(detection[5:])#Returns the max value 


    bboxes.append(bbox)
    class_ids.append(class_id)
    scores.append(score)
   



#APPLY NMS -NON MAXIMUM SUPPRESSION 

bboxes,class_ids, scores = util.NMS(bboxes,class_ids, scores)

print(len(bboxes))

#PLOT
for bbox_,bbox in enumerate(bboxes) :
    xc,yc,w,h = bbox
    cv.putText(img,class_names[class_ids[bbox_]],(int(xc - (w/2)),
                                                  int(yc + (h/2)-100)),
                                                  cv.FONT_HERSHEY_SIMPLEX,
                                                  7,(0,0,255),15)


    img = cv.rectangle(img,
                       (int(xc - (w/2)),int(yc - (h/2))),
                       (int(xc + (w/2)),int(yc + (h/2))),
                       (0,255,0),
                       15)
plt.imshow(cv.cvtColor(img,cv.COLOR_BGR2RGB))
plt.show()
