#@author- Sumit Rai

import cv2 as cv
import numpy as np
from facenet_pytorch import MTCNN
from PIL import Image
import matplotlib
import tqdm


mtcnn = MTCNN(select_largest=False,post_process=False,device='cpu')

cap = cv.VideoCapture(0)

while True:
    _, frame = cap.read()
    frame = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
    face = Image.fromarray(frame)
    face = mtcnn(frame)
    boxes, probs,landmarks = mtcnn.detect(frame,landmarks=True)
    for box,landmark in zip(boxes,landmarks):
        x1,y1,x2,y2 = box[0],box[1],box[2],box[3]
        cv.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),1)
        for i in landmark:
            x,y = i[0],i[1]
            cv.circle(frame,(x,y),2,(255,0,0),-1)
    img = face.permute(1,2,0).int().numpy()
    img = np.uint8(img)
    if img is not None:
        img = cv.cvtColor(img,cv.COLOR_RGB2BGR)
        cv.imshow('mtcnn',img)
    
    else:
        print('Empty Frame')
        exit(1)

    frame = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
    cv.imshow('frame',frame)
    
    if cv.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv.destroyAllWindows()
    