#@author- Sumit Rai
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from torchvision import transforms
from PIL import Image
from skimage import io

import cv2 as cv
import numpy as np
from facenet_pytorch import MTCNN
from PIL import Image
import matplotlib
import tqdm
import torch 
import torch.nn as nn
from torchvision import transforms

from facenet_pytorch import InceptionResnetV1

mtcnn = MTCNN(select_largest=False,post_process=False,device='cpu')


class Flatten(nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self,x):
    x = x.view(x.size(0),-1)
    return x

class normalize(nn.Module):
  def __init__(self):
    super(normalize,self).__init__()

  def forward(self,x):
    x = F.normalize(x,p=2,dim=1)
    return x

PATH = 'entire_model.pt'

def eval(file):
    image = io.imread(file)
    image = cv.resize(img,(160,160)).transpose((2,1,0))
    output = loaded_model(torch.tensor(image[np.newaxis,...]).float())[0].squeeze().detach()
    return output


filename = 'model_2.0.pth'
loaded_model = pickle.load(open(filename, 'rb'))

trasform = transforms.Compose([transforms.Resize(160)])

print(loaded_model)







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
    print(face.shape)
    if img is not None:
        img = cv.cvtColor(img,cv.COLOR_RGB2BGR)
        cv.imshow('mtcnn',img)
    
    else:
        print('Empty Frame')
        exit(1)

    input_img = face.unsqueeze(1)
    cv.imwrite('input_img.jpg',img)
    print(input_img.shape)
    output = eval('input_img.jpg')
    print(output)
    pred,_= torch.max(output,0)
    print(_)
    val = _.item()
    name = ''

    if val == 1:
        name = 'shivansh'

    elif val ==2:
        name = 'shreyas'

    elif val == 3:
        name = 'Sumit'

    elif val == 4:
        name = 'Vidhi'

    else:
        pass

    cv.putText(frame,name,(50,50),cv.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2 )
    frame = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
    cv.imshow('frame',frame)
    
    if cv.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv.destroyAllWindows()
    
