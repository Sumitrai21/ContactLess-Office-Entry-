import os
import numpy as np
from facenet_pytorch import MTCNN
import cv2 as cv
from PIL import Image

root = 'sumit_rai/'

name_list = os.listdir(root) 

mtcnn = MTCNN(select_largest=False,post_process=False,device='cpu')

for i in name_list:
    final_name = os.path.join(root,i)
    img = cv.imread(final_name)
    img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
    face = Image.fromarray(img)
    try:
        face = mtcnn(face)
        img = face.permute(1,2,0).int().numpy()
        img = np.uint8(img)
        if img is not None:
      #  save_img = np.float32(img)
            save_img = cv.cvtColor(img,cv.COLOR_RGB2BGR)
            cv.imwrite('sumit/'+i,save_img)

        else:
            print('Empty')
            exit(1)

    except:
        pass





    