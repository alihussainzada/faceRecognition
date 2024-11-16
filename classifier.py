import cv2
import numpy as np
from PIL import Image
import os

print(dir(cv2.face))

folder_path = "images"
def classifier(data_dir):
    path = [os.path.join(data_dir,f) for f in os.listdir(data_dir)]
    faces = []
    ids = []
    for image in path:
        img = Image.open(image).convert('L')
        imageNP = np.array(img,'uint8')
        id = int(os.path.split(image)[1].split(".")[1]) 
        faces.append(imageNP)
        ids.append(id)
    ids = np.array(ids)
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces,ids)
    clf.write("classifier.xml")
classifier(f"{folder_path}")
    
        