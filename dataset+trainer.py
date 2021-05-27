## This file will first take set of photos and then store it in the specified "DATASET" folder
## Then it'll train those set of photos and create a trainer yml file

import os
import cv2
import numpy as np
from PIL import Image

face_classifier = cv2.CascadeClassifier('C:/Users/DELL/Desktop/python/working/haarcascade_frontalface_default.xml')
recognizer=cv2.face.LBPHFaceRecognizer_create()
path='C:/Users/DELL/Desktop/python/working/DATASET/'
a=input("enter ur id number")

def face_extractor(img):
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)

    if faces is():
        return None

    for(x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+w]

    return cropped_face


cap = cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = cap.read()
    if face_extractor(frame) is not None:
        count+=1
        face = cv2.resize(face_extractor(frame),(300,300))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
 
        file_name_path = 'C:/Users/DELL/Desktop/python/working/DATASET/User.'+a+'.'+str(count)+'.jpg'
        cv2.imwrite(file_name_path,face)

        cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.imshow('Face Cropper',face)
    else:
        print("Face not Found")
        pass

    if cv2.waitKey(1)==13 or count==60:
        break

cap.release()
cv2.destroyAllWindows()
print('Colleting Samples Complete!!!')


def getImagesWithID(path):
	imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
	faces=[]
	IDs=[]
	for imagePath in imagePaths:
		faceImg=Image.open(imagePath).convert('L');
		faceNp=np.array(faceImg,'uint8')
		ID=int(os.path.split(imagePath)[-1].split('.')[1])
		faces.append(faceNp)
		print (ID)
		IDs.append(ID)
		cv2.imshow("training",faceNp)
		cv2.waitKey(10)
	return IDs,faces

Ids,faces=getImagesWithID(path)
recognizer.train(faces,np.array(Ids))
recognizer.save('C:/Users/DELL/Desktop/python/working/trainningData.yml')
cv2.destroyAllWindows()