import l293d.driver as l293d

import cv2
import numpy as np

##motor 1 uses pin 22,pin 18,pin 16
motor1=l293d.motor(22,18,16)

faceDetect=cv2.CascadeClassifier('C:/Users/DELL/Desktop/python/working/haarcascade_frontalface_default.xml')
cam=cv2.VideoCapture(0);
rec=cv2.face.LBPHFaceRecognizer_create();
rec.read("C:/Users/DELL/Desktop/python/working/trainningData.yml")
id=0
fontFace=cv2.FONT_HERSHEY_SIMPLEX
fontScale=1
fontColor=(255,255,255)

while(True):
	ret,img=cam.read();
	gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	faces=faceDetect.detectMultiScale(gray,1.3,5);
	for(x,y,w,h) in faces:
		cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)
		id,conf=rec.predict(gray[y:y+h,x:x+w])	
		if(id == 1):
			id = 'unknown'
		elif(id == 2):
			id = "milan"  
		elif(id==3):
			id = "Shahrukh khan"
		else:
			id="unknown"
		cv2.putText(img,str(id),(x,y+h),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255));
		if id in range(2,100):
			for i  in range(0,150):
				motor1.clockwise()
			l293d.cleanup()
	cv2.imshow("Face",img);
	if(cv2.waitKey(1)==ord('q')):
		break;
cam.release()
cv2.destroyAllWindows()


##run the  motors so visible 


