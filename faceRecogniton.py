import cv2
from math import ceil
import sys
import os
import numpy as np

cascadePath = 'haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascadePath)
cap = cv2.VideoCapture(0)
recognizer = cv2.face.LBPHFaceRecognizer_create()
min_confidence = 20
trained = 0

def getNewLabel(name):
	current_directory = os.getcwd()
	final_directory = os.path.join(current_directory, 'dataset', name)
	if not os.path.exists(final_directory):
		os.makedirs(final_directory)    	
	return final_directory

def readData():
	images = []
	image_label = []
	names = []
	cd = os.getcwd()
	dataset_dir = os.path.join(cd, 'dataset')
	folders = os.listdir(dataset_dir)
	for i in range(len(folders)):
		names.append(folders[i]) 
		wd = os.path.join(dataset_dir,folders[i])
		folder_imgs = os.listdir(wd)
		for j in folder_imgs:
			im = cv2.imread(os.path.join(wd,j),0)
			faces = faceCascade.detectMultiScale(im, 1.1, 5, minSize = (50,50))
			for (x,y,w,h) in faces:
				im_arr = np.array(im[x:x+w,y:y+h],'uint8')
				images.append(im_arr)
				image_label.append(i)
				
	cv2.destroyAllWindows()
	return images, image_label, names

image_data, labels, names = readData()
if(image_data!=[]):
	recognizer.train(image_data, np.array(labels))
	trained = 1;

font = cv2.FONT_HERSHEY_DUPLEX

c = 0

while True:
	ret,frame = cap.read()
	if(c==1):
		cv2.imwrite(final_path,frame)
		c = 0
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = faceCascade.detectMultiScale(gray, 1.1, 5, minSize = (50,50))
	for (x,y,w,h) in faces:
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
		test = gray[x:x+w,y:y+h]
		test_img = np.array(test,'uint8')
		if(trained==1):
			if(test_img.any()):
				index, confidence = recognizer.predict(test_img)
			if(confidence>=min_confidence):
				#Adding label to detected face
				cv2.putText(frame,names[index],(x,y-20),font,.6,(0,255,0))
				cv2.putText(frame,"Confidence: "+str(ceil(confidence))+"%",(x,y+h+20),font,.5,(0,255,255))
	cv2.imshow('Video', frame)
	k = cv2.waitKey(1) & 0xFF
	if k == ord('q') or k==27:
		break
	elif k == ord('c') or k == ord('C'):
		print('Enter label name for this capture: ',end="")
		newlabel = input()
		working_dir = getNewLabel(newlabel)
		files = os.listdir(working_dir)
		number = len(files) + 1
		final_path = os.path.join(working_dir, str(number)+'.png')
		c = 1

cap.release()
cv2.destroyAllWindows()