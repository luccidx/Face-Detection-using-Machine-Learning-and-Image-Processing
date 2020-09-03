# Created By G Santosh Kumar

import cv2
import numpy as np
import os

faceClassifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
def CreateDataSet():
    cam = cv2.VideoCapture(0)
    Sam = 0
    while True:
        Con, img = cam.read()
        if Con:
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = faceClassifier.detectMultiScale(gray_image, 1.3, 6)
            for x,y,w,h in faces:
                face = gray_image[y:y+h, x:x+w]
                face = cv2.resize(face, (200, 200))
                
                cv2.imwrite("Images/user.{}.jpg".format(Sam), face)
                Sam = Sam + 1
                face = cv2.putText(face, str(Sam), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                cv2.imshow("My Face", face)
            
            
            if Sam == 100:
                print("We Collected Your Faces...")
                break
                
            if cv2.waitKey(1) == 13:
                break
    cam.release()
    cv2.destroyAllWindows()

CreateDataSet()

path = "Images/"
all_images = os.listdir(path)

Training_Data = []
Labels = np.arange(1, len(all_images)+1)
Labels = np.asarray(Labels, dtype = np.int32)
for i in all_images:
    if i[-4:] == ".jpg":
        img_path = path + i
        print(img_path)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if len(image) < 1:
            print(i, img_path)
        Narray = np.asarray(image, dtype = np.uint8)
        Training_Data.append(Narray)

Face_Model = cv2.face_LBPHFaceRecognizer.create()
Face_Model.train(Training_Data, Labels)
print("We Trained Your Machine with Your Faces.")


#import subprocess
from gtts import gTTS
from playsound import playsound
faceClassifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cam = cv2.VideoCapture(0)
while True:
    Con, img = cam.read()
    if Con:
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceClassifier.detectMultiScale(gray_image, 1.3, 6)
        for x,y,w,h in faces:
            face = gray_image[y:y+h, x:x+w]
            face = cv2.resize(face, (200, 200))

            pred = Face_Model.predict(face)
            print(pred)
            if pred[1]< 42:
                face = cv2.putText(img, "Hey! Santosh", (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
                #var = gTTS(text = "Hey Santosh , Whatsup? ",lang = 'en') 
                #var.save('m.mp3') 
                #playsound('.\m.mp3')   
                #subprocess.call(["say", "Hey Santosh. How Can i Help You.."],shell=True)
            else:
                face = cv2.putText(img, "Unknown Face @", (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
                cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)
        cv2.imshow("MyFace", img)
        if cv2.waitKey(1) == 13:
            break
cam.release()
cv2.destroyAllWindows()