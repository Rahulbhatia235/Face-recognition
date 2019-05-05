# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 19:38:43 2019

@author: ritu
"""
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 21:36:34 2019

@author: ritu
"""
import sqlite3
import cv2
import numpy as np
import os
from os import listdir
from os.path import isfile,join
data_path="C:/Users/ritu/Desktop/misc/output/"
only_files=[f for f in listdir(data_path) if isfile(join(data_path,f))]

Training_data,Label=[],[]
for i,files in enumerate(only_files):
    image_path=data_path+only_files[i]
    id=os.path.split(image_path)[-1].split('.')[1]
   
    
    images=cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    Training_data.append(np.asarray(images,dtype=np.uint8))
    
    Label.append(id)
Label=np.asarray(Label,np.int32)

model=cv2.face.LBPHFaceRecognizer_create()
model.train(np.asarray(Training_data),np.asarray(Label))
#model.save("C:/Users/ritu/Desktop/misc/recognizer/trainingdata.yml")
print("Model Training Complete")
face_classifier=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
def getProfile(id):
    conn=sqlite3.connect("C:/Users/ritu/Desktop/misc/FaceBase.db")
    cmd="SELECT * FROM People WHERE ID="+str(id)
    cursor=conn.execute(cmd)
    profile=None
    for row in cursor:
        profile=row
    conn.close()

    return profile
    
def face_detector(img):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_classifier.detectMultiScale(gray,1.3,5)
    if faces is():
        return img,[]
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        roi=img[y:y+h,x:x+h]
        roi=cv2.resize(roi,(200,200))
    return img,roi

Id=0
filename="C:\\Users\\ritu\\Desktop\\misc\\Facedetector\\facedetector.avi"
codec=cv2.VideoWriter_fourcc('X','V','I','D')
VideoFileOutput=cv2.VideoWriter(filename,codec,(30),(640,480))
cap=cv2.VideoCapture(0) 
while True:
    ret,frame=cap.read()
    
    image,face=face_detector(frame)
    try:
        face=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
        id,conf=model.predict(face)
        
        profile=getProfile(id)
    
        if profile!=None:
            cv2.putText(image,str(profile[1]),(240,350),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            
            
            
           # display=str(confidence)+"% confidence it is Rahul"
           # cv2.putText(image,display,(100,100),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,0),2)
       
        
            #cv2.putText(image,"UNLOCKED!!",(240,400),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            
            
                   
    except:
        
        cv2.putText(image,"NO face Found",(240,400),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)  
    cv2.imshow("face Recognizer",image)
    VideoFileOutput.write(frame)
    if cv2.waitKey(20)==13:
        break
cap.release()
cv2.destroyAllWindows()


                                            
