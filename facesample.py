# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 21:45:21 2019

@author: ritu
"""
import sqlite3
import cv2
face_classifier=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
def InsertorDetect(Id,Name):
    conn=sqlite3.connect("C:/Users/ritu/Desktop/misc/FaceBase.db")
    cmd="SELECT * FROM People WHERE ID="+str(Id)
    cursor=conn.execute(cmd)
    c=0
    for row in cursor:
       c=1
    if(c==1):
        cmd="UPDATE People SET Name="+str(Name)+"WHERE ID="+str(Id)
    else:
        cmd="INSERT INTO People(Id,Name) Values("+str(Id)+","+str(Name)+")"
    conn.execute(cmd)
    conn.commit()
    conn.close()
def face_extractor(image):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces=face_classifier.detectMultiScale(gray,1.3,5)
    if faces is ():
        return None
    
    for (x,y,w,h) in faces:
        cropped=image[y:y+h,x:x+w]
    return cropped
    
cap=cv2.VideoCapture(0)
c=0
id=input("Enter id of user")
name=input("Enter Name")
InsertorDetect(id,name)
while True:
    ret,frame=cap.read()

    if face_extractor(frame) is not None:
        c=c+1
        face=cv2.resize(face_extractor(frame),(200,200))
        face=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
        file="C:/Users/ritu/Desktop/misc/output/user."+id+"."+str(c)+".jpg"
        cv2.imwrite(file,face)
        cv2.putText(face,str(c),(50,50),cv2.FONT_ITALIC,1,(0,255,0),2)
        cv2.imshow("faceSample",face)
        if cv2.waitKey(5)==13 or c==100:
            break
    else:
        print("Face Not Found")
cap.release()
cv2.destroyAllWindows()
