# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 11:40:36 2021

@author: manin
"""


import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
import datetime


mymodel=load_model(r'/home/anmol/Desktop/FaceMaskDetector-master/mymodel.h5')



face_cascade=cv2.CascadeClassifier(r'/home/anmol/Desktop/FaceMaskDetector-master/haarcascade_frontalface_default.xml')




cap=cv2.VideoCapture(0)
while cap.isOpened():
    _,frame=cap.read()
    face=face_cascade.detectMultiScale(frame,scaleFactor=1.1,minNeighbors=4)
    for x,y,w,h in face:
        face_img=frame[y:y+h,x:x+w]
        cv2.imwrite('face.jpg',face_img)
        img=image.load_img('face.jpg',target_size=(150,150,3))
        img=image.img_to_array(img)
        img=np.expand_dims(img,axis=0)
        ans=mymodel.predict(img)  
        if ans[0][0]==1:
            cv2.putText(frame,'WithoutMAsk',(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
        else:
            cv2.putText(frame,'Masked',(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)

    
    cv2.imshow('camera',frame)
    if cv2.waitKey(1)==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


    
    
    























# cap=cv2.VideoCapture(0)
# face_cascade=cv2.CascadeClassifier(r'/home/anmol/Desktop/FaceMaskDetector-master/haarcascade_frontalface_default.xml')

# while cap.isOpened():
#     _,img=cap.read()
#     face=face_cascade.detectMultiScale(img,scaleFactor=1.1,minNeighbors=4)
#     for(x,y,w,h) in face:
#         face_img = img[y:y+h, x:x+w]
#         cv2.imwrite('temp.jpg',face_img)
#         test_image=image.load_img('temp.jpg',target_size=(150,150,3))
#         test_image=image.img_to_array(test_image)
   
#         test_image=np.expand_dims(test_image,axis=0)
#         pred=mymodel.predict(test_image)[0][0]
#         if pred==1:
#             cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
#             cv2.putText(img,'NO MASK',((x+w)//2,y+h+20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
#         else:
#             cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
#             cv2.putText(img,'MASK',((x+w)//2,y+h+20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
       
#     cv2.imshow('img',img)
    
#     if cv2.waitKey(1)==ord('q'):
#         break
    
# cap.release()
# cv2.destroyAllWindows()
