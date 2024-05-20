import cv2 as cv
import numpy as np

face_cascade=cv.CascadeClassifier("haarcascade_frontalface_default.xml")
img=cv.imread('shahid.png')
gray=cv.cvtColor(img,cv.COLOR_BGR2BGRA)
faces=face_cascade.detectMultiScale(gray , 1.3 ,6)
print(faces)
for x,y,w,h in faces:
    cv.rectangle(img , (x,y),(x+w,y+h) ,(0,255,0),4)

cv.imshow('orginal' , img)
cv.waitKey(0)
cv.destroyAllWindows