# Import necessary libraries

import cv2 as cv
import numpy as np

 # Load the pre-trained Haar cascade for face detection
face_cascade=cv.CascadeClassifier("haarcascade_frontalface_default.xml")

# Read an image from the file system
img=cv.imread('shahid.png')

# Convert the image to grayscale, which is required for the face detection algorithm
gray=cv.cvtColor(img,cv.COLOR_BGR2BGRA)

# method to detect faces in the grayscale image.
faces=face_cascade.detectMultiScale(gray , 1.3 ,5)

print(faces)

 # Draw rectangles around the detected faces in the original image
for x,y,w,h in faces:
    cv.rectangle(img , (x,y),(x+w,y+h) ,(0,255,0),2)

# Display the image with detected faces
cv.imshow('orginal' , img)

cv.waitKey(0)

# Destroy all OpenCV windows
cv.destroyAllWindows