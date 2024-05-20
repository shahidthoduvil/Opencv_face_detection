# Import necessary libraries
import cv2 as cv

# Load the pre-trained Haar cascade for face detection

face_cascade =cv.CascadeClassifier("haarcascade_frontalface_default.xml")
video_capture=cv.VideoCapture(0)


while True:
    ret , frame=video_capture.read()
    if not ret:
        break

    gray=cv.cvtColor(frame , cv.COLOR_BGR2GRAY)

    faces=face_cascade.detectMultiScale(gray , 1.3 ,6)

    for (x,y,w,h) in faces:
        cv.rectangle(frame ,(x,y),(x+w,y+h) ,(0,255,0),4)

    cv.imshow("video",frame)
    if cv.waitKey(1) & 0xFF  == ord('q'):
        break

video_capture.release()

 # Destroy all OpenCV windows

cv.destroyAllWindows()
