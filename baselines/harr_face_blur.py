import datetime
import argparse
import imutils
import time
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

        img[y:y+h, x:x+w] = cv2.blur(img[y:y+h, x:x+w],(int(max(w,h)/2), int(min(w,h)/2)))
        # img[y:y+h, x:x+w] = cv2.GaussianBlur(img[y:y+h, x:x+w], (5,5),0)
        cv2.imshow('frame',img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
