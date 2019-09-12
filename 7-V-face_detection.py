# -*- coding: utf-8 -*-
"""
Created on May  6 17:00:05 2016

@author: omerfarukkoc
"""
import cv2
import sys

yuzAlgilamaCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
gozAlgilamaCascade = cv2.CascadeClassifier("haarcascade_eye.xml")
video = cv2.VideoCapture(0)

while True:
    ret,frame = video.read()
    gri = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    yuzler = yuzAlgilamaCascade.detectMultiScale(gri, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    for (x, y, w, h) in yuzler:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,100,100), 2)
        ilgilibolge_gri = gri[y:y+h, x:x+w]
        ilgilibolge_renkli = frame[y:y+h, x:x+w]
        gozler = gozAlgilamaCascade.detectMultiScale(ilgilibolge_gri)

        for (ex, ey, ew, eh) in gozler:
            cv2.rectangle(ilgilibolge_renkli, (ex,ey), (ex+ew, ey+eh), (255,0,0), 2)

    cv2.imshow("YuzAlgilama", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
