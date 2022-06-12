import cv2 as cv

cap=cv.VideoCapture(0)

while(cap.isOpened()):
    ret,frame=cap.read()
    
    gray_frame=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    
    face_cascade=cv.CascadeClassifier(cv.data.haarcascades+"haarcascade_frontalface_default.xml")
    faces=face_cascade.detectMultiScale(gray_frame)
    
    for x,y,w,h in faces:
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    
    cv.imshow('Face Detection',frame)
    
    if cv.waitKey(1)&0xFF==ord('\r'):
        break

cap.release()
cv.destroyAllWindows()    