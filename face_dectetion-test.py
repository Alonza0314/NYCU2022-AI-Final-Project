import cv2
from time import sleep

face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
eye_cascade=cv2.CascadeClassifier("haarcascade_eye.xml")

def face_detect(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(img)
    if len(faces)>0:
        return 1
    return 0

def eye_detect(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    eyes=eye_cascade.detectMultiScale(img)
        
    if len(eyes)>0:
        return 1
    return 0

def face_camera():
    cap=cv2.VideoCapture(0)
    
    while(cap.isOpened()):
        ret,frame=cap.read()
        crop=frame[150:300,150:300]
        cv2.rectangle(frame,(235,100),(430,375),(255,0,0),4)
        
        flag_f=face_detect(crop)
        flag_e=eye_detect(crop)
        img=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
        
        eyes=eye_cascade.detectMultiScale(img)
        for x,y,w,h in eyes:
            cv2.rectangle(frame,(235+x,100+y),(235+x+w,100+y+h),(255,0,0),2)
        
        if flag_f==1:
            cv2.rectangle(frame,(235,100),(430,375),(0,0,255),4)
        elif flag_e==0:
            cv2.rectangle(frame,(235,100),(430,375),(255,0,0),4)
        elif flag_f==0 and flag_e==1:
            cv2.rectangle(frame,(235,100),(430,375),(0,255,0),4)
        
        frame=cv2.flip(frame,1,dst=None)
        cv2.imshow('Face Dectection',frame)
        
        if cv2.waitKey(1)&0xFF==ord('\r'):
            break
        
        sleep(0.1)
    
    cap.release()
    cv2.destroyAllWindows()
    
if __name__=='__main__':
    face_camera()