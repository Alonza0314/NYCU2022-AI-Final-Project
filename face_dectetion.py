import cv2
import numpy as np
from time import sleep
import os
from tensorflow import keras
import tensorflow_hub as hub
os.environ['CUDA_VISIBLE_DEVICES']='-1'

model = keras.models.load_model('model_new.h5',custom_objects={'KerasLayer':hub.KerasLayer})

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
        cropp=frame[100:375,235:430]
        
        x=[]
        crop=cv2.resize(cropp,(224,224))
        x.append(crop)
        
        crop=np.array(x)
        crop=crop.astype('float32')
        crop/=255
        
        flag_e=eye_detect(cropp)
        
        if flag_e==0:
            cv2.rectangle(frame,(235,100),(430,375),(255,0,255),4)
        else:
            predi=model.predict(crop)
            pred_classes = np.argmax(predi, axis=1)
            if pred_classes[0]==0:
                cv2.rectangle(frame,(235,100),(430,375),(255,0,0),4)
            elif pred_classes[0]==1:
                cv2.rectangle(frame,(235,100),(430,375),(0,255,0),4)
            elif pred_classes[0]==2:
                cv2.rectangle(frame,(235,100),(430,375),(0,0,255),4)
        
        frame=cv2.flip(frame,1,dst=None)
        cv2.imshow('Face Dectection',frame)
        
        if cv2.waitKey(1)&0xFF==ord('\r'):
            break
        
        sleep(0.05)
    
    cap.release()
    cv2.destroyAllWindows()
    
if __name__=='__main__':
    face_camera()