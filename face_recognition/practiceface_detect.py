import cv2
import numpy as np

camera = cv2.VideoCapture(0)
face_detetctor = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

skip=0
cropped_faces=[]

name = input("Enter your name :- ")

while True:
    flag,img = camera.read()
    if flag==False:
        continue
        
    faces = face_detetctor.detectMultiScale(img,1.3,5)
   
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
        offset=10
        cropped_img = img[y-offset:y+h+offset,x-offset:x+w+offset]
        cropped_img =  cv2.resize(cropped_img,(100,100))
    
        if skip%10 == 0:
            cropped_faces.append(cropped_img)
            print(len(cropped_faces))

    
    cv2.imshow("window",img)

        
        
    keypressed = cv2.waitKey(1) & 0xFF
    if keypressed == ord('q'):
        break
    
    skip+=1
    
no = len(cropped_faces)    
cropped_faces = np.asarray(cropped_faces)
cropped_faces = cropped_faces.reshape(no,3*10000)
np.save("./recognition_project/" + name + ".npy",cropped_faces)            
            
camera.release()
cv2.destroyAllWindows()
