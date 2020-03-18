import os
import cv2
import numpy as np

path = "./recognition_project/"

data=[]
face_labels=[]

mydict={}

cnt = 0
for fil in os.listdir(path):
	if fil.endswith(".npy"):
		
		mydict[cnt] = fil[:-4]
		data_item = np.load(path+fil)
		data.append(data_item)
		
		labels = cnt*( np.ones((data_item.shape[0],)) )
		face_labels.append(labels)
		cnt+=1	
		

data = np.concatenate(data,axis=0)
face_labels = np.concatenate(face_labels,axis=0)

print(type(data))
print(type(face_labels))

print(face_labels.shape)
print(data.shape)		


def dist(a1,a2):
    return (np.sum((a1-a2)**2))**0.5

def knn(X,Y,test_point,k=10):
    
    distances=[]
    for i in range(X.shape[0]):
        d = dist(test_point,X[i])
        distances.append((d,Y[i]))
        
    distances =  sorted(distances)    
    labels = np.asarray(distances[:k])[:,1]    
    uniq,freq = np.unique(labels,return_counts=True)
    return  int(uniq[np.argmax(freq)])
    
    
camera = cv2.VideoCapture(0)
face_detetctor = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

while True:
    flag,img = camera.read()
    if flag ==False:
        continue
    allfaces = face_detetctor.detectMultiScale(img,1.3,5)
    offset = 10
    for (x,y,w,h) in allfaces:
        cropped = img[y-offset:y+h+offset,x-offset:x+w+offset]
        cropped = cv2.resize(cropped,(100,100))
        cropped = cropped.reshape((1,-1))
        #print(cropped.shape)
        prediction = knn(data,face_labels,cropped)
        predicted_name = mydict[prediction]

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img,predicted_name,(x,y-10),font,2,(255,0,0),2,cv2.LINE_AA)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)

    cv2.imshow("window",img)

    keypressed = cv2.waitKey(1) & 0xFF 
    if keypressed == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()    
