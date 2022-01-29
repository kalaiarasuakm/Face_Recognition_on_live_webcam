import cv2
import face_recognition
import numpy as np
import os

# importing the images fom the folder
path = 'image'
# creating a list for the image
image = []
#Assigning the names of images in the classNames variable as a list
classNames = []
myList = os.listdir(path)
print(myList)
#importing the images one by one
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    image.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)
#Encoding the images from the folder
def findEncodings(image):
    encodeList = []
    for img in image:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

encodeListKnown = findEncodings(image)
print("Encoding Complete")

#Reading the webcam
cap = cv2.VideoCapture(0)

while True:
    #checking the video has frame and read the frame
    success, img = cap.read()
    #reducing the size of image
    imgS = cv2.resize(img, (0,0),None, 0.25, 0.25)
    #converting bgr to rgb
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    #Locating the face in the webcam
    faceWeb = face_recognition.face_locations(imgS)
    #Ecnoding the image of the faceWeb
    encodeFaceWeb = face_recognition.face_encodings(imgS, faceWeb)

    for encodeFace, faceLoc in zip(encodeFaceWeb, faceWeb):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1,x2,y2,x1=faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img,(x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1,y2-35),(x2,y2),(0, 255, 0), cv2.FILLED)
            cv2.putText(img, name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)

    #Displaying the live webcam
    cv2.imshow("pict", img)
    if cv2.waitKey(30) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
