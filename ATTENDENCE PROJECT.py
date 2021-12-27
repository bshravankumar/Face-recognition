import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime


path = 'Project-image'
images = []
classNames = []
myList = os.listdir(path)
print(myList)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

print(classNames)


def findencodings(__images__):
    encodelist = []
    for __img__ in images:
        __img__ = cv2.cvtColor(__img__, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(__img__)[0]
        encodelist.append(encode)
    return encodelist


def markattendance(__name__):
    with open('Attendance.csv', 'r+') as f:
        mydatalist = f.readlines()

        namelist = []
        for line in mydatalist:
            entry = line.split(',')
            namelist.append(entry[0])
            print(entry)

        if __name__ not in namelist:
            print("details not available")
            now = datetime.now()
            dtstring = now.strftime('%d/%m/%Y')
            f.writelines(f' {name}, {dtstring}\n')


encodelistknown = findencodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodelistknown, encodeFace)
        faceDis = face_recognition.face_distance(encodelistknown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markattendance(__name__)
    cv2.imshow('Webcam', img)
    cv2.waitKey(1)
