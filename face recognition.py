import cv2
import face_recognition
# STEP 1 giving data
imgjai_paul = face_recognition.load_image_file('Project-image/jai_paul.jpg')
imgjai_paul = cv2.cvtColor(imgjai_paul, cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('Project-image/jai_paul test.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)
# step 2 giving guid lines for the given data
faceLoc = face_recognition.face_locations(imgjai_paul)[0]
encodejai_paul = face_recognition.face_encodings(imgjai_paul)[0]
cv2.rectangle(imgjai_paul, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

faceLoctest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (faceLoctest[3], faceLoctest[0]), (faceLoctest[1], faceLoctest[2]), (255, 0, 255), 2)
# step 3 comparing the data
result = face_recognition.compare_faces([encodejai_paul], encodeTest)
faceDis = face_recognition.face_distance([encodejai_paul], encodeTest)
print(result, faceDis)
cv2.putText(imgTest, f'{result}{round(faceDis[0],2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

cv2.imshow('jai_paul', imgjai_paul)
cv2.imshow('jai_paul test', imgTest)
cv2.waitKey(0)
