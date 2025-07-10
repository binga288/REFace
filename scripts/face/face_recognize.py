import cv2
import face_recognition

target_image = "./examples/the-boys.jpg"
homelander_image = "./examples/maeve.jpg"

homelander_image = face_recognition.load_image_file(homelander_image)
homelander_face_encoding = face_recognition.face_encodings(homelander_image)[0]

loaded_image = face_recognition.load_image_file(target_image)
face_locations = face_recognition.face_locations(loaded_image, model="cnn")
face_encodings = face_recognition.face_encodings(loaded_image, face_locations)

cv2_image = cv2.imread(target_image)
for face_encoding, face_location in zip(face_encodings, face_locations):
    match = face_recognition.compare_faces([homelander_face_encoding], face_encoding)
    distance = face_recognition.face_distance([homelander_face_encoding], face_encoding)[0]

    if match[0]:
        top, right, bottom, left = face_location
        cv2.rectangle(cv2_image, (left, top), (right, bottom), (0, 0, 255), 2)

cv2.imshow("Face Detection", cv2_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

