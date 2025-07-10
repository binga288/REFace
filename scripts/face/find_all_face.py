import cv2
import face_recognition

target_image = "./examples/the-boys.jpg"

loaded_image = face_recognition.load_image_file(target_image)
face_locations = face_recognition.face_locations(loaded_image, model="cnn")
print(face_locations)

cv2_image = cv2.imread(target_image)
for location in face_locations:
    top, right, bottom, left = location
    cv2.rectangle(cv2_image, (left, top), (right, bottom), (0, 0, 255), 2)

cv2.imshow("Face Detection", cv2_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

