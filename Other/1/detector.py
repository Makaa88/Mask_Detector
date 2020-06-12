import cv2
import numpy as np
from keras.models import load_model
from PIL import Image



face_cascade_name = "../Cascades/haarcascade_frontalface_default.xml"
mouth_cascade_name = "../Cascades/haarcascade_smile.xml"
face_cascade = cv2.CascadeClassifier(face_cascade_name)
mouth_cascade = cv2.CascadeClassifier(mouth_cascade_name)

model = load_model("facial_features_model.h5")
capture = cv2.VideoCapture(0)

while (True):
    rat, frame = capture.read()
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame,  (x, y), (x+w, y+h), (255,0,0))
        face = gray_image[y:y+h, x:x+w]
        face = cv2.resize(face, (96, 96))
        mouth = mouth_cascade.detectMultiScale(face)
        face_image = Image.fromarray(face, 'L')
        face_image_array = np.array(face_image)
        face_image_array = np.expand_dims(face_image_array, axis=0)
        face_image_array = np.expand_dims(face_image_array, axis=3)
        pr = model.predict(face_image_array)

        #cv2.rectangle(frame, (x + pr[0][12][0], y +pr[0][12][1] ), (x + pr[0][11][0] + w, y + pr[0][11][1] + h), (255, 0, 0))


        # if prediction_result[0][0] > prediction_result[0][1]:
        #     cv2.putText(frame, "Mask " + str("{:.2f}".format(prediction_result[0][0]*100)), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1)
        # else:
        #     cv2.putText(frame, "No Mask " + str("{:.2f}".format(prediction_result[0][0]*100)), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
        #                 1)


    cv2.imshow('Clip', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()