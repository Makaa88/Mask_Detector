import cv2
import numpy as np
from keras.models import load_model
from PIL import Image



face_cascade_name = "Cascades/haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(face_cascade_name)

model = load_model("detector_model.h5")
capture = cv2.VideoCapture('mask.mp4')
#capture = cv2.VideoCapture(0)

while (True):
    rat, frame = capture.read()
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=4)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame,  (x, y), (x+w, y+h), (255,0,0))
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (224,224))
        face_image = Image.fromarray(face, 'RGB')
        face_image_array = np.array(face_image)
        face_image_array = np.expand_dims(face_image_array, axis=0)
        prediction_result = model.predict(face_image_array)

        if prediction_result[0][0] > 0.55:
            cv2.putText(frame, "Mask " + str("{:.2f}".format(prediction_result[0][0]*100)), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1)
        else:
            cv2.putText(frame, "No Mask " + str("{:.2f}".format(prediction_result[0][0]*100)), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                        1)


    cv2.imshow('Clip', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()