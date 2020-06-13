from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from PIL import Image

proto_path = "face_detector/deploy.prototxt"
wPath = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"

net = cv2.dnn.readNet(proto_path, wPath)

model = load_model("detector_model.h5")

IMAGE_PATH = "4.jpg"
image = cv2.imread(IMAGE_PATH)
origin = image.copy()
(h,w) = image.shape[:2]

blob = cv2.dnn.blobFromImage(image, 1.0, (300,300), (104.,177., 123.))
net.setInput(blob)
detections = net.forward()

for i in range(0, detections.shape[2]):
    conf = detections[0, 0, i, 2]

    if conf > 0.5:
        box = detections[0, 0, i, 3:7] * np.array([w,h,w,h])
        (sX, sY, eX, eY) = box.astype("int")

        (sX, sY) = (max(0, sX), max(0, sY))
        (eX, eY) = (min(w - 1, eX), min(h - 1, eY))

        face = image[sX:eX, sY:eY]
        #face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(face, (224,224))
        face_image = Image.fromarray(face, 'RGB')
        face = img_to_array(face)
        #face = preprocess_input(face)
        face = np.expand_dims(face, axis=0)

        (mask, no_mask) = model.predict(face)[0]
        cv2.rectangle(image, (sX, sY), (eX, eY), (255, 0, 0))
        if mask > no_mask:
            cv2.putText(image, "Mask", (sX, sY), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
        else:
            cv2.putText(image, "No Mask", (sX, sY), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                        1)

cv2.imshow("result", image)
cv2.waitKey(0)