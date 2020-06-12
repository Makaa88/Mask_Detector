from PIL import Image
import os
import numpy as np
import cv2
import pickle

current_dir = os.path.dirname(os.path.abspath(__file__))
training_set_dir = os.path.join(current_dir, "data")
face_cascade_name = "Cascades/haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(face_cascade_name)

recognizer = cv2.face.LBPHFaceRecognizer_create()

id = 0
labels_ids = {}
labels = []
data = []

for root, dirs, files in os.walk(training_set_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ", "-").lower()
            if label not in labels_ids:
                labels_ids[label] = id
                id += 1

            current_id = labels_ids[label]
            #pil = Image.open(path).convert("L")
            image = cv2.imread(path)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray_image = cv2.resize(gray_image, (224, 224))
            image_array = np.array(gray_image, "float")
            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.3, minNeighbors=5)

            for(x, y, w , h) in faces:
                region_of_interest = image_array[y:y+h, x:x+w]
                data.append(region_of_interest)
                labels.append(current_id)

with open("labels.pickle", "wb") as file:
    pickle.dump(labels_ids, file)

recognizer.train(data, np.array(labels))
recognizer.save("trained.yml")