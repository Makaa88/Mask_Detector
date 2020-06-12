import os
from keras.models import Model
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import AveragePooling2D
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras import backend as K
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input
from keras.utils import to_categorical
import numpy as np
import keras.applications.vgg16
import pickle
import cv2

current_dir = os.path.dirname(os.path.abspath(__file__))
training_set_dir = os.path.join(current_dir, "data2/Training")
test_set_dir = os.path.join(current_dir, "data2/Test")

data = []
labels = []

for root, dirs, files in os.walk(training_set_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            image = load_img(path, target_size=(224, 224))
            image = img_to_array(image)
            image = preprocess_input(image)
            data.append(image)

            label = os.path.basename(root).replace(" ", "-").lower()
            labels.append(label)

data = np.array(data, dtype='float32') / 255.0
labels = np.array(labels)
print(labels)


lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels)

imd = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.1, zoom_range=0.15,
                         horizontal_flip=True, fill_mode='nearest')



base_model = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224,224,3)))
main_model = base_model.output
main_model = AveragePooling2D(pool_size=(7,7))(main_model)
main_model = Flatten(name="flatten")(main_model)
main_model = Dense(128, activation='relu')(main_model)
main_model = Dropout(0.5)(main_model)
main_model = Dense(2, activation='softmax')(main_model)

model = Model(inputs=base_model.input, outputs=main_model)

for layer in model.layers:
    layer.trainable = False




EPOCHS = 20
BS = 32

opt = Adam(lr= 0.0001, decay=0.0001/EPOCHS)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

model.fit_generator(imd.flow(trainX, trainY, batch_size=BS), steps_per_epoch=len(trainX)//BS,
                    validation_data=(testX, testY), validation_steps=len(testX)//BS, epochs=EPOCHS)


model.save("keras.mode.model", save_format="h5")
with open("keras_labels.pickle", "wb") as file:
    pickle.dump(lb, file)







