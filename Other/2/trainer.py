import os
import numpy as np
from keras.models import Sequential
from keras.layers import MaxPooling2D, Dense, Conv2D, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

training_set_dir = "../data2/Training"
test_set_dir = "../data2/Test"

model = Sequential([
    Conv2D(100, (3,3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2,2),
    Conv2D(100, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dropout(0.5),
    Dense(50, activation='relu'),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

training_data = ImageDataGenerator(rescale=1.0/255., rotation_range=40, width_shift_range=0.2,
                                   height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
training_generator = training_data.flow_from_directory(training_set_dir, batch_size=10, target_size=(150,150))

test_data = ImageDataGenerator(rescale=1.0/255.)
test_generator = test_data.flow_from_directory(test_set_dir, batch_size=10, target_size=(150, 150))

checkpoint = ModelCheckpoint('trained_model.h5', monitor='val_loss', save_best_only=True, mode='auto')

model.fit_generator(training_generator, epochs=30, validation_data=test_generator, callbacks=[checkpoint])

