import os
import numpy as np
from keras.models import Sequential
from keras.layers import BatchNormalization, Convolution2D, Activation, MaxPooling2D, Dense, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint
import pandas

current_dir = os.path.dirname(os.path.abspath(__file__))
training_set_dir = os.path.join(current_dir, "training.csv")
test_set_dir = os.path.join(current_dir, "test.csv")

test_data = pandas.read_csv(test_set_dir)
training_data = pandas.read_csv(training_set_dir)
training_data.fillna(method="ffill", inplace=True)

data = []
labels = []

for i in range(0, 7049):
    image = training_data['Image'][i].split(' ')
    image = ['0' if x == '' else x for x in image]
    data.append(image)

data = np.array(data, dtype='float')
trainX = data.reshape(-1, 96, 96, 1)

training_data = training_data.drop('Image', axis=1)

for  i in range(0, 7049):
    label = training_data.iloc[i,:]
    labels.append(label)

trainY = np.array(labels, dtype='float')

model = Sequential()

model.add(BatchNormalization(input_shape=(96, 96, 1)))
model.add(Convolution2D(24, (5,5), padding="same", init='he_normal', input_shape=(96, 96, 1), dim_ordering="tf"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"))

model.add(Convolution2D(36, (5,5)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"))

model.add(Convolution2D(48, (5,5)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"))

model.add(Convolution2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="valid"))

model.add(Convolution2D(64, (3,3)))
model.add(Activation("relu"))

model.add(GlobalAveragePooling2D())

model.add(Dense(500, activation="relu"))
model.add(Dense(90, activation="relu"))
model.add(Dense(30))


model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath='facial_features_model.h5', verbose=1, save_best_only=True)

epochs = 30
hist = model.fit(trainX, trainY, validation_split=0.2, shuffle=True, epochs=epochs, batch_size=20, callbacks=[checkpointer], verbose=1)