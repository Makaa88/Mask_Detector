from keras.layers import Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
training_set_dir = os.path.join(current_dir, "data2/Training")
test_set_dir = os.path.join(current_dir, "data2/Test")

training_data = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_data = ImageDataGenerator(rescale=1./255)

training_set = training_data.flow_from_directory(training_set_dir, target_size=(224,224), batch_size=32, class_mode='categorical')
test_set = test_data.flow_from_directory(test_set_dir, target_size=(224,224), batch_size=32, class_mode='categorical')

vgg = VGG16(input_shape=[224, 224, 3], weights='imagenet', include_top=False)

for layer in vgg.layers:
    layer.trainable = False

add_layer = Flatten()(vgg.output)
prediction = Dense(2, activation='softmax')(add_layer)

model = Model(inputs=vgg.input, outputs=prediction)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit_generator(training_set, validation_data=test_set, epochs=5, steps_per_epoch=len(training_set), validation_steps=len(test_set))
model.save("detector_model.h5")