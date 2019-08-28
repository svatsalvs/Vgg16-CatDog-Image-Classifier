from keras import applications
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense
import cv2
import numpy as np

img_width, img_height = 150, 150

model = applications.VGG16(weights='imagenet', include_top=False,input_shape=(img_width, img_height, 3))

for layer in model.layers[:-3]:
	layer.trainable = False

x = model.output
x = Flatten()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(128, activation="relu")(x)
predictions = Dense(2, activation="softmax")(x)

model_final = Model(input=model.input, output=predictions)

model_final.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])

model_final.load_weights('modelCatDog.h5', by_name = True)

print('model compiled')

path_of_images = "" #Enter path of images to be predicted
list_of_images = os.listdir(path_of_images)

for image in list_of_images:

	img = cv2.imread(os.path.join(path_of_images, image))
	cv2.imshow('image',img)
	cv2.waitKey(0)
	img = cv2.resize(img,(150,150))
	img = img/255
	img = np.reshape(img,[1,150,150,3])
	classes = model_final.predict(img)
	print((classes))
	cv2.destroyAllWindows()