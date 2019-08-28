from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense

img_width, img_height = 150, 150

train_data_dir = '' #Enter tarining data path
validation_data_dir = '' #Enter validation data path
nb_train_samples = 2000
nb_validation_samples = 800
epochs = 25
batch_size = 16

model = applications.VGG16(weights='imagenet', include_top=False,input_shape=(img_width, img_height, 3))
print('Model loaded.')
#model.summary()

for layer in model.layers[:-3]:
    layer.trainable = False

x = model.output
x = Flatten()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(128, activation="relu")(x)
predictions = Dense(2, activation="softmax")(x)

model_final = Model(input=model.input, output=predictions)
#model_final.summary()

model_final.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    horizontal_flip=True,
    fill_mode="nearest",
    zoom_range=0.3,
    width_shift_range=0.3,
    height_shift_range=0.3,
    rotation_range=0)

test_datagen = ImageDataGenerator(
	rescale=1. / 255,
    horizontal_flip=True,
	fill_mode="nearest",
	zoom_range=0.3,
	width_shift_range=0.3,
	height_shift_range=0.3,
	rotation_range=0)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical")

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical")

model_final.fit_generator(
    train_generator,
    #samples_per_epoch=nb_train_samples,
    epochs=epochs,
    steps_per_epoch = int(nb_train_samples/batch_size),
    validation_data=validation_generator,
    validation_steps = int(nb_validation_samples/batch_size))
    #nb_val_samples=nb_validation_samples)
#model.save_weights('first_try.h5')
checkpoint_val_acc = ModelCheckpoint('vgg16_val_acc.{acc:.4f}-{epoch:03d}.h5', monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
checkpoint_ac = ModelCheckpoint('vgg16_acc.{acc:.4f}-{epoch:03d}.h5', monitor='acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
#lr_reduce = ReduceLROnPlateau(monitor='loss', patience=10, factor=0.1, min_lr=0.00001)
#early = EarlyStopping(monitor='val_acc', min_delta=0, patience=50, verbose=1, mode='auto')
photo = TensorBoard(log_dir='logs')