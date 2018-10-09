from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint
from keras import backend as K

#code is orginially set up to pull from two different image files.
'''
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    samplewise_center=True,
    preprocessing_function=preprocess_input)

test_datagen = ImageDataGenerator(rescale=1. / 255, preprocessing_function=preprocess_input)
'''
from keras.preprocessing.image import ImageDataGenerator
nb_train_samples = 16675
nb_validation_samples = 8325
nb_classes = 2  # number of classes
epochs =5
BATCH_SIZE = 10

# dimensions of our images.
IMAGE_SIZE= 299

data_generator = ImageDataGenerator(rescale=1./255, validation_split=0.33)

TRAINING_DIR = r'C:\Users\Andrew Portal\Pictures\PetImages'

train_generator = data_generator.flow_from_directory(TRAINING_DIR, target_size=(IMAGE_SIZE, IMAGE_SIZE), shuffle=True, seed=13,
                                                     class_mode='categorical', batch_size=BATCH_SIZE, subset="training")

validation_generator = data_generator.flow_from_directory(TRAINING_DIR, target_size=(IMAGE_SIZE, IMAGE_SIZE), shuffle=True, seed=13,
                                                     class_mode='categorical', batch_size=BATCH_SIZE, subset="validation")

'''
rtrain_data_dir = r'C:\\Users\Andrew Portal\Pictures\PetImages'
validation_data_dir = r'C:\\Users\Andrew Portal\Pictures\PetImages'
'''

'''
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,

    target_size=(img_width, img_height),
    batch_size=batch_size,
class_mode='categorical')
'''
# checkpoint
filepath="weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)
print('Finished Loading')

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)

# let's add a fully-connected layer 1024
x = Dense(256, activation='relu')(x)
# and a logistic layer
predictions = Dense(2, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

hist = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // BATCH_SIZE,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // BATCH_SIZE,
    callbacks=callbacks_list)


model.save('inception.model')
print("model saved")