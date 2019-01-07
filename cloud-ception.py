from keras.applications.inception_v3 import InceptionV3
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator

from keras.models import Model
from keras.layers import Conv2D, Dense, GlobalAveragePooling2D, Dropout, Flatten,
from keras.callbacks import ModelCheckpoint

data_generator = ImageDataGenerator(rescale=1./255, validation_split=0.10)

TRAINING_DIR = 'train'


nb_train_samples = 28846
nb_validation_samples = 3203
nb_classes = 3 # number of classes
epochs = 5
BATCH_SIZE = 64

# dimensions of our images.
IMAGE_SIZE= 299




#consider color_mode=""  greyscale or rgb
train_generator = data_generator.flow_from_directory(
    TRAINING_DIR,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset="training")

validation_generator = data_generator.flow_from_directory(
    TRAINING_DIR,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset = "validation")

# checkpoint
filepath="weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]



# create the base pre-trained model
base_model = InceptionV3(weights = 'imagenet', input_shape=(299,299,3), include_top=False)
print('Finished Loading')

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False


x = base_model.output

x = Conv2D(256, kernel_size = (3,3), padding = 'valid')(x)

x = Flatten()(x)

# drop out layer helps with overfitting

x = Dropout(0.5)(x)

# and a logistic layer
predictions = Dense(3, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)
#model.load_weights("weights-improvement-02-1.69.hdf5")

#adjust parameters of optimizer
#adam = optimizers.adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
# compile the , loss should equal sparse when train and test are in seperate folders

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])




# can a steps_per_epoch (the number of batch iterations before a training epoch is considered finished)
# to avoid going through whole dataset

hist = model.fit_generator(
    train_generator,
    epochs=epochs,
    steps_per_epoch=nb_train_samples // BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // BATCH_SIZE,
    callbacks=callbacks_list)


model.save('cloud-ception.model')
print("model saved")