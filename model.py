from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
sz = 310  # Adjusted size based on your preprocessed images

# Step 1 - Building the CNN
classifier = Sequential()

# First convolution layer and pooling
classifier.add(Convolution2D(32, (3, 3), input_shape=(sz, sz, 1), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Second convolution layer and pooling
classifier.add(Convolution2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Flattening the layers
classifier.add(Flatten())

# Adding fully connected layers with dropout
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dropout(0.4))
classifier.add(Dense(units=96, activation='relu'))
classifier.add(Dropout(0.4))
classifier.add(Dense(units=64, activation='relu'))

# Output layer
classifier.add(Dense(units=27, activation='softmax'))  # softmax for more than 2 classes

# Compiling the CNN
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 2 - Preparing the train/test data and training the model
classifier.summary()

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('processed_data/trainingData',
                                                 target_size=(sz, sz),
                                                 batch_size=10,
                                                 color_mode='grayscale',
                                                 class_mode='categorical')
print(len(training_set))
test_set = test_datagen.flow_from_directory('processed_data/testingData',
                                            target_size=(sz, sz),
                                            batch_size=10,
                                            color_mode='grayscale',
                                            class_mode='categorical')
print(len(test_set))
# Assuming you have 27 classes, adjust the steps_per_epoch and validation_steps accordingly
classifier.fit_generator(
    training_set,
    steps_per_epoch=1750,  # Update with the actual count
    epochs=5,
    validation_data=test_set,
    validation_steps=1750  # Update with the actual count
)

# Saving the model
model_json = classifier.to_json()
with open("model-bw.json", "w") as json_file:
    json_file.write(model_json)

print('Model Saved')
classifier.save_weights('model-bw.h5')
print('Weights saved')
