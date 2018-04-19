import csv
import cv2
import random
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers import Conv2D, MaxPooling2D, Cropping2D
from keras import optimizers

# Read the file that contains the path of the images and angles of the data set
lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    #next(reader, None) # skip the field name
    for line in reader:
        lines.append(line)

# Split the data set into training and validation data sets.
lines_train, lines_valid = train_test_split(lines, test_size=0.2)

def getImage(path):
    '''
    Load an image and convert it from BGR to RGB. 
    This is necessary because OpenCV reads the images in BGR and the simulator uses RGB
    '''
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

def generator(samples, batch_size=32):
    '''
    Generator of batch samples to feed the model with data.
    It is used to not load all the images in memory.
    '''
    num_samples = len(samples)
    current_path = './data/IMG/' 
    while 1: # Loop forever so the generator never terminates
        random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            measurements = []
            for batch_sample in batch_samples:

                measurement_center = float(batch_sample[3])
                # Correction applied to the central camera provided angle.
                correction = 0.255
                measurement_left = measurement_center + correction 
                measurement_right = measurement_center - correction
                
                # Use the images of the three cameras. 
                image_center = getImage(current_path + batch_sample[0].split('/')[-1])
                image_left   = getImage(current_path + batch_sample[1].split('/')[-1])
                image_right  = getImage(current_path + batch_sample[2].split('/')[-1])

                images.append(image_center)
                images.append(image_left)
                images.append(image_right)
                measurements.append(measurement_center)
                measurements.append(measurement_left)
                measurements.append(measurement_right)
                
                # Augmentation of data by flipping all the provided images and inverting the angles.
                images.append(cv2.flip(image_center, 1))
                images.append(cv2.flip(image_left, 1))
                images.append(cv2.flip(image_right, 1))
                measurements.append(measurement_center*-1.0)
                measurements.append(measurement_left*-1.0)
                measurements.append(measurement_right*-1.0)
                
            X_train = np.array(images)
            y_train = np.array(measurements)
            yield shuffle(X_train, y_train)
           
BATCH_SIZE = 32
# create the generator functions
train_generator = generator(lines_train, batch_size=BATCH_SIZE)
validation_generator = generator(lines_valid, batch_size=BATCH_SIZE)

# Definition of the model arquitechture
model = Sequential()
# Normalization of data
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
# Cropping images 70 pixels at the top an 25 pixels at the bottom
model.add(Cropping2D(((70,25),(0,0))))
model.add(Conv2D(24,5,strides=(2,2),activation='relu'))
model.add(Dropout(0.5))
model.add(Conv2D(36,5,strides=(2,2),activation='relu'))
#model.add(Dropout(0.5))
model.add(Conv2D(48,5,strides=(2,2),activation='relu'))
#model.add(Dropout(0.5))
model.add(Conv2D(64,3,activation='relu'))
#model.add(Dropout(0.5))
model.add(Conv2D(64,3,activation='relu'))
#model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100,activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(50,activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(10,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1))

# Definition of the Adam optimizer with a specific learning rate.
adam = optimizers.Adam(lr=0.0001)
# Use of Mean Square Error as loss function. 
model.compile(loss='mse', optimizer=adam)

model.fit_generator(train_generator,
                    validation_data = validation_generator,
                    steps_per_epoch=len(lines_train) / BATCH_SIZE,
                    validation_steps = len(lines_valid) / BATCH_SIZE,
                    epochs=5)

# Save the model.
model.save('model.h5')
