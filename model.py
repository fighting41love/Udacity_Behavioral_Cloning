import pickle
import csv
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.image as mpimg

data_csv = 'newdata/driving_log.csv'
training_dat = pd.read_csv(data_csv, names=None, header=None)
# -46
X_center = [dirs[-46:] for dirs in training_dat[0]]
X_left = [dirs[-44:] for dirs in training_dat[1]]
X_right = [dirs[-45:] for dirs in training_dat[2]]
X_all = [X_center, X_left, X_right]
Y_center = training_dat[3]

def get_training_image(all_data, y):
    num = np.random.randint(0, len(Y_center))
    # print(all_data[0][num])
    # print(all_data[1][num])
    # print(all_data[2][num])
    if num % 3 == 0:
        image_dir = all_data[0][num]
        angle = y[num]
    elif num % 3 == 1:
        image_dir = all_data[1][num]
        angle = y[num] + 0.07
    else:
        image_dir = all_data[2][num]
        angle = y[num] - 0.07
    image = mpimg.imread(image_dir)
    # re_image = image.reshape(1,160,320,3)
    return image, angle

# generator
def generator(all_data, y_angles, batch_size):
    images = []
    angles = []
    while True:
        for i in range(len(y_angles)):
            # print('prepare new data')
            image, angle = get_training_image(all_data, y_angles)
            image, angle = process_image(image, angle)
            
            images.extend(image)
            angles.append(angle)
            if len(images) >= batch_size:
                # print('yield')
                yield np.array(images), np.array(angles)
                images = []
                angles = []

def process_image(img, steering):
    flip_prob = np.random.random()
    
    if (flip_prob > 0.5):
        # flip the image and reverse the steering angle
        steering = -1.0 * steering
        img = cv2.flip(img.astype(np.float32), 1)
    
    new_img = cv2.GaussianBlur(img, (1,1), 0)
    new_img, steering = brightness_image(new_img, steering)

return new_img.reshape(1,160,320,3), steering


def brightness_image(img, steering):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv[:,:,2] = hsv[:,:,2] * (1 + np.random.uniform(-0.6, 0.2))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), steering

train_generator = generator(X_all, Y_center, 32)

# Build the model

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.utils import plot_model


model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
# model.add(MaxPooling2D())
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
# model.add(MaxPooling2D())
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.summary()
# plot_model(model, to_file='model.png')

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, steps_per_epoch=len(X_center), epochs=1, verbose=1, callbacks=None)

model.save('model.h5')
