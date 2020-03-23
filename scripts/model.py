import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json
import os
import boto3

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense

from keras.preprocessing.image import ImageDataGenerator

config = {'img_pixels': 256,
          'n_filters': 64,
          'layer_nodes': 512,
          'batchsize': 32,
          'epochs': 50,
          'kernel_size': (4,4),
          'pool_size': (2,2),
          'dropout':0.2,
          'steps_per_epoch': 50,
          'validation_steps': 5,
          'version': 1
          }

# # Load env variables
# img_pixels = os.environ['img_pixels']
# n_filters = os.environ["n_filters"]
# layer_nodes = os.environ["layer_nodes"]
# batchsize = os.environ["batchsize"]
# epochs = os.environ["epochs"]
# kernel_size = os.environ['kernel_size']
# pool_size = os.environ['pool_size']
# dropout = os.environ['dropout']
# steps_per_epoch = os.environ['steps_per_epoch']
# validation_steps = os.environ['validation_steps']
# version = os.environ['version']

# config
img_pixels = config['img_pixels']
n_filters = config['n_filters']
layer_nodes = config['layer_nodes']
batchsize = config['batchsize']
epochs = config['epochs']
kernel_size = config['kernel_size']
pool_size = config['pool_size']
dropout = config['dropout']
steps_per_epoch = config['steps_per_epoch']
validation_steps = config['validation_steps']
model_version = config['version']

# AWS config
bucket = 'himi-car-classification'
s3_client = boto3.client('s3')
s3_resource = boto3.resource('s3')

# Count number of folders for classes
folders = 0
for _, dirnames, filenames in os.walk("../scraped_images_2020/train"):
  # ^ this idiom means "we won't be using this value"
    folders += len(dirnames)

# layers
car_classifier = Sequential()
#Adding 1st Convolution and Pooling Layer
car_classifier.add(Conv2D(n_filters,kernel_size=(3,3),input_shape=(img_pixels,img_pixels,3),activation='relu'))
car_classifier.add(MaxPool2D(pool_size=pool_size))
car_classifier.add(Dropout(dropout))
#Adding 2nd Convolution and Pooling Layer
car_classifier.add(Conv2D(n_filters,kernel_size=(3,3),activation='relu'))
car_classifier.add(MaxPool2D(pool_size=pool_size))
car_classifier.add(Dropout(dropout))
#Adding 3rd Convolution and Pooling Layer
car_classifier.add(Conv2D(n_filters,kernel_size=(3,3),activation='relu'))
car_classifier.add(MaxPool2D(pool_size=pool_size))
car_classifier.add(Dropout(dropout))
#Adding 4th Convolution and Pooling Layer
car_classifier.add(Conv2D(n_filters,kernel_size=(3,3),activation='relu'))
car_classifier.add(MaxPool2D(pool_size=pool_size))
car_classifier.add(Dropout(dropout))
#Adding 5th Convolution and Pooling Layer
car_classifier.add(Conv2D(n_filters,kernel_size=(3,3),activation='relu'))
car_classifier.add(MaxPool2D(pool_size=pool_size))
car_classifier.add(Dropout(dropout))

#Flatten
car_classifier.add(Flatten())

#Adding Input and Output Layer
car_classifier.add(Dense(units=layer_nodes,activation='relu'))
car_classifier.add(Dense(units=layer_nodes,activation='relu'))
car_classifier.add(Dense(units=layer_nodes,activation='relu'))
car_classifier.add(Dense(units=folders,activation='softmax'))

car_classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Data agumentation
train_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory('../scraped_images_2020/train',
                                               target_size=(img_pixels,img_pixels),
                                               batch_size=batchsize,
                                               class_mode='categorical',
                                               shuffle=True,
                                               seed=42)

test_data = test_datagen.flow_from_directory('../scraped_images_2020/test',
                                             target_size=(img_pixels,img_pixels),
                                             batch_size=1,
                                             class_mode='categorical',
                                             shuffle=True,
                                             seed=42)


history = car_classifier.fit_generator(train_data,
                                       steps_per_epoch=steps_per_epoch,
                                       epochs=epochs,
                                       validation_data=test_data,
                                       validation_steps=validation_steps
                                      )


metrics = pd.DataFrame.from_dict(history.history)
metrics = pd.concat([pd.Series(range(0,30),name='epochs'),metrics],axis=1)
metrics = metrics.reset_index().drop('epochs', axis=1).rename(columns={'index': 'epochs'})
metrics['config'] = str(config)

# Save metrics
metrics.to_csv("../models/cars_classifier_metrics2.csv", index=False)

# Save model and weights in s3

# serialize model to JSON
model_json = car_classifier.to_json()
s3_client.put_object(Body=model_json,
                     Bucket=bucket,
                     Key='models/tuned_cnn_model.json')
# with open("../models/cars_classifier_tuned_100eP_50ba_1ba(val).json", "w") as json_file:
#     json_file.write(model_json)

# serialize weights to HDF5
car_classifier.save_weights("tuned_cnn_weights.h5")

s3_resource.Bucket(bucket).upload_file("tuned_cnn_weights.h5", "models/tuned_cnn_weights.h5")

print("Saved model and weights to s3")