import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense

from keras.preprocessing.image import ImageDataGenerator

# config
config = {'img_pixels': 200,
          'n_filters': 32,
          'layer_nodes': 400,
          'batchsize': 50,
          'epochs':50,
          'pool_size': (3,3),
          'dropout':0.2,
          'steps_per_epoch': 100,
          'validation_steps': 50
         }

# config
img_pixels = config['img_pixels']
n_filters = config['n_filters']
layer_nodes = config['layer_nodes']
batchsize = config['batchsize']
epochs = config['epochs']
pool_size = config['pool_size']
dropout = config['dropout']
steps_per_epoch = config['steps_per_epoch']
validation_steps = config['validation_steps']

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
car_classifier.add(Dense(units=196,activation='softmax'))

car_classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Data agumentation
train_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory('../car_data/train',
                                               target_size=(img_pixels,img_pixels),
                                               batch_size=batchsize,
                                               class_mode='categorical',
                                               shuffle=True,
                                               seed=42)

test_data = test_datagen.flow_from_directory('../car_data/test',
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