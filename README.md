# Mini-Project--Application-of-NN

(Expected the following details in the report )
## Project Title: 
Deep Neural Network for Malaria Infected Cell Recognition
## Project Description  
To develop a deep neural network for Malaria infected cell recognition and to analyze the performance.
## Algorithm:
To develop a deep neural network for Malaria infected cell recognition and to analyze the performance.
Use ImageDataGenerator to augment the data and flow the data directly from the dataset directory to the model.
Split the data into train and test.
Build the convolutional neural network
Train the model with training data
Evaluate the model with the testing data
Evaluate the model with the testing data
## Program:
```python
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow.keras import models
from sklearn.metrics import classification_report,confusion_matrix
import tensorflow as tf

# to share the GPU resources for multiple sessions
from tensorflow.compat.v1.keras.backend import set_session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True # dynamically grow the memory used on the GPU
config.log_device_placement = True # to log device placement (on which device the operation ran)
sess = tf.compat.v1.Session(config=config)
set_session(sess)

%matplotlib inline

# for college server
my_data_dir = 'content/cell_images'

os.listdir(my_data_dir)

test_path = my_data_dir+'/test/'
train_path = my_data_dir+'/train/'

os.listdir(train_path)

len(os.listdir(train_path+'/uninfected/'))

len(os.listdir(train_path+'/parasitized/'))

os.listdir(train_path+'/parasitized')[0]

para_img= imread(train_path+
                 '/parasitized/'+
                 os.listdir(train_path+'/parasitized')[0])
                 
plt.imshow(para_img)

# Checking the image dimensions
dim1 = []
dim2 = []
for image_filename in os.listdir(test_path+'/uninfected'):
    img = imread(test_path+'/uninfected'+'/'+image_filename)
    d1,d2,colors = img.shape
    dim1.append(d1)
    dim2.append(d2)
    
sns.jointplot(x=dim1,y=dim2)

image_shape = (130,130,3)

help(ImageDataGenerator)

image_gen = ImageDataGenerator(rotation_range=20, # rotate the image 20 degrees
                               width_shift_range=0.10, # Shift the pic width by a max of 5%
                               height_shift_range=0.10, # Shift the pic height by a max of 5%
                               rescale=1/255, # Rescale the image by normalzing it.
                               shear_range=0.1, # Shear means cutting away part of the image (max 10%)
                               zoom_range=0.1, # Zoom in by 10% max
                               horizontal_flip=True, # Allo horizontal flipping
                               fill_mode='nearest' # Fill in missing pixels with the nearest filled value
                              )
                              
image_gen.flow_from_directory(train_path)

image_gen.flow_from_directory(test_path)


model = models.Sequential()
model.add(layers.Conv2D(filters=32, kernel_size=(3,3),input_shape=image_shape,activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))

model.add(layers.Conv2D(filters=64, kernel_size=(3,3),activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))

model.add(layers.Conv2D(filters=64, kernel_size=(3,3),activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2,2)))

model.add(layers.Flatten())

model.add(layers.Dense(128))
model.add(layers.Activation('relu'))

model.add(layers.Dropout(0.5))

model.add(layers.Dense(1))
model.add(layers.Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
              
model.summary()

batch_size = 16

help(image_gen.flow_from_directory)

train_image_gen = image_gen.flow_from_directory(train_path,
                                               target_size=image_shape[:2],
                                                color_mode='rgb',
                                               batch_size=batch_size,
                                               class_mode='binary')
                                               
train_image_gen.batch_size

len(train_image_gen.classes)

train_image_gen.total_batches_seen

test_image_gen = image_gen.flow_from_directory(test_path,
                                               target_size=image_shape[:2],
                                               color_mode='rgb',
                                               batch_size=batch_size,
                                               class_mode='binary',shuffle=False)
                                               
train_image_gen.class_indices

results = model.fit(train_image_gen,epochs=20,
                              validation_data=test_image_gen
                             )
                             
model.save('cell_model.h5')

losses = pd.DataFrame(model.history.history)

losses[['loss','val_loss']].plot()

model.metrics_names

model.evaluate(test_image_gen)

pred_probabilities = model.predict(test_image_gen)

test_image_gen.classes

predictions = pred_probabilities > 0.5

print(classification_report(test_image_gen.classes,predictions))

confusion_matrix(test_image_gen.classes,predictions)
```
## Output:

## Training Loss, Validation Loss Vs Iteration Plot
![image](https://user-images.githubusercontent.com/105230321/205449681-3a5f9197-d25d-44d2-8194-1cd10346581c.png)
## Classification Report
![image](https://user-images.githubusercontent.com/105230321/205449708-6e815725-a414-4e95-8454-edc30054754e.png)
## Confusion Matrix
![image](https://user-images.githubusercontent.com/105230321/205449806-86115de9-9b54-438b-b9ea-d45253fc919e.png)
## Sample Data Prediction
![image](https://user-images.githubusercontent.com/105230321/205452594-ca6d1c2f-74aa-42d3-95c5-7c6faff330e4.png)

## Advantage :
Helps in recognizing the malaria cell
## Result:
Thus, successfully implemented convolutional neural network model for Malaria Infected Cell Regonition.
