import PIL

from keras import models
from keras import layers
from tensorflow.keras import optimizers
import os
import glob
import shutil
import sys
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
import os
from tensorflow.keras import callbacks
import pandas as pd

from tensorflow.keras import callbacks
from keras.callbacks import Callback

os.environ["CUDA_VISIBLE_DEVICES"]="1" 

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

batch_size = 8
epochs = 500

#Train
dataframe = pd.read_csv('/home/yupaporn/code/EffNet-Person-Classify/data_label_by_tan_10082023_splited_imgpath28_train_1img.csv') 
dataframe = dataframe[dataframe['img_no'] ==1
base_dir = '/media/SSD/BearHouse-Person-Data/'
os.chdir(base_dir)
train_dir = os.path.join(base_dir, 'train')

#validation
valframe = pd.read_csv( '/home/yupaporn/code/EffNet-Person-Classify/data_label_by_tan_10082023_splited_imgpath28_val_1img.csv') 
valframe = valframe[valframe['img_no'] ==1]
validation_dir = os.path.join(base_dir, 'validation')


from efficientnet.keras import EfficientNetB5 as Net
from efficientnet.keras import center_crop_and_resize, preprocess_input
conv_base = Net(weights='imagenet')
height = width = conv_base.input_shape[1]
input_shape = (height, width, 3)

# loading pretrained conv base model
conv_base = Net(weights='imagenet', include_top=False, input_shape=input_shape)

# create new model with a new classification layer
x = conv_base.output  
global_average_layer = layers.GlobalAveragePooling2D(name = 'head_pooling')(x)
dropout_layer_1 = layers.Dropout(0.50,name = 'head_dropout')(global_average_layer)
prediction_layer = layers.Dense(3, activation='softmax',name = 'prediction_layer')(dropout_layer_1)

model = models.Model(inputs= conv_base.input, outputs=prediction_layer) 
model.summary()

#showing before&after freezing
print('This is the number of trainable layers '
      'before freezing the conv base:', len(model.trainable_weights))
#conv_base.trainable = False  # freeze to keep convolutional base's weight
for layer in conv_base.layers:
    layer.trainable = False
print('This is the number of trainable layers '
      'after freezing the conv base:', len(model.trainable_weights))  #freez แล้วจะเหลือ max pool and dense

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=30,
      width_shift_range=0.2,
      height_shift_range=0.2,
      brightness_range=[0.5,1.5],
      shear_range=0.4,
      zoom_range=0.2,
      horizontal_flip=False,
      fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
        dataframe = dataframe,
        directory = train_dir,
        x_col = 'img_28_train',
        y_col = 'Category',
        target_size = (height, width),
        batch_size=batch_size,
        color_mode= 'rgb',
        class_mode='categorical')
test_generator = test_datagen.flow_from_dataframe(
        dataframe = valframe,
        directory = validation_dir,
        x_col = 'img_28_train',
        y_col = 'Category',
        target_size = (height, width),
        batch_size=batch_size,
        color_mode= 'rgb',
        class_mode='categorical')

os.chdir('/media/SSD/BearHouse-Person-Model') #เปลี่ยนตาม fold

root_logdir = '/media/SSD/BearHouse-Person-Model/mylogsB5_500_category_r1'  #เปลี่ยน R1_1 เปลี่ยนตาม fold
def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d_%H_%M_%S")
    return os.path.join(root_logdir,run_id)
run_logdir = get_run_logdir()

tensorboard_cb = callbacks.TensorBoard(log_dir = run_logdir)

# os.makedirs("./models", exist_ok=True)

def avoid_error(gen):
    while True:
        try:
            data, labels = next(gen)
            yield data, labels
        except:
            pass

 #Training
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['acc'])

history = model.fit_generator(
      avoid_error(train_generator),
      steps_per_epoch= len(dataframe)//batch_size,
      epochs=epochs,
      validation_data=avoid_error(test_generator), 
      validation_steps= len(valframe) //batch_size,
      callbacks = [tensorboard_cb])

model.save('./models/B5R1_Category_500.h5')



