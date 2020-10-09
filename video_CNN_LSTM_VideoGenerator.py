#%load_ext tensorboard
import os
import glob
import keras
from keras_video import VideoFrameGenerator
from slidingIEEE import SlidingFrameGenerator
from keras.layers import GlobalMaxPool2D
from keras.layers import LSTM
from keras.layers import Bidirectional
from datetime import datetime
import pandas as pd
# use sub directories names as classes
classes = ['patting','feeding','diaper_change']
classes.sort()
# some global params
#SIZE = (112, 112)
#CHANNELS = 3
#NBFRAME = 5
#BS = 8

# Set size to 224, 224
SIZE = (224, 224)
CHANNELS = 3
NBFRAME = 8
BS = 50

# pattern to get videos and classes
glob_pattern='/part1/part1/MotiNagarVideoData/part4/part4/shubham/videoLimit/{classname}/*.mp4'
# for data augmentation
data_aug = keras.preprocessing.image.ImageDataGenerator(
    zoom_range=.1,
    horizontal_flip=True,
    rotation_range=8,
    width_shift_range=.2,
    height_shift_range=.2)
#Create video frame generator
# train = VideoFrameGenerator(
#     classes=classes, 
#     glob_pattern=glob_pattern,
#     nb_frames=NBFRAME,
#     split=.2, 
#     shuffle=True,
#     batch_size=BS,
#     target_shape=SIZE,
#     nb_channel=CHANNELS,
#     transformation=data_aug,
#     use_frame_cache=True)
train = SlidingFrameGenerator(
    sequence_time=8,
    classes=classes, 
    glob_pattern=glob_pattern,
    nb_frames=NBFRAME,
    split=.2, 
    shuffle=True,
    batch_size=BS,
    target_shape=SIZE,
    nb_channel=CHANNELS,
    transformation=data_aug,
    use_frame_cache=True)
logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)


valid = train.get_validation_generator()
import keras_video.utils
#keras_video.utils.show_sample(train)

len(train)
from keras.layers import GlobalMaxPool2D
from tensorflow.keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3

def build_inceptionV3(shape=(224, 224, 3), nbout=3):
    #model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model = InceptionV3(
        weights='imagenet',
        include_top=False,
        pooling='None',
        input_shape=(224, 224, 3))
    model.trainable = False
    # Keep 9 layers to train﻿﻿
#     trainable = 9
#     for layer in model.layers[:-trainable]:
#         layer.trainable = False
#     for layer in model.layers[-trainable:]:
#         layer.trainable = True
    output = keras.layers.AveragePooling2D()
    return keras.Sequential([model, output])

from keras.layers import TimeDistributed, LSTM, Dense, Dropout, AveragePooling2D, Flatten, BatchNormalization
def action_model(shape=(NBFRAME, 224, 224, 3), nbout=3):
    # Create our convnet with (112, 112, 3) input shape
    #convnet = build_convnet(shape[1:])
    convnet = build_inceptionV3(shape[1:])
    # then create our final model
    model = keras.Sequential()
    # add the convnet with (5, 112, 112, 3) shape
    model.add(TimeDistributed(convnet, input_shape=shape))
    model.add(TimeDistributed(Flatten()))
    model.add(Bidirectional(LSTM(units=1024,activation='tanh',return_sequences=True,recurrent_dropout=0.5,dropout=0.3)))
    # here, you can also use GRU or LSTM
    #model.add(LSTM(1024, activation='relu', return_sequences=True))
    # and finally, we make a decision network
    model.add(LSTM(256, activation='relu'))
    model.add(Dropout(.25))
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(.25))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(.25))
    #model.add(BatchNormalization())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(nbout, activation='softmax'))
    model.summary()
    return model

INSHAPE=(NBFRAME,) + SIZE + (CHANNELS,) # (5, 224, 224, 3)
model = action_model(INSHAPE, len(classes))
optimizer = keras.optimizers.Adam()
#optimizer = keras.optimizers.SGD(lr=0.05)
model.compile(
    optimizer,
    'categorical_crossentropy',
    metrics=['acc']
)

EPOCHS=20
# create a "chkp" directory before to run that
# because ModelCheckpoint will write models inside
callbacks = [
    keras.callbacks.ReduceLROnPlateau(verbose=1),
    keras.callbacks.ModelCheckpoint(
        'chkp/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
        verbose=1),
    keras.callbacks.TensorBoard(log_dir=logdir,histogram_freq=1,write_images=True)
]
model.fit(
    train,
    validation_data=valid,
    verbose=1,
    epochs=EPOCHS
)

pred = model.predict(valid)

lenValid = len(valid)
answer = []

if BS == 1:
    for i in range (0,lenValid):
        answer.append(valid[i][1][0])
else:
    for i in range (0,lenValid):
        intermediateList = valid[i][1]
        for j in range (0, BS):
            answer.append(intermediateList[j])
            
            
df = pd.DataFrame(answer,columns=['answer_1','answer_2','answer_3'])
df.to_csv('answer_video.csv')

predList = pred.tolist()
df = pd.DataFrame(predList,columns=['pred_1','pred_2','pred_3'])
df.to_csv('pred_video.csv')


from sklearn.metrics import classification_report,confusion_matrix
import numpy as np

y_tp = []
y_pred = []
pred_final = pred.tolist()
   
for j in pred_final:
    print(j)
    y_tp.append(np.argmax(j))
for k in answer:
    print(k)
    y_pred.append(np.argmax(k))


print(confusion_matrix(y_tp,y_pred))

print(classification_report(y_tp,y_pred))

model.save('video_model_7oct.h5')

