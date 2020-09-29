# Seed value
# Apparently you may use different seed values at each stage
seed_value= 1

# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)
import pandas as pd

# 2. Set the `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set the `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)

# 4. Set the `tensorflow` pseudo-random generator at a fixed value
#tf.random.set_seed(seed_value)
# for later versions: 
#tf.compat.v1.set_random_seed(seed_value)

# 5. Configure a new global `tensorflow` session
from keras import backend as K
#session_conf = tf.ConfigProto()
#sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
#K.set_session(sess)
# for later versions:
# session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
# sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
# tf.compat.v1.keras.backend.set_session(sess)
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.optimizers import SGD
from sklearn.utils import shuffle
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.manifold import TSNE
#from keras.applications.vgg16 import VGG16
from keras.layers import LSTM
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
#import numpy as np
from inceptionV3 import InceptionV3PyTorch
from inceptionV3Transfer import InceptionV3TransferPyTorch

import glob
from keras.layers import LeakyReLU
from neonate_dataset import NeonatalDataset, collate_skip_empty, colors_per_class
import argparse
from tqdm import tqdm
import cv2
import torch
import os
import shutil
import re
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import time
import copy
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD

def get_features(dataset, batch, num_images,trainedModel,baseModel):
    print('get_features called on dataset. trainedModel=',trainedModel,'   baseModel=',baseModel)
    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    if(trainedModel):
      model = baseModel
      model.eval()
      model.to(device)
    else:
      # initialize our implementation of ResNet
      model = InceptionV3PyTorch(pretrained=True)
      model.eval()
      model.to(device)
    print('get_features model loaded')
    # read the dataset and initialize the data loader
    dataset = NeonatalDataset(dataset, num_images)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch, collate_fn=collate_skip_empty, shuffle=True)

    # we'll store the features as NumPy array of size num_images x feature_size
    features = None
    print('get_features data loaded')
    # we'll also store the image labels and paths to visualize them later
    labels = []
    image_paths = []
    """
    for batch in tqdm(dataloader, desc='Running the model inference'):
        images = batch['image'].to(device)
        labels += batch['label']
        image_paths += batch['image_path']
        print(type(images))
        print('Size=', images.size())

        with torch.no_grad():
            output = model.forward(images)

        current_features = output.cpu().numpy()
        print('Feature shape in get_features', len(current_features[0]))
        print('current_features=', current_features)
        if features is not None:
            features = np.concatenate((features, current_features))
        else:
            features = current_features
    print('get_features done ')

    """
    for batch in tqdm(dataloader, desc='Running the model inference'):
        images = batch['image'].to(device)
        labels += batch['label']
        image_paths += batch['image_path']
        print('len(images)=',len(images))
        print('len(labels) = ', len(labels))
        print('Size=', images.size())
        with torch.no_grad():
            if trainedModel:
                output = model.forwardEval(images)
            else:
                output = model.forward(images)
            #print('output shape',output.shape)

        current_features = output.cpu().numpy()
        print('current_features shape',current_features.shape)
        print('Feature shape in get_features', len(current_features))
        print('current_features=', current_features)
        if features is not None:
            features = np.concatenate((features, current_features))
        else:
            features = current_features
    print('get_features done ',len(features),len(labels), len(image_paths))


    return features, labels, image_paths

# scale and move the coordinates so they fit [0; 1] range
def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))

    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)

    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range


def scale_image(image, max_image_size):
    image_height, image_width, _ = image.shape

    scale = max(1, image_width / max_image_size, image_height / max_image_size)
    image_width = int(image_width / scale)
    image_height = int(image_height / scale)

    image = cv2.resize(image, (image_width, image_height))
    return image


def draw_rectangle_by_class(image, label):
    image_height, image_width, _ = image.shape

    # get the color corresponding to image class
    color = colors_per_class[label]
    image = cv2.rectangle(image, (0, 0), (image_width - 1, image_height - 1), color=color, thickness=5)

    return image


def compute_plot_coordinates(image, x, y, image_centers_area_size, offset):
    image_height, image_width, _ = image.shape

    # compute the image center coordinates on the plot
    center_x = int(image_centers_area_size * x) + offset

    # in matplotlib, the y axis is directed upward
    # to have the same here, we need to mirror the y coordinate
    center_y = int(image_centers_area_size * (1 - y)) + offset

    # knowing the image center, compute the coordinates of the top left and bottom right corner
    tl_x = center_x - int(image_width / 2)
    tl_y = center_y - int(image_height / 2)

    br_x = tl_x + image_width
    br_y = tl_y + image_height

    return tl_x, tl_y, br_x, br_y


def visualize_tsne_images(tx, ty, images, labels, plot_size=1000, max_image_size=100):
    # we'll put the image centers in the central area of the plot
    # and use offsets to make sure the images fit the plot
    offset = max_image_size // 2
    image_centers_area_size = plot_size - 2 * offset

    tsne_plot = 255 * np.ones((plot_size, plot_size, 3), np.uint8)

    # now we'll put a small copy of every image to its corresponding T-SNE coordinate
    for image_path, label, x, y in tqdm(
            zip(images, labels, tx, ty),
            desc='Building the T-SNE plot',
            total=len(images)
    ):
        print(image_path)
        image = cv2.imread(image_path)

        # scale the image to put it to the plot
        print(image)
        print(max_image_size)
        image = scale_image(image, max_image_size)

        # draw a rectangle with a color corresponding to the image class
        image = draw_rectangle_by_class(image, label)

        # compute the coordinates of the image on the scaled plot visualization
        tl_x, tl_y, br_x, br_y = compute_plot_coordinates(image, x, y, image_centers_area_size, offset)

        # put the image to its TSNE coordinates using numpy subarray indices
        tsne_plot[tl_y:br_y, tl_x:br_x, :] = image

    plt.imshow(tsne_plot[:, :, ::-1])
    plt.show()


def visualize_tsne_points(tx, ty, labels):
    # initialize matplotlib plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # for every class, we'll add a scatter plot separately
    for label in colors_per_class:
        # find the samples of the current class in the data
        indices = [i for i, l in enumerate(labels) if l == label]

        # extract the coordinates of the points of this class only
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)

        # convert the class color to matplotlib format:
        # BGR -> RGB, divide by 255, convert to np.array
        color = np.array([colors_per_class[label][::-1]], dtype=np.float) / 255

        # add a scatter plot with the correponding color and label
        ax.scatter(current_tx, current_ty, c=color, label=label)

    # build a legend using the labels we set previously
    ax.legend(loc='best')

    # finally, show the plot
    plt.show()


def visualize_tsne(tsne, images, labels, plot_size=1000, max_image_size=100):
    # extract x and y coordinates representing the positions of the images on T-SNE plot
    tx = tsne[:, 0]
    ty = tsne[:, 1]

    # scale and move the coordinates so they fit [0; 1] range
    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    # visualize the plot: samples as colored points
    visualize_tsne_points(tx, ty, labels)

    # visualize the plot: samples as images
    visualize_tsne_images(tx, ty, images, labels, plot_size=plot_size, max_image_size=max_image_size)


colors_per_class = {
    'diaper_change' : [254, 202, 87],
    'patting' : [255, 107, 107],
    'feeding' : [10, 189, 227],
}

# class Net(Module):   
#     def __init__(self):
#         super(Net, self).__init__()

#         self.cnn_layers = Sequential(
#             # Defining a 2D convolution layer
#             Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
#             BatchNorm2d(4),
#             ReLU(inplace=True),
#             MaxPool2d(kernel_size=2, stride=2),
#             # Defining another 2D convolution layer
#             Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
#             BatchNorm2d(4),
#             ReLU(inplace=True),
#             MaxPool2d(kernel_size=2, stride=2),
#         )

#         self.linear_layers = Sequential(
#             Linear(4 * 7 * 7, 10)
#         )

#     # Defining the forward pass    
#     def forward(self, x):
#         x = self.cnn_layers(x)
#         x = x.view(x.size(0), -1)
#         x = self.linear_layers(x)
#         return x

def train_model(model, criterion, optimizer, num_epochs=10):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train_test', 'validate_test']:
            if phase == 'train':
                #scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            current_loss = 0.0
            current_corrects = 0

            # Here's where the training happens
            print('Iterating through data...')

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # We need to zero the gradients, don't forget it
                optimizer.zero_grad()

                # Time to carry out the forward training poss
                # We only need to log the loss stats if we are in training phase
                with torch.set_grad_enabled(phase == 'train_test'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train_test':
                        loss.backward()
                        optimizer.step()

                # We want variables to hold the loss statistics
                current_loss += loss.item() * inputs.size(0)
                current_corrects += torch.sum(preds == labels.data)

            epoch_loss = current_loss / dataset_sizes[phase]
            epoch_acc = current_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # Make a copy of the model if the accuracy on the validation set has improved
            if phase == 'validate_test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_since = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_since // 60, time_since % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # Now we'll load in the best model weights and return it
    model.load_state_dict(best_model_wts)
    return model

def load_InceptionV3_model(dataloader_train):
    modelName = 'cnn_29august_balanced.h5'
    modelExists = os.path.isfile('./'+modelName)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = InceptionV3TransferPyTorch().to(device)
    if modelExists:
        model = torch.load('./'+modelName)
    else:
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        # Train the model
        total_step = len(train_loader)
        for epoch in range(num_epochs):
            model.train()
            for i, (images, labels) in enumerate(train_loader):
                images = images.to(device)
                labels = labels.to(device)
                #print(type(images))
                #print(type(labels))
                #print('Size=', images.size())
                #print('DIM=', images.dim())

                # Backward and optimize
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model.forward(images)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                running_corrects = torch.sum(preds == labels.data)
                print(len(labels.data))
                print(type(running_corrects))
                loss.backward()
                optimizer.step()
                print ('Epoch [{}/{}], Step [{}/{}], Accuracy: {:.4f}, Loss: {:.4f}' 
                           .format(epoch+1, num_epochs, i+1, total_step,running_corrects, loss.item()))
        torch.save(model, modelName)
    return model





#   pretrained_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(224,224,3), pooling='avg')
#   pretrained_model.trainable = False  
#   model = Sequential()
#   model.add(pretrained_model)
#   model.add(Dense(2048))
#   model.add(Dense(3, activation='softmax'))
#   model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#   es =   EarlyStopping(monitor='val_loss', patience=8, verbose=0)
#   transfer = model.fit_generator(train_generator,epochs=1,validation_data=validation_generator,callbacks=[es])
#   model.summary()
#   base_model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
  
  #base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(224,224,3), pooling='avg')
  #base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(224,224,3), pooling='avg')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
learning_rate = 0.001
num_images = 1500
datasetPath = '/Users/harpreetsingh/part4/training_data/'
num_epochs = 1
num_classes = 3

#datasetPath = '/Users/harpreetsingh/CHIL/Research/Paper/Paper_3_b_NEO_Tiny/code/for_ieee_paper/raw-img'
#datasetPath = '/Users/chi/Documents/IEEE_Paper/train_test'
#datasetPath = '/Users/harpreetsingh/Downloads/trainfps_animal'
batch_size = 50

"""
#Without Transfer Learning plot the TSNE plot based on ImageNet Features
features, labels, image_paths = get_features(
    dataset=datasetPath+'back_validate/',
    batch=batch_size,
    num_images=num_images,trainedModel=False,
    baseModel = None
)
tsne = TSNE(n_components=2, n_iter=5000).fit_transform(features)
visualize_tsne(tsne, image_paths, labels)

"""

#Passing Images through model and making then 299 x 299 as required by pytorch inception v3 model
preprocess = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_data = datasets.ImageFolder(datasetPath+'back_train/', transform=preprocess)
print('data loaded', len(train_data))
train_loader = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True, num_workers=0)
print('train_loader length', len(train_loader))

#validation_data = datasets.ImageFolder(datasetPath+'validate/', transform=preprocess)
#dataloader_validate = torch.utils.data.DataLoader(validation_data, batch_size, shuffle=True, num_workers=0)

base_model = load_InceptionV3_model(train_loader)

#Custom model containing Inception V3 layer followed by custom layer to learn patting, diaper change, and feeding
#print('---------BASE MODEL CREATED WITH TRANSFER LEARNING ----------',base_model)
#With Transfer Learning plot the TSNE plot based on ImageNet Features
features, labels, image_paths = get_features(
    dataset=datasetPath+'back_validate/',
    batch=batch_size,
    num_images=num_images,
    trainedModel=True,
    baseModel = base_model
)
print('features length',features.shape)

tsne_transferLearning = TSNE(n_components=2, perplexity=35, n_iter=20000).fit_transform(features)
visualize_tsne(tsne_transferLearning, image_paths, labels)




