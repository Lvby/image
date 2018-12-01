from __future__ import division, print_function, absolute_import
import pandas as pd
from glob import glob
import fnmatch
import cv2
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, learning_curve, GridSearchCV
from keras.optimizers import SGD, RMSprop, Adam, Adagrad, Adadelta
from sklearn.utils import class_weight
#Import tflearn and some helpers
import tflearn
from tflearn.data_utils import shuffle
import os,sys,time,signal
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
import datetime as dt
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Sequential, model_from_json,load_model,Model
from tensorflow.keras import layers
from tensorflow.keras import Model
import keras.utils
from keras import utils as np_utils

from keras.utils.np_utils import to_categorical
#import seaborn as sns

#####1. 数据处理读入整理增强#####

# 1)read data /the dataset has already been marked as class0=no cancer and class1=has cancer

image = glob( '../his/**/*.png',recursive=True)
#image = glob( '../dataImg/*/**/*.png',recursive=True)
lowerIndex=0
upperIndex = len(image)
class0 = '../his/0/*.png'
class1 = '../his/1/*.png'
classZero = fnmatch.filter(image, class0)
classOne = fnmatch.filter(image, class1)
#print(classZero)

def process_images(lowerIndex,upperIndex):
	X = []
	Y = []
	WIDTH=99
	HEIGHT=150

	for img in image[lowerIndex:upperIndex]:
		full_size_image = cv2.imread(img)
		X.append(cv2.resize(full_size_image, (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC))
		if img in classZero:
			Y.append(0)
		elif img in classOne:
			Y.append(1)
		else: 
			return
	return X, Y

X,Y = process_images(0,upperIndex)


df = pd.DataFrame(image)#load the dataset as a panda dataframe
df["images"]=X
df["labels"]=Y

X=np.array(X)


# 2)split training and testing 分开训练集与测试集

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size=0.20,  random_state=123,stratify=Y)



# plotImages=classZero
# i_ = 0
# plt.rcParams['figure.figsize'] = (10.0, 10.0)
# plt.subplots_adjust(wspace=0, hspace=0)
# for l in plotImages[:4]:
# 	im = cv2.imread(l)
# 	im = cv2.resize(im, (50, 50)) 
# 	color=[255,255,255]
# 	new_im = cv2.copyMakeBorder(im,20,20, 20,20, 20,
#     value=color)
# 	plt.subplot(1, 4, i_+1).set_title("Benign")
# 	plt.imshow(new_im); plt.axis('on')
# 	i_ += 1
# plt.show()

# plotImages=classOne
# i_ = 0
# plt.rcParams['figure.figsize'] = (10.0, 10.0)
# plt.subplots_adjust(wspace=0, hspace=0)
# for l in plotImages[:4]:
# 	im = cv2.imread(l)
# 	im = cv2.resize(im, (50, 50)) 
# 	color=[255,255,255]
# 	new_im = cv2.copyMakeBorder(im,20,20, 20,20, 20,
#     value=color)
# 	plt.subplot(1, 4, i_+1).set_title("Malignant")
# 	plt.imshow(new_im); plt.axis('on')
# 	i_ += 1
# plt.show()

img_input = layers.Input(shape=(150, 99, 3))


# First convolution extracts 16 filters that are 3x3
# Convolution is followed by max-pooling layer with a 2x2 window
x = layers.Conv2D(16, 3, activation='relu')(img_input)
x = layers.MaxPooling2D(2)(x)

# Second convolution extracts 32 filters that are 3x3
# Convolution is followed by max-pooling layer with a 2x2 window
x = layers.Conv2D(32, 3, activation='relu')(x)
x = layers.MaxPooling2D(2)(x)

# Third convolution extracts 64 filters that are 3x3
# Convolution is followed by max-pooling layer with a 2x2 window
x = layers.Conv2D(64, 3, activation='relu')(x)
x = layers.MaxPooling2D(2)(x)

# Flatten feature map to a 1-dim tensor so we can add fully connected layers
x = layers.Flatten()(x)

# Create a fully connected layer with ReLU activation and 512 hidden units
x = layers.Dense(512, activation='relu')(x)

 #Add a dropout rate of 0.5
x = layers.Dropout(0.5)(x)

x = layers.Flatten()(x)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x = layers.Dense(1024, activation='relu')(x)
# Add a dropout rate of 0.2
x = layers.Dropout(0.2)(x)

# Create output layer with a single node and sigmoid activation
output = layers.Dense(1, activation='sigmoid')(x)

# Create model:
# input = input feature map
# output = input feature map + stacked convolution/maxpooling layers + fully 
# connected layer + sigmoid output layer
model = Model(img_input, output)

model.summary()



from tensorflow.keras.optimizers import RMSprop

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['acc'])
from tensorflow.keras.preprocessing.image import ImageDataGenerator


datagen = ImageDataGenerator(
        rotation_range=20,
    	width_shift_range=0.2,
    	height_shift_range=0.2,
    	horizontal_flip=True,
    	vertical_flip=True,
    	rescale=1./255,  
    	shear_range=0.2,
    	zoom_range=0.2

  )


history = model.fit_generator(
	  datagen.flow(X_train, Y_train, batch_size=50),      
      steps_per_epoch=len(X_train)/32,   # 2000 images = batch_size * steps
      epochs=20,
      verbose=2,
      validation_data=[X_test, Y_test]
      )




# plt.figure(figsize=(8,8))
# plt.subplot(1,2,1)
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.savefig('./accuracy_curve.png')
# plt.clf()
# # summarize history for loss
# plt.subplot(1,2,2)
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.savefig('./loss_curve.png')


# # # #print(model)

# # # #visulaizing intermediate prepresentation 训练图片加工后的效果

# # #Evaluating accuracy and loss for the model

# # # Retrieve a list of accuracy results on training and test data
# # # sets for each training epoch

# # filt = ['acc'] # try to add 'loss' to see the loss learning curve
# # for k in filter(lambda x : np.any([kk in x for kk in filt]), metrics.keys()):
# # 	l = np.array(metrics[k])
# # 	plt.plot(l, c= 'r' if 'val' not in k else 'b', label='val' if 'val' in k else 'train')
# # 	x = np.argmin(l) if 'loss' in k else np.argmax(l)
# # 	y = l[x]
# # 	plt.scatter(x,y, lw=0, alpha=0.25, s=100, c='r' if 'val' not in k else 'b')
# # 	plt.text(x, y, '{} = {:.4f}'.format(x,y), size='15', color= 'r' if 'val' not in k else 'b')  
# # 	plt.show() 
# # plt.legend(loc=4)
# # plt.axis([0, None, None, None])
# # plt.grid()
# # plt.xlabel('Number of epochs')


# def plot_confusion_matrix(cm, classes,
#                           normalize=False,
#                           title='Confusion matrix',
#                           cmap=plt.cm.Blues):
#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     """
#     plt.figure(figsize = (5,5))
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=90)
#     plt.yticks(tick_marks, classes)
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, cm[i, j],
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")
#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
 


# # #clean up

# # #import os, signal moved to the top
os.kill(os.getpid(), signal.SIGKILL)



# # # #### cleanup ####




