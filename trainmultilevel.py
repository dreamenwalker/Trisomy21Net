# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 17:37:37 2020

@author: Dreamen

for sunyongqing construct the deep learning model

"""

import os
from keras.models import Sequential, Model
#Import from keras_preprocessing not from keras.preprocessing
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers

import pandas as pd
import numpy as np
import sys
from sklearn.metrics import roc_auc_score,accuracy_score
sys.path.append("/data/zlw/sunyongqing1/code/")
from resnetmultilevel import resnet_self
import sys
sys.path.append("./code")
from callback import MultipleClassAUROC
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping,CSVLogger
## default parameter
from configparser import ConfigParser
config_file = "./code/config.ini"
cp = ConfigParser()
cp.read(config_file)
class_names = cp["DEFAULT"].get("class_names").split(",")
generator_workers = cp["TRAIN"].getint("generator_workers")
use_trained_model_weights = cp["TRAIN"].getboolean("use_trained_model_weights")
if use_trained_model_weights:
        # resuming mode
        print("** use trained model weights **")
        # load training status for resuming
        training_stats_file = os.path.join(output_dir, ".training_stats.json")
        if os.path.isfile(training_stats_file):
            # TODO: add loading previous learning rate?
            training_stats = json.load(open(training_stats_file))
        else:
            training_stats = {}
else:
        # start over
        training_stats = {}
from keras import backend as K
import tensorflow as tf
def focal_loss(gamma=2, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        alpha_t = y_true*alpha + (K.ones_like(y_true)-y_true)*alpha
        p_t = y_true*y_pred + (K.ones_like(y_true)-y_true)*(K.ones_like(y_true)-y_pred) + K.epsilon()
        FL = - alpha_t * K.pow((K.ones_like(y_true)-p_t),gamma) * K.log(p_t)
        return K.mean(FL)
    return focal_loss_fixed
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.chdir('/data/zlw/sunyongqing1')
import time
date1 = time.strftime('%m%d%H%M',time.localtime(time.time()))
source_image_dir = r'./data/negative'
# description see the log record
pathneg = pd.read_csv('./csv/negativepathall0906order.csv')# with order for patient ID
pathpos = pd.read_csv('./csv/positivepathall0906order.csv')
'''*****path******'''
os.chdir("/data/zlw/sunyongqing1/")
output_dir = './h5/'+ f'{date1}'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
roc_dir = f'./auc/{date1}/weights.h5'
if not os.path.exists(os.path.dirname(roc_dir)):
    os.mkdir(os.path.dirname(roc_dir))
saveriskpath = './csv/risk'
tensorboard_dir = './tensorboard'
if not os.path.exists(saveriskpath):
    os.mkdir(saveriskpath)
output_dirAccAUC = './auc/'+ f'{date1}'
if not os.path.exists(output_dirAccAUC):
    os.mkdir(output_dirAccAUC)
#****************************************************

#906
pathtrain = pd.concat([pathneg[:1500],pathpos[:800]])# 767 to end approximately 30 patients
# pathval = pd.concat([pathneg[600:900],pathpos[767:]])
pathtest= pd.concat([pathneg[1500:1700],pathpos[800:]])
#P2
pathtrain = pd.concat([pathneg[:699],pathneg[902:1709],pathpos[0:701],pathpos[851:]])
pathtest = pd.concat([pathneg[699:902],pathpos[701:851]])

#P1
pathtrain = pd.concat([pathneg[:699],pathneg[902:1709],pathpos[0:700],pathpos[850:]])
pathtest = pd.concat([pathneg[699:902],pathpos[700:850]])

pathtrain = pd.concat([pathneg[:100],pathneg[300:1709],pathpos[0:700],pathpos[850:]])
pathtest = pd.concat([pathneg[100:300],pathpos[700:850]])
# pathtrain,pathval=pathall,pathall
#check postitive patient

# traindf=pd.read_csv('./trainLabels.csv',dtype=str)
# testdf=pd.read_csv("./sampleSubmission.csv",dtype=str)
# traindf["id"]=traindf["id"].apply(append_ext)
# testdf["id"]=testdf["id"].apply(append_ext)
# datagen=ImageDataGenerator(rescale=1./255.,validation_split=0.25)
datagen = ImageDataGenerator(rescale=1./255.,validation_split=0.20,
      rotation_range=20,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      vertical_flip=True,
      fill_mode='nearest')
train_generator=datagen.flow_from_dataframe(
                    dataframe=pathtrain,
                    directory=None,
                    x_col="originpath",
                    y_col="label",
                    # subset="training",
                    batch_size=16,
                    seed=1,
                    shuffle=True,
                    class_mode="binary",#if more than one class, here should be categorical
                    target_size=(224,224))

test_datagen=ImageDataGenerator(rescale=1./255.)
valid_generator=test_datagen.flow_from_dataframe(
                dataframe=pathtrain,
                directory=None,
                x_col="originpath",
                y_col="label",
                # subset="validation",
                batch_size=16,
                seed=1,
                shuffle=False,
                class_mode="binary",
                target_size=(224,224))

test_datagen=ImageDataGenerator(rescale=1./255.)
test_generator=test_datagen.flow_from_dataframe(
                dataframe=pathtest,
                directory=None,
                x_col="originpath",
                y_col='label',
                batch_size=16,
                seed=42,
                shuffle=False,
                class_mode="binary",
                target_size=(224,224))

input_shape = 224
batch_size = 32
model = resnet_self((input_shape, input_shape, 3))#resnet18
model_weights_file =None
# model_weights_file = '/data/zlw/sunyongqing1/h5/09261209/179-0.959-0.926.h5' for P2
# model_weights_file = '/data/zlw/sunyongqing1/h5/09271051/300-0.858-0.841.h5' # for P1
if model_weights_file:
    model.load_weights(model_weights_file)
#    for layer in model.layers: #total 0-66[:45]
#          layer.trainable = False
# for layer in model.layers:
#     layer.trainable = False
model.compile(optimizers.Adam(lr=0.0001),loss="mean_squared_error", metrics=["accuracy"])#mean_squared_error binary_crossentropy

auc = MultipleClassAUROC(sequence=valid_generator,sequence1=test_generator,pathtrain = pathtrain,df= pathtest,
             saveriskpath=saveriskpath,class_names=class_names,weights_path=roc_dir, log_path=output_dirAccAUC)
checkpoint = ModelCheckpoint(
        filepath=os.path.join(output_dir, "{epoch:02d}-{accuracy:.3f}-{val_accuracy:.3f}.h5"),
        save_weights_only=True,save_best_only=False,verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=30)
csv_logger = CSVLogger(os.path.join(output_dir, 'training.log'))
callbacks = [
            checkpoint,
            TensorBoard(log_dir=os.path.join(tensorboard_dir, "logs"), batch_size=batch_size),
            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10,
                              verbose=1, mode="min", min_lr=1e-6),
            csv_logger,
            auc,
            # early_stopping,
                ]
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
history = model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    callbacks=callbacks, 
                    epochs=300)

print(f'*********the log of AUC path is {output_dirAccAUC}********')
print(f'*********the weight path is {output_dir}********')
'''
model.save('./h5load/weight.h5')

acc = history.history['accuracy']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


# -*- coding=utf-8 -*-
"""
Created on 2019-6-19 21:39:53
@author: fangsh.Alex
"""
import keras
import cv2
import numpy as np
import keras.backend as K
 
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import load_img,img_to_array
K.set_learning_phase(1) #set learning phase
 
 
# weight_file_dir = '/data/sfang/logo_classify/keras_model/checkpoint/best_0617.hdf5'
img_path = '/data/zlw/sunyongqing1/CAMpicture/fetusforCAM.jpg'
 
# model = keras.models.load_model(weight_file_dir)
image = load_img(img_path,target_size=(224,224))
 
x = img_to_array(image)
x = np.expand_dims(x,axis=0)
x = preprocess_input(x)
pred = model.predict(x)
class_idx = np.argmax(pred[0])
 
class_output = model.output[:,class_idx]
last_conv_layer = model.get_layer("res5a_branch2b")
gap_weights = model.get_layer("global_pool6")
 
grads = K.gradients(class_output,gap_weights.output)[0]
iterate = K.function([model.input],[grads,last_conv_layer.output[0]])
pooled_grads_value, conv_layer_output_value = iterate([x])
pooled_grads_value = np.squeeze(pooled_grads_value,axis=0)
for i in range(512):
    conv_layer_output_value[:,:,i] *= pooled_grads_value[i]
 
heatmap = np.mean(conv_layer_output_value, axis=-1)
heatmap = np.maximum(heatmap,0)#relu激活。
heatmap /= np.max(heatmap)
#
img = cv2.imread(img_path)
img = cv2.resize(img,dsize=(224,224),interpolation=cv2.INTER_NEAREST)
# img = img_to_array(image)
heatmap = cv2.resize(heatmap,(img.shape[1],img.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap,cv2.COLORMAP_JET)
superimposed_img = cv2.addWeighted(img,0.6,heatmap,0.4,0)
cv2.imwrite(f'./CAMpicture/CAM{date1}.jpg', superimposed_img)
# cv2.imshow('Grad-cam',superimposed_img)
cv2.waitKey(0)
'''
