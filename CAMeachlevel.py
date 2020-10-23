# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 15:55:39 2020

@author: Dreamen
"""
#%%
import keras
import cv2
import numpy as np
import keras.backend as K
import matplotlib.pyplot as plt
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing.image import load_img,img_to_array
from tqdm import tqdm
K.set_learning_phase(1) #set learning phase

import os
os.chdir(r'F:\Sunyongqing1\code\CAM')
from resnetmultilevel import resnet_self
import time

from keras.preprocessing.image import load_img,img_to_array
date1 = time.strftime('%m%d%H%M',time.localtime(time.time()))
input_shape = 224
batch_size = 32

model = resnet_self((input_shape, input_shape, 3))#resnet18
# model_weights_file = r'F:\Sunyongqing1\code\CAM/262-0.954-0.965.h5'
model_weights_file=r'F:\Sunyongqing1\code\CAM/titansun09160837-300-0.941-0.956.h5'
model.load_weights(model_weights_file)

# img_path = r'F:\Sunyongqing1\code\CAM\CAMpicture\posCAM\pos220IMG_20150826_1_17.jpg'
# model = keras.models.load_model(weight_file_dir)

os.chdir(r'F:\Sunyongqing1\code\CAM\grad-cam-keras-master/')
pathpatientdir =  r'F:\Sunyongqing1\code\CAM\CAMpicture\imagesv2' 
pathpatientdirCAMsave =  r'F:\Sunyongqing1\code\CAM\CAMpicture\CAMlevel1-16'
pathpatientdirlist=os.listdir(pathpatientdir)

for eachdir in tqdm(pathpatientdirlist):
    pic_folder = os.path.join(pathpatientdir,eachdir)
    pic_cam_folder = pathpatientdirCAMsave+f'/{eachdir}CAM/'
    if not os.path.exists(pic_cam_folder):
        os.makedirs(pic_cam_folder)
    list_name=os.listdir(pic_folder)
    for i, file_name in enumerate(list_name):
        img = load_img(pic_folder+'/' +file_name, target_size=(224, 224))
        x = img_to_array(img)
        x = np.expand_dims(x,axis=0)
        x = preprocess_input(x)
        pred = model.predict(x)
        class_idx = np.argmax(pred[0])

        class_output = model.output[:,class_idx]
        #需根据自己情况修改2. 把block5_conv3改成自己模型最后一层卷积层的名字
        '''
        level 6 act5a_branch2b
        level 5 : fpn_C4toC5  fpn_C4addC5
        level 4: fpn_C3addC4
        level 3: fpn_C2addC3
        level 2: fpn_C1addC2
        level 1: fpn_C1addC2
        '''
        last_conv_layer = model.get_layer("fpn_C1addC2")#act5a_branch2b fpn_C4addC5
        import keras.backend as K
        grads = K.gradients(class_output,last_conv_layer.output)[0]
        pooled_grads = K.mean(grads,axis=(0,1,2))
        iterate = K.function([model.input],[pooled_grads,last_conv_layer.output[0]])
        pooled_grads_value, conv_layer_output_value = iterate([x])
        ##需根据自己情况修改3. 512是我最后一层卷基层的通道数，根据自己情况修改
        for i in range(16):
            conv_layer_output_value[:,:,i] *= pooled_grads_value[i]

        heatmap = np.mean(conv_layer_output_value, axis=-1)
        heatmap = np.maximum(heatmap,0)
        heatmap /= np.max(heatmap)
        import cv2
        img = cv2.imread(pic_folder+'/' +file_name)
        img = cv2.resize(img,dsize=(224,224),interpolation=cv2.INTER_NEAREST)
        # img = img_to_array(image)
        heatmap = cv2.resize(heatmap,(img.shape[1],img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        # plt.matshow(heatmap)
        # cv2.imshow('aa.jpg',heatmap)
        heatmap = cv2.applyColorMap(heatmap,cv2.COLORMAP_JET)
        superimposed_img = cv2.addWeighted(img,0.6,heatmap,0.4,0)
        superimposed_img1 = img + heatmap*0.6
        # cv2.imshow('Grad-cam',superimposed_img)
        # cv2.imshow('Grad-cam1',superimposed_img1)
        cv2.imwrite(pic_cam_folder + file_name,superimposed_img1)
# %%
