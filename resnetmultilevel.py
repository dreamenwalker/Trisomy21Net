# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 15:35:42 2020
@author: Dreamen
"""
from __future__ import print_function
import numpy as np
import warnings
import keras
from keras import layers
from keras.layers import Input,Dense,Activation,Flatten,Conv2D,MaxPooling2D,GlobalMaxPooling2D,ZeroPadding2D
from keras.layers import GlobalAveragePooling2D,AveragePooling2D,BatchNormalization,Lambda,Multiply
from keras.models import Model
from keras.preprocessing import image
from keras.regularizers import l1,l2,l1_l2
import keras.backend as K
import keras.layers as KL
import keras.engine as KE
import keras.models as KM
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras_applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs
# import tensorflow as tf
# tf.test.gpu_device_name()

def L12_reg(weight_matrix):
    return None
    # return 0.01 * K.sum(K.abs(weight_matrix)) + 0.01 * K.sum(K.pow(weight_matrix,2))

def se_block(input_tensor, c=16):#c is reduction ratio
    num_channels = int(input_tensor._keras_shape[-1]) # Tensorflow backend
    bottleneck = int(num_channels // c)

    se_branch = GlobalAveragePooling2D()(input_tensor)
    se_branch = Dense(bottleneck, use_bias=False, activation='relu')(se_branch)
    se_branch = Dense(num_channels, use_bias=False, activation='sigmoid')(se_branch)

    out = Multiply()([input_tensor, se_branch])
    return out
def identity_block(input_tensor, kernel_size, filters, stage, block,use_bias=True, train_bn=True):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    stage is phase for different
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    bn_axis = 3
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    x = Conv2D(filters1, (3, 3), strides= (1,1),padding="SAME", name=conv_name_base + '2a', kernel_initializer="he_normal",
                      kernel_regularizer=L12_reg)(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, kernel_initializer="he_normal",
               padding='same', name=conv_name_base + '2b',kernel_regularizer=L12_reg)(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    # x = Activation('relu')(x)

    # x = Conv2D(filters3, kernel_size,(1, 1), name=conv_name_base + '2c', kernel_initializer="he_normal",
    #                   kernel_regularizer=L12_reg)(x)
    # x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2), use_bias=True, train_bn=True):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.

    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters

    bn_axis = 3

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1,  kernel_size=(3, 3),
                           padding="same",strides=strides,kernel_regularizer=L12_reg, kernel_initializer="he_normal",
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', kernel_initializer="he_normal",
               name=conv_name_base + '2b',kernel_regularizer=L12_reg)(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    # x = Activation('relu')(x)# the relu function used for the subsection and input is the sum of

    # x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c', kernel_initializer="he_normal",
    #                   kernel_regularizer=L12_reg)(x)
    # x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)
    # x = se_block(x)
    shortcut = Conv2D(filters3, (1, 1), strides=strides, kernel_regularizer=L12_reg, kernel_initializer="he_normal",
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x

#%%
def resnet_self(input_tensor = None, include_top=True,num_outputs=1,
                 input_shape=(224,224,3), architecture = 'resnet50', stage5=True, train_bn=True):
    """Build a ResNet graph.
        architecture: Can be resnet50 or resnet101
        stage5: Boolean. If False, stage5 of the network is not created
        train_bn: Boolean. Train or freeze Batch Norm layers
    """
    img_input = Input(shape=input_shape,name = 'input')

    # assert architecture in ["resnet50", "resnet101"]
    # Stage 1
    # CC = x = KL.ZeroPadding2D((3, 3))(img_input)
    C0 = x = KL.Conv2D(32, (3, 3), strides=(1, 1), name='conv1', padding="same",use_bias=True, kernel_initializer="he_normal",
                      kernel_regularizer=L12_reg)(img_input)
    x = BatchNormalization(axis=3, name='bn_conv1')(x)
    x = KL.Activation('relu')(x)
    C1 = x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    # Stage 2
    C2 = x = conv_block(x, 3, [64, 64, 64], stage=2, block='a', strides=(2, 2), train_bn=train_bn)
    # x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', train_bn=train_bn)
    # C2 = x = identity_block(x, 3, [64, 64, 64], stage=2, block='c', train_bn=train_bn)
    # Stage 3
    C3 = x = conv_block(x, 3, [128, 128, 128], stage=3, block='a', train_bn=train_bn)
    # x = identity_block(x, 3, [128, 128, 512], stage=3, block='b', train_bn=train_bn)# at shortcut no conv see notebook
    # x = identity_block(x, 3, [128, 128, 512], stage=3, block='c', train_bn=train_bn)
    # C3 = x = identity_block(x, 3, [128, 128, 128], stage=3, block='d', train_bn=train_bn)
    # Stage 4
    C4 = x = conv_block(x, 3, [256, 256, 256], stage=4, block='a', train_bn=train_bn)
    # block_count = {"resnet50": 5, "resnet101": 22}[architecture] # if architecture is resnet50, the block_count is 5
    # for i in range(block_count):
    # C4 = x = identity_block(x, 3, [256, 256, 256], stage=4, block=chr(98 + i), train_bn=train_bn)#chr(98) is b
    # C4 = x
    # Stage 5
    if stage5:
        C5 =x = conv_block(x, 3, [512, 512, 512], stage=5, block='a', train_bn=train_bn)
        # x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', train_bn=train_bn)
        # C5 = x = identity_block(x, 3, [512, 512, 512], stage=5, block='c', train_bn=train_bn)
    else:
        C5 = None
    # _, C2, C3, C4, C5 = resnet_graph(input_image=None, architecture="resnet50",stage5=True, train_bn=True)

    # rpn_feature_maps = [P2, P3, P4, P5, P6]
    #C1 out 55*55*64 the channel of C1 is the same as the C2, so no pooling
    # C12_input = KL.Conv2D(256, (1, 1), name='fpn_C1toC2')(C1)
    # C12_input = Activation('relu')(C12_input)
    #c2 out 55*55*256    C1 output 56*56*64
    '''new edition in 914'''
    pooledL1 = MaxPooling2D(pool_size=(2, 2))(C1)
    C2_inputfromC1conved = KL.Conv2D(64, kernel_size =(1, 1), strides=(1, 1), padding="same",name='fpn_C1toC2')(pooledL1)
    C2_output = KL.Add(name="fpn_C1addC2")([C2,C2_inputfromC1conved])
    # stage 3
    pooledL2 = MaxPooling2D(pool_size=(2, 2))(C2)
    C3_inputfromC2conved = KL.Conv2D(128, kernel_size =(1, 1), name='fpn_C2toC3',kernel_regularizer=L12_reg)(pooledL2)
    #c3 out 28*28*512
    C3_output = KL.Add(name="fpn_C2addC3")([C3,C3_inputfromC2conved])
    #stage 4
    pooledL3 = MaxPooling2D(pool_size=(2, 2))(C3)
    C4_inputfromC3conved = KL.Conv2D(256, kernel_size =(1, 1),  kernel_initializer="he_normal",name='fpn_C3toC4')(pooledL3)
    #c4 out 14*14*1024
    C4_output = KL.Add(name="fpn_C3addC4")([C4,C4_inputfromC3conved])
    # stage 5 C5 output 7*7*2048
    pooledL4 = MaxPooling2D(pool_size=(2, 2))(C4)
    C5_inputfromC4conved = KL.Conv2D(512, kernel_size =(1, 1),  kernel_initializer="he_normal",name='fpn_C4toC5')(pooledL4)
    #c5 out output 7*7*2048
    C5_output = KL.Add(name="fpn_C4addC5")([C5,C5_inputfromC4conved])

    gpC1 = GlobalAveragePooling2D(dim_ordering='default', name='global_pool1')(C1)
    gpC2 = GlobalAveragePooling2D(dim_ordering='default', name='global_pool2')(C2_output)
    gpC3 = GlobalAveragePooling2D(dim_ordering='default', name='global_pool3')(C3_output)
    gpC4 = GlobalAveragePooling2D(dim_ordering='default', name='global_pool4')(C4_output)

    gpC5 = GlobalAveragePooling2D(dim_ordering='default', name='global_pool5')(C5_output)
    gpC6 = GlobalAveragePooling2D(dim_ordering='default', name='global_pool6')(C5)
    gpCall= [gpC5,gpC6]
    featureall = KL.concatenate(gpCall)
    merge1 = Dense(64, activation='relu', name='Dense1')(featureall)
    merge1 = KL.Dropout(0.6,name = 'dropout')(merge1)
    # the gradient for main task if not allowed back propagate to the subtask
    # merge1 =  KL.concatenate([gpP1,featureall2])#if is multi-task, replace featureall2 by stop_grad
    output1 = KL.Dense(32,activation='relu', name='Dense2')(merge1)
    # output1 = KL.Dropout(0.5,name = 'dropout1')(output1)
    output1 = KL.Dense(num_outputs,activation='sigmoid', name='risk_pred')(output1)
    # input_shape = (224,224,3)
    # img_input = Input(shape=input_shape,name = 'input')
    model10 = Model(inputs = img_input, outputs =  output1, name='model50')
    return model10
#%%
if __name__ == '__main__':
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    from keras.utils import plot_model
    model50 = resnet_self(include_top=True)
    print(model50.summary())
    os.chdir('/data/zlw/sunyongqing1') 
    plot_model(model50,to_file='./FPNmultilevel823.pdf',show_shapes=True)
