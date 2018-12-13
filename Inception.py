import numpy as np
import tensorflow as tf

from keras import layers
from keras import models
import keras.backend as K

def inception_v1(x,n_1x1,n_3x3_pre,n_3x3,n_5x5_pre,n_5x5,n_maxpool):
    '''
    :param x:输入
    :param n:feature map 数
    :return:
    '''
    inception_1x1=layers.Conv2D(n_1x1,(1,1),padding='same',activation='relu')(x)
    inception_3x3pre=layers.Conv2D(n_3x3_pre,(1,1),padding='same',activation='relu')(x)
    inception_3x3=layers.Conv2D(n_3x3,(3,3),padding='same',activation='relu')(inception_3x3pre)
    inception_5x5pre=layers.Conv2D(n_5x5_pre,(1,1),padding='same',activation='relu')(x)
    inception_5x5=layers.Conv2D(n_5x5,(5,5),padding='same',activation='relu')(inception_5x5pre)
    inception_maxpool=layers.MaxPool2D((3,3),strides=(1,1),padding='same')(x)
    inception_maxpool_after=layers.Conv2D(n_maxpool,(1,1),padding="same",activation='relu')(inception_maxpool)
    output=layers.concatenate([inception_1x1,inception_3x3,inception_5x5,inception_maxpool_after],axis=3)

    return output

def inception_v2(x,n_1x1,n_3x3_pre,n_3x3,n_3x3_pre_2,n_3x3_1,n_3x3_2,n_maxpool):
    '''
    将5x5卷积换为两个3x3卷积
    '''
    inception_1x1 = layers.Conv2D(n_1x1, (1, 1), padding='same', activation='relu')(x)
    inception_3x3pre = layers.Conv2D(n_3x3_pre, (1, 1), padding='same', activation='relu')(x)
    inception_3x3 = layers.Conv2D(n_3x3, (3, 3), padding='same', activation='relu')(inception_3x3pre)
    inception_5x5_1pre = layers.Conv2D(n_3x3_pre_2, (1, 1), padding='same', activation='relu')(x)
    inception_5x5_1= layers.Conv2D(n_3x3_1, (3, 3), padding='same', activation='relu')(inception_5x5_1pre)
    inception_5x5_2 = layers.Conv2D(n_3x3_2, (3, 3), padding='same', activation='relu')(inception_5x5_1)
    inception_maxpool = layers.MaxPool2D((3, 3), strides=(1, 1), padding='same')(x)
    inception_maxpool_after = layers.Conv2D(n_maxpool, (1, 1), padding="same", activation='relu')(inception_maxpool)
    output = layers.concatenate([inception_1x1,inception_3x3,inception_5x5_2,inception_maxpool_after], axis=3)

    return output

def inception_v3(x,n_1x1,n_3x3_pre,n_3x3,n_3x3_pre_2,n_3x3_1,n_3x3_2,n_maxpool):
    '''
    将5x5卷积换为两个3x3卷积
    '''
    inception_1x1 = layers.Conv2D(n_1x1, (1, 1), padding='same', activation='relu')(x)
    inception_3x3pre = layers.Conv2D(n_3x3_pre, (1, 1), padding='same', activation='relu')(x)
    inception_3x3 = layers.Conv2D(n_3x3, (3, 3), padding='same', activation='relu')(inception_3x3pre)
    inception_5x5_1pre = layers.Conv2D(n_3x3_pre_2, (1, 1), padding='same', activation='relu')(x)
    inception_5x5_1 = layers.Conv2D(n_3x3_1, (3, 3), padding='same', activation='relu')(inception_5x5_1pre)
    inception_5x5_2 = layers.Conv2D(n_3x3_2, (3, 3), padding='same', activation='relu')(inception_5x5_1)
    inception_maxpool = layers.MaxPool2D((3, 3), strides=(1, 1), padding='same')(x)
    inception_maxpool_after = layers.Conv2D(n_maxpool, (1, 1), padding="same", activation='relu')(inception_maxpool)
    output = layers.concatenate([inception_1x1, inception_3x3, inception_5x5_2, inception_maxpool_after], axis=-1)

    return output



