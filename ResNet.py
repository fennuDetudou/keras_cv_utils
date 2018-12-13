import numpy as np

from keras import layers
import keras.backend as K

def Resnet_v1(x,output_filters):
    input_filters=x._keras_shape[-1]
    res_x=layers.Conv2D(filters=output_filters,kernel_size=(3,3),padding='same')(x)
    res_x=layers.BatchNormalization()(res_x)
    res_x=layers.ReLU()(res_x)
    res_x=layers.Conv2D(filters=output_filters,kernel_size=(3,3),padding='same')(res_x)
    res_x=layers.BatchNormalization()(res_x)

    if input_filters == output_filters:
        pass
    else:
        x=layers.Conv2D(filters=output_filters,kernel_size=(1,1),padding='same')(x)

    output=layers.add([x,res_x])
    output=layers.ReLU()(output)

    return output


def Resnet_v2(x,output_filters):
    input_filters=x._keras_shape[-1]
    res_x = layers.BatchNormalization()(x)
    res_x = layers.ReLU()(res_x)
    res_x = layers.Conv2D(filters=output_filters, kernel_size=(3, 3), padding='same')(res_x)
    res_x = layers.BatchNormalization()(res_x)
    res_x = layers.ReLU()(res_x)
    res_x = layers.Conv2D(filters=output_filters, kernel_size=(3, 3), padding='same')(res_x)

    if input_filters == output_filters:
        pass
    else:
        x = layers.Conv2D(filters=output_filters, kernel_size=(1, 1), padding='same')(x)

    output = layers.add([x, res_x])

    return output

