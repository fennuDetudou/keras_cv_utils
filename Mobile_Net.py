from keras import layers
import keras.backend as K
from keras.activations import relu
from keras.utils.generic_utils import CustomObjectScope

def relu6(x):
    return relu(x,max_value=6)

def mobile_net_v1_block(x,output_filters):
    x=layers.DepthwiseConv2D(kernel_size=(3,3),padding='same')(x)
    x=layers.Activation(relu6)(x)
    x=layers.Conv2D(filters=output_filters,kernel_size=(1,1),padding='same')(x)
    output=layers.Activation(relu6)(x)

    return output

def mobile_net_v2_block(inputs,output_filters,t):
    x=layers.Conv2D(output_filters*t,kernel_size=(1,1),padding='same')(inputs)
    x=layers.Activation(relu6)(x)
    x=layers.DepthwiseConv2D((3,3),padding='same',activation='relu')(x)
    #### no activation
    x=layers.Conv2D(output_filters,kernel_size=(1,1),padding='same')(x)

    if not K.get_variable_shape(inputs)[3]==output_filters:
        inputs=layers.Conv2D(output_filters,(1,1),padding='same')(inputs)
    output=layers.Add()([inputs,x])

    return output
