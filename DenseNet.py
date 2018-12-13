from keras import layers
import keras.backend as K

def composition_function(input,growth_rate,densenet_b=False):
    if densenet_b:
        x=layers.BatchNormalization()(input)
        x=layers.ReLU()(x)
        input=layers.Conv2D(4*growth_rate,(1,1),padding='same')(x)

    x=layers.BatchNormalization()(input)
    x=layers.ReLU()(x)
    output=layers.Conv2D(growth_rate,(3,3),padding='same')(x)
    return output

def dense_block(input,depth,growth_rate,densenet_b=False):
    sorted_feature=input
    for i in range(depth):
        feature=composition_function(sorted_feature,growth_rate,densenet_b)
        sorted_feature=layers.concatenate([sorted_feature,feature],axis=-1)

    return sorted_feature

