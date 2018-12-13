from keras import layers
import keras.backend as K

def se_block(x,ratio):
    '''
    :param x:
    :param r: Fex中间层的隐藏节点数，论文中为16
    :return:
    '''
    squeeze=layers.GlobalAveragePooling2D()(x)
    #### 获取输入的shape,得到通道数
    shape=x._keras_shape[-1]
    excitation_1=layers.Dense(ratio,activation='relu')(squeeze)
    excitation_2=layers.Dense(shape,activation='sigmoid')(excitation_1)

    return layers.multiply([x,excitation_2])


