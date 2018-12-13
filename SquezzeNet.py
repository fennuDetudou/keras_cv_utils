from keras import layers

def fire_block(x,s_11,e_11,e_33):
    '''
     论文建议 s_11<e_11+e_33
    :param s_11:squeeze filter number
    :param e_11: expand 1x1 filter number
    :param e_33: expand 3x3 filter number
    :return: concat expands
    '''
    squeeze=layers.Conv2D(filters=s_11,kernel_size=(1,1),padding='same',activation='relu')(x)
    expand_1=layers.Conv2D(e_11,kernel_size=(1,1),padding='same',activation='relu')(squeeze)
    expand_2=layers.Conv2D(e_33,kernel_size=(3,3),padding='same',activation='relu')(squeeze)
    output=layers.Concatenate(axis=3)([expand_1,expand_2])

    return output

