# keras_cv_utils

##功能1：对常见网络模块以Python包的形式进行封装

###包含的结构

1. GoogLeNet：inception_v1、inception_v2、inception_v3
1. ResNet ：resnet_v1、resnet_v2
1. Squeeze-and-Excitation Networks: se_block
1. DenseNet : dense_block
1. SqueezeNet : fire_block
1. MobileNet : mobile_net_v1_block、mobile_net_v2_block

#### 使用说明

1. 追求更高的精度，则选用1、2、3、4 网络模块
1. 追求更快的速度，则选用5、6网络模块

### 使用示例

#### 搭建神经网络模型

```python
from keras import models
from keras import layers
# 从包中导入所需的模块
from Mobile_Net import mobile_net_v1_block,mobile_net_v2_block
from SENet import se_block
from DenseNet import dense_block
from Inception import inception_v3
from SquezzeNet import fire_block
from ResNet import Resnet_v1

inputs=layers.Input(shape=(10,10,10))

x=mobile_net_v1_block(inputs,10)
x=mobile_net_v2_block(x,10,1)
x=se_block(x,16)
x=dense_block(x,4,4)
x=inception_v3(x,4,4,4,4,4,4,4)
x=fire_block(x,3,4,4)
x=Resnet_v1(x,10)

model=models.Model(inputs=inputs,outputs=x)
```

最终得到的模型结构如下：

![](https://github.com/fennuDetudou/keras_cv_utils/blob/master/model.png?raw=true)

