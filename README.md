# keras_cv_utils

## 功能1：对常见网络模块以Python包的形式进行封装

### 使用示例

#### 搭建神经网络模型

```python
from keras import models
from keras import layers
# 从包中导入所需的模块
from Mobile_Net import mobile_net_v1_modeul,mobile_net_v2_module
from SENet import se_block
from DenseNet import dense_block
from Inception import inception_v3
from SquezzeNet import fire_module
from ResNet import Resnet_v1

inputs=layers.Input(shape=(10,10,10))

x=mobile_net_v1_modeul(inputs,10)
x=mobile_net_v2_module(x,10,1)
x=se_block(x,16)
x=dense_block(x,4,4)
x=inception_v3(x,4,4,4,4,4,4,4)
x=fire_module(x,3,4,4)
x=Resnet_v1(x,10)

model=models.Model(inputs=inputs,outputs=x)
```

最终得到的模型结构如下：

![](https://github.com/fennuDetudou/keras_cv_utils/blob/master/model.png?raw=true)

