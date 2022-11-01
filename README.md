# YOLOX剪枝(官网的)
本文主要是根据csdn 爱吃肉的鹏，他的pytorch-yolox剪枝代码所改，非常感谢大佬的帮助，正好是我所需要的，但是我发现B站大佬这个代码训练出来的精度会比官网低10%的map 0.5，在我的Dota-1.5数据集，有一个类总是AP=0，我感觉应该是B站大佬的代码写的不完整，导致推理的精度低，所以我直接用的YOLOX官网，与大佬的版本相结合。又加了一个YOLOX的整体剪枝。
## 环境
https://github.com/Megvii-BaseDetection/YOLOX.git
这里把YOLOX的代码下载到本地，并根据其安装好环境，这里就不再赘述了。
## 安装包
pip install torch_pruning==0.2.7
## 本文实现的功能
### 1、支持单个卷积剪枝
### 2、支持网络层剪枝
### 3、支持模型的整体剪枝
### 4、剪枝后微调训练
### 5、修改了激活函数 silu->mish(因为项目用)
数据集格式：采用VOC数据集格式
## 网络剪枝
参考论文:Pruning Filters for Efficient ConvNets
首先进行模型的正常训练
#### 1、修改代码
## 网络剪枝
参考论文:Pruning Filters for Efficient ConvNets
在剪枝之前需要通过tools/prunmodel.py  save_whole_model(weights_path,num_classes)函数将模型的权重和结构都保存下来
weights_path:权重路径
num_classes:自己数据集的类别数
### 支持对某个卷积的剪枝  调用Conv_pruning(whole_model_weights):
pruning_idxs=strategy(v,amout=0.4) #0.4是剪枝率，根据需要自己修改，数越大剪枝的越多
对于单独一个卷积的剪枝，需要修改两个地方，这里的卷积层需要打印模型获得，不要自己盲目瞎猜:
![在这里插入图片描述](https://img-blog.csdnimg.cn/147d6237d8324e34b10315848bb62680.png)
支持网络层的剪枝:调用layer_pruning(whole_model_weights):
![在这里插入图片描述](https://img-blog.csdnimg.cn/860153e31a6e41aa8268603b9986415f.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/3d4cdffb923043288ded1d6d2d260032.png)

剪枝以后，会打印模型的参数量变化
### 支持模型整体剪枝!
python exps/network_slim/main.py
这里面的total_step=1表示只进行一轮剪枝
ch_sparsity=0.5表示剪枝率
我测试了yolox-s
![这是剪枝后成功的表现。](https://img-blog.csdnimg.cn/620993fb09e0471e87e383ed3d26bd3c.png)


## 剪枝后的微调训练
在tools/train.py中加入
![在这里插入图片描述](https://img-blog.csdnimg.cn/41ad9c89e4b54b0f9bf455c50c22abd1.png)
然后在yolox/core/trainer.py中
![在这里插入图片描述](https://img-blog.csdnimg.cn/3e9419f2cc27428c9c7eeeca86fe3b90.png)
这里的model_path换成自己剪枝后保存下来的pth文件就可以。

