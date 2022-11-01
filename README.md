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
（1）将yolox/data/datasets/voc_classes.py中的标签信息，进行修改，换成你自己的voc数据集
![在这里插入图片描述](https://img-blog.csdnimg.cn/b68598b89573460eb69e408f09efdce7.png)
(2) 修改类别数量
修改exps/example/yolox_voc/yolox_voc_s.py中的self.num_classes
![在这里插入图片描述](https://img-blog.csdnimg.cn/ea737458b30546dd82e7cbbbb29e6d53.png)
这里我用的s模型，类别有16个
(3)、修改训练集信息
修改exps/example/yolox_voc/yolox_voc_s.py中的VOCDection
![在这里插入图片描述](https://img-blog.csdnimg.cn/61b3170702cc4df89c6c6ecc17f6af53.png)
data_dir可以写你VOC数据集的绝对路径，
images_sets修改为
![在这里插入图片描述](https://img-blog.csdnimg.cn/28bdfe01b04542ba8aa57cf7298f8374.png)
修改yolox/data/datasets/voc.py中，VOCDection函数中的读取txt文件
![在这里插入图片描述](https://img-blog.csdnimg.cn/cc234f31cbe440dbb9775a084035352d.png)
我这里是因为![在这里插入图片描述](https://img-blog.csdnimg.cn/da07f1f0181d49db9514eb1774161bd1.png)
所以image_sets为2007 trainval
（4）修改exps/example/yolox_voc/yolox_voc_s.py中的get_eval_loader函数
![在这里插入图片描述](https://img-blog.csdnimg.cn/838c5977bfa847dfb82d50f01cfc646a.png)
(5）修改不同的网络结构
以YOLOX_S网络为例，比如在exps/default/yolox_s.py中，self.depth=0.33,self.width=0.50.和YOLOX中的不同网络调度方式一样。
![在这里插入图片描述](https://img-blog.csdnimg.cn/ecac74545d9c4d81937c0cc035713e91.png)
再修改yolox/exp/yolox_base.py中的,self_depth和self.width。
(6)修改其他参数
训练100个epoch.因为服务器只有一个2080ti
![在这里插入图片描述](https://img-blog.csdnimg.cn/8390629322a449859d3f1f7997da3f46.png)
## YOLOX训练
python tools/train.py -d 1 -b 1 -f exps/example/yolox_voc/yolox_voc_s.py
![在这里插入图片描述](https://img-blog.csdnimg.cn/196b115b7b5a4d4890fa2e65839af449.png)
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
![在这里插入图片描述](https://img-blog.csdnimg.cn/44799acec7d145b69c8f0439a64a571f.jpeg)
剪枝以后，会打印模型的参数量变化
### 支持模型整体剪枝
![在这里插入图片描述](https://img-blog.csdnimg.cn/25ea31433e954189b70efcfe9e7c03b9.png)
这里的ch_sparsity为剪枝率。
![在这里插入图片描述](https://img-blog.csdnimg.cn/c93e6948ffb84c35b6c5e98caa341ff5.png)
这是剪枝后成功的表现。
## 剪枝后的微调训练
在tools/train.py中加入
![在这里插入图片描述](https://img-blog.csdnimg.cn/41ad9c89e4b54b0f9bf455c50c22abd1.png)
然后在yolox/core/trainer.py中
![在这里插入图片描述](https://img-blog.csdnimg.cn/3e9419f2cc27428c9c7eeeca86fe3b90.png)
这里的model_path换成自己剪枝后保存下来的pth文件就可以。




















