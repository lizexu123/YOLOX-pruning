安装环境 
https://github.com/Megvii-BaseDetection/YOLOX.git
这里把YOLOX的代码下载到本地，并根据其安装好环境，这里就不再赘述了。
安装包
pip install torch_pruning==0.2.7
本文实现的功能
1、支持单个卷积剪枝
2、支持网络层剪枝
3、支持模型的整体剪枝
4、剪枝后微调训练
5、修改了激活函数 silu->mish(因为项目用)
数据集格式：采用VOC数据集格式

网络剪枝
参考论文:Pruning Filters for Efficient ConvNets
首先进行模型的正常训练
