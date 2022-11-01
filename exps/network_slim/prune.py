import torch
import torch.nn as nn
import torch_pruning as tp
from yolox.models.yolox import YOLOX
from loguru import logger

'''
剪枝的时候根据模型结构去剪，不要盲目的猜
剪枝完需要进行一个微调训练
'''
def save_whole_model(weights_path,num_classes):
    model=YOLOX()
    model_dict=model.state_dict()
    pretrained_dict=torch.load(weights_path)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys() == pretrained_dict.keys()}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.eval()
    torch.save(model,'/home/xcn/new-data/lzx/YOLOX/model_data/whole_model.pth')
    print("保存完成\n")
#当遇到错误时,如果在打印出log的时候没有配置Traceback的输出，很有可能无法追踪错误。loguru提供了装饰器@logger.catch()就可以直接进行Traceback的记录。
@logger.catch
def Conv_pruning(whole_model_weights):
    logger.add('../logs/Conv_pruning.log',rotation='1MB')
    model=torch.load(whole_model_weights)#模型的加载
    model_dict=model.state_dict()#获取模型的字典
    #-----------------------特定卷积的剪枝---------------------------------#
    #                       比如要剪枝以下卷积
    #                      backbone.conv1.conv.weight
    #-----------------------------------------------------------------------#
    for k,v in model_dict.items():
        if k == 'backbone.backbone.dark2.0.conv.weight': 
            #1.setup strategy(L1 Norm) 设置策略(L1 Norm)
            strategy=tp.strategy.L1Strategy()
            #2. build layer dependency 构建层依赖关系
            DG=tp.DependencyGraph()
            DG.build_dependency(model,example_inputs=torch.randn(1,3,640,640))
            num_params_before_pruning=tp.utils.count_params(model)
            #3.get a pruning plan from the dependency graph 
            pruning_idx=strategy(v,amount=0.4)
            pruning_plan=DG.get_pruning_plan((model.backbone.backbone.dark2)[0].conv,tp.prune_conv,idxs=pruning_idx)
            logger.info(pruning_plan)
            #4. execute this plan(prune the model)
            pruning_plan.exec()
            #获得剪枝以后的参数量
            num_params_after_pruning=tp.utils.count_params(model)
            logger.info("Params:%s=>%s\n"%(num_params_before_pruning,num_params_after_pruning))
    torch.save(model,'/home/xcn/new-data/lzx/YOLOX/model_data/Conv_pruning.pth')
    logger.info("剪枝完成\n")
@logger.catch
def layer_pruning(whole_model_weights):
    logger.add("../logs/layer_pruning.log",rotation='1MB')
    model=torch.load(whole_model_weights)#模型的加载
    x=torch.rand(1,3,640,640)
    #-----------------对整个模型的剪枝--------------------#
    strategy=tp.strategy.L1Strategy()
    DG=tp.DependencyGraph()
    DG=DG.build_dependency(model,example_inputs=x)
    num_params_before_pruning=tp.utils.count_params(model)
    #可以对照yolox结构进行剪枝
    included_layers=list((model.backbone.backbone.dark2.modules()))#对主干进行剪枝
    for m in model.modules():
        if isinstance(m,nn.Conv2d) and m in included_layers:
            pruning_plan=DG.get_pruning_plan(m,tp.prune_conv,idxs=strategy(m.weight,amount=0.4))
            logger.info(pruning_plan)
            #执行剪枝
            pruning_plan.exec()
    #获取剪枝以后的参数量
    num_params_after_pruning=tp.utils.count_params(model)
    #输出一下剪枝前后的参数量
    logger.info("Params:%s=>%s\n"%(num_params_before_pruning,num_params_after_pruning))
    torch.save(model,'/home/xcn/new-data/lzx/YOLOX/model_data/layer_pruning.pth')
    logger.info("剪枝完成\n")


# model=torch.load("/home/xcn/new-data/lzx/YOLOX-Slim/weights/yolox_s.pth")
# print('model',model)
# model_dict=model.load_state_dict(model)
# for k,v in model_dict.items():
#     print(k)
# print(model)
# layer=nn.ModuleList(m for m in model.backbone.backbone.dark2.modules())
# layer_pruning("/home/xcn/new-data/lzx/YOLOX-Slim/weights/yolox_s.pth")
# layer_pruning('/home/xcn/new-data/lzx/YOLOX/weights/model.pth')
# weights_path='/home/xcn/new-data/lzx/YOLOX/weights/yolox_s.pth'
# num_classes=16
# save_whole_model(weights_path,num_classes)
Conv_pruning('/home/xcn/new-data/lzx/YOLOX/model_data/whole_model.pth')
