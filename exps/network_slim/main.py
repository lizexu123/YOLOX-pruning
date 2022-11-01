import torch
import sys
from torch_pruning import importance
import torch_pruning as tp
from yolox.models.yolox import YOLOX
model=YOLOX()
print(model)

#记录一下模型原来的size
ori_size=tp.utils.count_params(model)
example_inputs=torch.randn(1,3,224,224)
imp=tp.importance.MagnitudeImportance(p=1) #L1 norm pruning
ignored_layers=[]
for m in model.modules():
    if isinstance(m,torch.nn.Linear) and m.out_features==1000:
        ignored_layers.append(m)

total_step=1
pruner=tp.pruner.LocalMagnitudePruner(
    model,
    example_inputs,
    importance=imp,
    total_steps=total_step,#number of iterations
    ch_sparsity=0.5,#
    ignored_layers=ignored_layers,
)
for i in range(total_step):
    pruner.step()
    print(
        "Params:%.2f M=>%.2f M"
        %(ori_size/1e6,tp.utils.count_params(model)/1e6)
    )