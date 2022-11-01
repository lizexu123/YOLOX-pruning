import abc
import this
import torch
import torch.nn as nn
from . import functional
import random

class Importance:
    pass

def rescale(x):
    return (x - x.min()) / (x.max() - x.min())

#importance_mat=[]
'''
一个类实例可以变成一个可调用对象，只需要实现一个特殊方法__call__()
'''

class MagnitudeImportance(Importance):
    def __init__(self, p=1, local=False, reduction="mean"):
        self.p = p
        self.local = local
        self.reduction = reduction

    @torch.no_grad()
    def __call__(self, plan):
        importance_mat = []
        non_importance = True
        #plan就是Pruning plan回打印出来，我这个模型可以剪枝哪些层
        #print('plan',plan) 
        '''
        [DEP] BatchnormPruner on layer4.1.bn1 (BatchNorm2d(307, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)) => ElementWiseOpPruner on _ElementWiseOp(ReluBackward1)
idxs [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306]
        
        '''
        for dep, idxs in plan:      
            # print('dep',dep)
            # print('idxs',idxs)
            layer = dep.target.module
            # print('layer',layer) #layer会打印每一层
            #layer为 比如Conv2d(3,64,kernel_size=(7,7),stride=(2,2),padding=(3,3),bias=False)
            prune_fn = dep.handler
            #print('prune_fn',prune_fn)
            #如果prune_fn属于下面这其中一种
            if prune_fn in [
                functional.prune_conv_out_channel,
                functional.prune_linear_out_channel,
            ]:
                w = (layer.weight)[idxs]
                #torch.flatten(w,1)在第1维度将w展平
                #torch.norm(input,p)是对输入张量求范数
                this_importance = torch.norm(torch.flatten(w, 1), dim=1, p=self.p)
                importance_mat.append(this_importance)
                non_importance = False
            elif prune_fn in [
                functional.prune_conv_in_channel,
                functional.prune_linear_in_channel,
            ]:
                w = (layer.weight)[:, idxs].transpose(0, 1)
                w = torch.flatten(w, 1)
                if w.shape[0] != importance_mat[0].shape[0]:  # for conv-flatten-linear
                    if (
                        w.shape[0] % importance_mat[0].shape[0] != 0
                    ):  # TODO: support Group Convs
                        continue
                    w = w.view(
                        importance_mat[0].shape[0],
                        w.shape[0] // importance_mat[0].shape[0],
                        w.shape[1],
                    )
                #torch.norm做范数运算
                this_importance = torch.norm(torch.flatten(w, 1), dim=1, p=self.p)
                #print('this_importance',this_importance)
                importance_mat.append(this_importance)
                #print('importance_mat',importance_mat)
                non_importance = False
            elif prune_fn == functional.prune_batchnorm:
                if layer.affine is not None:
                    w = (layer.weight)[idxs].view(-1, 1)
                    this_importance = torch.norm(w, dim=1, p=self.p)
                    importance_mat.append(this_importance)
            if self.local:
                break
        importance_mat = torch.stack(importance_mat, dim=0)
        if non_importance:
            return None
        if self.reduction == "sum":
            return importance_mat.sum(dim=0)
        elif self.reduction == "mean":
            return importance_mat.mean(dim=0)
        elif self.reduction == "max":
            return importance_mat.max(dim=0)[0]
        elif self.reduction == "min":
            return importance_mat.min(dim=0)[0]
        elif self.reduction == "prod":
            return importance_mat.prod(dim=0)


class RandomImportance(Importance):

    @torch.no_grad()
    def __call__(self, plan):
        _, idxs = plan[0]
        return torch.randn( (len(idxs), ) )


class SensitivityImportance(Importance):
    def __init__(self, local=False, reduction="mean") -> None:
        self.local = local
        self.reduction = reduction

    def __call__(self, loss, plan):
        loss.backward()
        with torch.no_grad():
            importance = 0
            n_layers = 0
            for dep, idxs in plan:
                layer = dep.target.module
                prune_fn = dep.handler
                n_layers += 1
                if prune_fn in [
                    functional.prune_conv_out_channel,
                    functional.prune_linear_in_channel,
                ]:
                    w_dw = (layer.weight * layer.weight.grad)[idxs]
                    importance += torch.norm(torch.flatten(w_dw, 1), dim=1)
                    if layer.bias:
                        w_dw = (layer.bias * layer.bias.grad)[idxs].view(-1, 1)
                        importance += torch.norm(w_dw, dim=1)
                elif prune_fn in [
                    functional.prune_conv_in_channel,
                    functional.prune_linear_in_channel,
                ]:
                    w_dw = (layer.weight * layer.weight.grad)[:, idxs].transpose(0, 1)
                    importance += torch.norm(torch.flatten(w_dw, 1), dim=1)
                elif prune_fn == functional.prune_batchnorm:
                    if layer.affine:
                        w_dw = (layer.weight * layer.weight.grad)[idxs].view(-1, 1)
                        importance += torch.norm(w_dw, dim=1)
                        w_dw = (layer.bias * layer.bias.grad)[idxs].view(-1, 1)
                        importance += torch.norm(w_dw, dim=1)
                else:
                    n_layers -= 1

                if self.local:
                    break
            if self.reduction == "sum":
                return importance
            elif self.reduction == "mean":
                return importance / n_layers
            


class HessianImportance(Importance):
    def __init__(self) -> None:
        pass

class BNScaleImportance(Importance):
    def __init__(self, group_level=False, reduction='mean'):
        self.group_level = group_level
        self.reduction = reduction

    def __call__(self, plan):
        importance_mat = []

        for dep, idxs in plan:
            # Conv-BN
            module = dep.target.module
            if isinstance(module, nn.BatchNorm2d) and module.affine:
                imp = torch.abs(module.weight.data)
                importance_mat.append( imp )
                if not self.group_level:
                    return imp
        importance_mat = torch.stack(importance_mat, dim=0)
        if self.reduction == "sum":
            return importance_mat.sum(dim=0)
        elif self.reduction == "mean":
            return importance_mat.mean(dim=0)
        elif self.reduction == "max":
            return importance_mat.max(dim=0)[0]
        elif self.reduction == "min":
            return importance_mat.min(dim=0)[0]


        
class StrcuturalImportance(Importance):
    def __init__(self, p=1, local=False, reduction="mean"):
        self.p = p
        self.local = local
        self.reduction = reduction

    @torch.no_grad()
    def __call__(self, plan):
        importance_mat = []
        non_importance = True
        for dep, idxs in plan:
            layer = dep.target.module
            prune_fn = dep.handler
            if prune_fn in [
                functional.prune_conv_out_channel,
                functional.prune_linear_out_channel,
            ]:
                w = (layer.weight)[idxs]
                this_importance = torch.norm(torch.flatten(w, 1), dim=1, p=self.p)
                importance_mat.append(rescale(this_importance))
                non_importance = False
            elif prune_fn in [
                functional.prune_conv_in_channel,
                functional.prune_linear_in_channel,
            ]:
                w = (layer.weight)[:, idxs].transpose(0, 1)
                w = torch.flatten(w, 1)
                if w.shape[0] != importance_mat[0].shape[0]:  # for conv-flatten-linear
                    if (
                        w.shape[0] % importance_mat[0].shape[0] != 0
                    ):  # TODO: support Group Convs
                        continue
                    w = w.view(
                        importance_mat[0].shape[0],
                        w.shape[0] // importance_mat[0].shape[0],
                        w.shape[1],
                    )
                this_importance = torch.norm(torch.flatten(w, 1), dim=1, p=self.p)
                importance_mat.append(rescale(this_importance))
                non_importance = False
            elif prune_fn == functional.prune_batchnorm:
                if layer.affine is not None:
                    w = (layer.weight)[idxs].view(-1, 1)
                    this_importance = torch.norm(w, dim=1, p=self.p)
                    importance_mat.append(rescale(this_importance))
            if self.local:
                break
        importance_mat = torch.stack(importance_mat, dim=0)
        if non_importance:
            return None
        if self.reduction == "sum":
            return importance_mat.sum(dim=0)
        elif self.reduction == "mean":
            return importance_mat.mean(dim=0)
        elif self.reduction == "max":
            return importance_mat.max(dim=0)[0]
        elif self.reduction == "min":
            return importance_mat.min(dim=0)[0]
        elif self.reduction == "prod":
            return importance_mat.prod(dim=0)


class LAMPImportance(Importance):
    def __init__(self, p=2, local=False, reduction="mean"):
        self.p = p
        self.local = local
        self.reduction = reduction

    @torch.no_grad()
    def __call__(self, plan):
        importance = 0
        n_layers = 0
        non_importance = True
        for dep, idxs in plan:
            layer = dep.target.module
            prune_fn = dep.handler
            n_layers += 1
            if prune_fn in [
                functional.prune_conv_out_channel,
                functional.prune_linear_out_channel,
            ]:
                w = (layer.weight)[idxs]
                this_importance = torch.norm(torch.flatten(w, 1), dim=1, p=self.p)
                this_importance = rescale(this_importance)
                importance+=this_importance
                #if layer.bias is not None:
                #    w = (layer.bias)[idxs].view(-1, 1)
                #    importance += torch.norm(w, dim=1, p=self.p)
                non_importance = False
            elif prune_fn in [
                functional.prune_conv_in_channel,
                functional.prune_linear_in_channel,
            ]:
                w = (layer.weight)[:, idxs].transpose(0, 1)
                w = torch.flatten(w, 1)
                if w.shape[0] != importance.shape[0]:  # for conv-flatten-linear
                    if (
                        w.shape[0] % importance.shape[0] != 0
                    ):  # TODO: support Group Convs
                        continue
                    w = w.view(
                        importance.shape[0],
                        w.shape[0] // importance.shape[0],
                        w.shape[1],
                    )
                this_importance = torch.norm(torch.flatten(w, 1), dim=1, p=self.p)
                this_importance = rescale(this_importance)
                importance += this_importance
                non_importance = False
            elif prune_fn == functional.prune_batchnorm:
                continue
                if layer.affine is not None:
                    #scale = layer.weight / sqrt_rv
                    #bias = layer.bias - rm / sqrt_rv * layer.weight
                    w = (layer.weight)[idxs].view(-1, 1)
                    importance += rescale(torch.norm(w, dim=1, p=self.p))
                    #w = (bias)[idxs].view(-1, 1)
                    #importance *= torch.norm(w, dim=1, p=self.p)
            #        non_importance = False
            else:
                n_layers -= 1
            if self.local:
                break
        argsort_idx = torch.argsort(importance).tolist()[::-1] # [7, 5, 2, 3, 1, ...]
        sorted_importance = importance[argsort_idx]
        cumsum_importance = torch.cumsum(sorted_importance, dim=0 )
        sorted_importance = sorted_importance / cumsum_importance 
        inversed_idx = torch.arange(len(sorted_importance))[argsort_idx].tolist() # [0, 1, 2, 3, ..., ]
        importance = sorted_importance[inversed_idx]
        if non_importance:
            return None
        if self.reduction == "sum":
            return importance
        elif self.reduction == "mean":
            return importance / n_layers
