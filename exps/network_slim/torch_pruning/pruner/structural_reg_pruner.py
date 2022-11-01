from .. import dependency, functional, utils
from numbers import Number
from typing import Callable
from .basepruner import LocalPruner, GlobalPruner
import torch
import torch.nn as nn

class LocalStructrualRegularizedPruner(LocalPruner):
    def __init__(
        self,
        model,
        example_inputs,
        importance,
        total_steps=1,
        beta=1e-4,
        pruning_rate_scheduler: Callable = None,
        ch_sparsity=0.5,
        layer_ch_sparsity=None,
        round_to=None,
        ignored_layers=None,
        user_defined_parameters=None,
        output_transform=None,
    ):
        super(LocalStructrualRegularizedPruner, self).__init__(
            model=model,
            example_inputs=example_inputs,
            total_steps=total_steps,
            pruning_rate_scheduler=pruning_rate_scheduler,
            ch_sparsity=ch_sparsity,
            layer_ch_sparsity=layer_ch_sparsity,
            round_to=round_to,
            ignored_layers=ignored_layers,
            user_defined_parameters=user_defined_parameters,
            output_transform=output_transform,
        )
        self.importance = importance
        self.dropout_groups = {}
        self.beta = beta
        self.plans = self.get_all_plans()
    
    def estimate_importance(self, plan):
        return self.importance(plan)

    def structrual_dropout(self, module, input, output):
        return self.dropout_groups[module][0](output)

    def regularize(self, model):

        for plan in self.plans:
            for dep, idxs in plan:
                layer = dep.target.module
                prune_fn = dep.handler
                if prune_fn in [
                    functional.prune_conv_out_channel,
                    functional.prune_linear_out_channel,
                ]:
                    # regularize output channels
                    layer.weight.grad.data.add_(self.beta*torch.sign(layer.weight.data))
                elif prune_fn in [
                    functional.prune_conv_in_channel,
                    functional.prune_linear_in_channel,
                ]:
                    # regularize input channels
                    layer.weight.grad.data.add_(self.beta*torch.sign(layer.weight.data))
                elif prune_fn == functional.prune_batchnorm:
                    # regularize BN
                    if layer.affine is not None:
                        layer.weight.grad.data.add_(self.beta*torch.sign(layer.weight.data))
            

        