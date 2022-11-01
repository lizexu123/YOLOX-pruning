from .dependency import TORCH_CONV, TORCH_BATCHNORM, TORCH_PRELU, TORCH_LINEAR
import torch, thop

def count_prunable_params_of_modules(module):
    if isinstance( module, ( TORCH_CONV, TORCH_LINEAR) ):
        num_params = module.weight.numel()
        if module.bias is not None:
            num_params += module.bias.numel()
        return num_params
    elif isinstance( module, TORCH_BATCHNORM ):
        num_params = module.running_mean.numel() + module.running_var.numel()
        if module.affine:
            num_params+= module.weight.numel() + module.bias.numel()
        return num_params
    elif isinstance( module, TORCH_PRELU ):
        if len( module.weight )==1:
            return 0
        else:
            return module.weight.numel
    else:
        return 0

def count_prunable_in_channels(module):
    if isinstance( module, TORCH_CONV ):
        return module.weight.shape[1]
    elif isinstance( module, TORCH_LINEAR ):
        return module.in_features
    elif isinstance( module, TORCH_BATCHNORM ):
        return module.num_features
    elif isinstance( module, TORCH_PRELU ):
        if len( module.weight )==1:
            return 0
        else:
            return len(module.weight)
    else:
        return 0

def count_prunable_out_channels(module):
    if isinstance( module, TORCH_CONV ):
        return module.weight.shape[0]
    elif isinstance( module, TORCH_LINEAR ):
        return module.out_features
    elif isinstance( module, TORCH_BATCHNORM ):
        return module.num_features
    elif isinstance( module, TORCH_PRELU ):
        if len( module.weight )==1:
            return 0
        else:
            return len(module.weight)
    else:
        return 0

def count_params(module):
    return sum([ p.numel() for p in module.parameters() ])

def count_macs_and_params(model, input_size, example_inputs=None):
    if example_inputs is None:
        example_inputs = torch.randn(*input_size)
    macs, params = thop.profile(model, inputs=(example_inputs, ), verbose=False)
    return macs, params

def count_total_prunable_channels(model):
    in_ch = 0
    out_ch = 0
    for m in model.modules():
        out_ch+=count_prunable_out_channels(m)
        in_ch+= count_prunable_in_channels(m)
    return out_ch, in_ch
    