import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

def init_model(net):
    """Init models with cuda and weights."""
    # init weights of model
    net.apply(init_weights)
    return net

def init_weights(layer):
    """Init weights for layers w.r.t. the original paper."""
    layer_name = layer.__class__.__name__
    if layer_name.find("Conv") != -1:
        layer.weight.data.normal_(0.0, 0.02)
    elif layer_name.find("BatchNorm") != -1:
        layer.weight.data.normal_(1.0, 0.02)
        layer.bias.data.fill_(0)