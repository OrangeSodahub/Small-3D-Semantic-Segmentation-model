from pyexpat import model
import torch
import open3d.ml.torch as ml3d
from abc import *

def build(model_config):
    # Number classes/categories
    num_class = model_config['num_class']
    # Number of input features
    num_input_features = model_config['num_input_features']

    # Feature dimension
    fea_dim = model_config['fea_dim']
    # Output feature dimension
    out_fea_dim = model_config['out_fea_dim']

    in_channels = model_config['in_channels']
    filters = model.config['filters']

    # SparseConv Network
    spconv = ml3d.layers.SparseConv(in_channels, filters, kernel_size=[3,3,3])

    # Loss function
    loss_func = torch.nn.CrossEntropyLoss()

    model = get_model_class(model_config['name'])

    return model

REGISTERED_MODELS_CLASSES = {}

def register_model(cls, name=None):
    global REGISTERED_MODELS_CLASSES
    if name is None:
        name = cls.__name__
    assert name not in REGISTERED_MODELS_CLASSES, f"exist class: {REGISTERED_MODELS_CLASSES}"
    REGISTERED_MODELS_CLASSES[name] = cls
    return cls


def get_model_class(name):
    global REGISTERED_MODELS_CLASSES
    assert name in REGISTERED_MODELS_CLASSES, f"available class: {REGISTERED_MODELS_CLASSES}"
    return REGISTERED_MODELS_CLASSES[name]