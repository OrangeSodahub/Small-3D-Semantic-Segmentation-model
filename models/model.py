import open3d.ml as ml3d
from abc import *
import torch

class Model(ABC):

    def __init__(self):
        

    def loss(self):
        loss = torch.nn.CrossEntropyLoss()
        return loss