import argparse
from tools import *
import open3d.ml.torch as ml3d
import pathlib as Path
import torch
import shutil
import logging
from termcolor import colored

def checkpoint_state(model, epoch):
    

def save_checkpoint(state, filename='checkpoint'):
    filename = '{}.pth'.format(filename)
    torch.save(state, filename)