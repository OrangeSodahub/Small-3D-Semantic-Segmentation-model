import open3d as o3d
import open3d.ml.torch as ml3d
import numpy as np
import os
from os.path import exists, join, dirname

def get_raw_data(pc_names, path):

    pc_data = []
    for i, name in enumerate(pc_names):
        # both points and labels in data_path 
        data_path = join(path, name + '.npy')
        point = np.load(data_path)[:, 0:3]
        label = np.squeeze(np.load(data_path)[:,6])

        data = {
            'point': point,
            'feat': None,
            'label': label,
        }
        pc_data.append(data)

    return pc_data

def main():
    vis = ml3d.vis.Visualizer()
    # set labels
    lut = ml3d.vis.LabelLUT()
    vis.set_lut("label", lut)

    data_path = '/home/zonlin/Open3D_assignment/dataset/raw_data/'
    pc_names = ['data']
    raw_data = get_raw_data(pc_names, data_path)

    vis.visualize(raw_data)

main()