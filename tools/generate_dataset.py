import argparse
from curses import meta
import numpy as np
import open3d as o3d
import tools
# from tools.tools import *
import shutil
import logging

np.set_printoptions(suppress=True)

def main(data_dir: str, log_dir: str, visible: bool):

    train_data = np.load(data_dir)
    np.savetxt('/home/zonlin/Open3D_assignment/dataset/train_data.txt',train_data)
    # delete the label and convert rgb from 0~255 to 0~1
    test_data = np.delete(train_data,6,1)/np.array([1,1,1,255,255,255])
    np.savetxt('/home/zonlin/Open3D_assignment/dataset/test_data.txt',test_data)
    logging.info(f"train_data and test_data have been generated")


    if visible:
        # show the raw point cloud
        data_pcd = o3d.io.read_point_cloud('/home/zonlin/Open3D_assignment/dataset/test_data.txt',format='xyzrgb')
        print("------ raw_data info -------")
        print(data_pcd)
        print("--------- raw_data ---------")
        o3d.visualization.draw_geometries([data_pcd])

if __name__=='__main__':
    log_dir = tools.setup_logger()

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", help="path to raw data.npy", metavar="FILE", required=True)
    parser.add_argument("--visible", help="Visualize the data", metavar="FILE", required=False)
    args = parser.parse_args()

    params = parser.parse_args()

    with open(params.data_dir, 'r'):
        try:
            raw_data = np.load(params.data_dir)

            main(params.data_dir, log_dir, params.visible)
        except:
            logging.error('Data file could not be read')
            exit(1)