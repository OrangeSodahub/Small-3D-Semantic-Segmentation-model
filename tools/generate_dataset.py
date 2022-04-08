import argparse
from curses import meta
import numpy as np
import open3d as o3d
import tools
from termcolor import colored
import shutil
import yaml
import logging
import os

np.set_printoptions(suppress=True)

def main(config: dict, log_dir: str, visible: bool):

    data_dir = config['dataset']['data_path']
    train_dir = config['dataset']['train_data_path']
    os.makedirs(train_dir,exist_ok=True)
    test_dir = config['dataset']['test_data_path']
    os.makedirs(test_dir,exist_ok=True)
    

    # load the xyzrgb
    train_data = np.load(data_dir+'data.npy')[:,0:6]
    np.save(train_dir+'data.npy',train_data)
    # convert rgb from 0~255 to 0~1
    test_data = train_data/np.array([1,1,1,255,255,255])
    np.save(test_dir+'data.npy',test_data)
    logging.info(f"train_data and test_data have been generated")


    if visible=='True':
        # show the raw point cloud
        np.savetxt(test_dir+'data.txt',test_data)
        data_pcd = o3d.io.read_point_cloud(test_dir+'data.txt',format='xyzrgb')
        print("------ raw_data info -------")
        print(data_pcd)
        print("-------- raw_data ----------")
        o3d.visualization.draw_geometries([data_pcd])

if __name__=='__main__':
    log_dir = tools.setup_logger()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="path to config file", metavar="FILE", required=False, default="./config/config.yaml")
    parser.add_argument("--visible", help="Visualize the data", metavar="FILE", required=False, default='False')
    args = parser.parse_args()

    params = parser.parse_args()

    with open(params.config, 'r') as f:
        try:
            config = yaml.load(f,Loader=yaml.FullLoader)
            shutil.copy(params.config,log_dir)

        except:
            logging.error('Config file could not be read')
            print(colored('Config file could not be read','red'))
            exit(1)
        
        main(config, log_dir, params.visible)
        print(colored('Finished generating train_data and test_data successfully', 'green'))