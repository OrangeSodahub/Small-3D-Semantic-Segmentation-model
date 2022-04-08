import argparse
from tools.tools import *
from termcolor import colored
import yaml
import numpy as np
import shutil
import logging

# Predict on the test_data
def main(config: dict, log_dir: str):
    data_dir = config['test']['path']
    test_data = np.load(data_dir+'data.npy')


if __name__ == '__main__':
    log_dir = setup_logger()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="path to config file", metavar="FILE", required=False, default="./config/config.yaml")
    args = parser.parse_args()

    params = parser.parse_args()

    with open(params.config, 'r') as stream:
        try:
            config = yaml.load(stream,Loader=yaml.FullLoader)
            # backup config file
            shutil.copy(params.config, log_dir)

        except yaml.YAMLError as exc:
            logging.error('Configuration file could not be read')
            print(colored('Configuration file could not be read','red'))
            exit(1)

        main(config, log_dir)