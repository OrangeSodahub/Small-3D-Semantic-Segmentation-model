import argparse
from tools.tools import *
import yaml
import shutil
import logging

# Predict on the test_data
def main():


if __name__ == '__main__':
    log_dir = setup_logger()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="experiment definition file", metavar="FILE", required=True)
    args = parser.parse_args()

    params = parser.parse_args()

    with open(params.config, 'r') as stream:
        try:
            config = yaml.load(stream,Loader=yaml.FullLoader)
            # backup config file
            shutil.copy(params.config, log_dir)

            isTrain = False

            if config['modus'] == 'TRAIN_VAL':
                isTrain = True
            elif config['modus'] == 'TEST':
                isTrain = False

            main(config, log_dir, isTrain)
        except yaml.YAMLError as exc:
            logging.error('Configuration file could not be read')
            exit(1)