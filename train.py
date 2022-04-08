import argparse
import numpy as np
from sklearn.utils import shuffle
from tools.tools import *
from models.model import *
from tools.train_utils import *
import torch.utils.data as data
import open3d.ml.torch as ml3d
import torch
import yaml
import shutil
import tqdm
import logging
from termcolor import colored

avg_iou_per_epoch = [0]
avg_class_acc_per_epoch = [0]
avg_loss_per_epoch = [0]

def main(config: dict, log_dir: str):
    # Prepare network
    epochs = config['train']['epoch']
    batch_size = config['train']['batch_size']
    epoch_start = 0

    ckpt_save_interval = config['train']['ckpt_save_interval']
    ckpt_dir = config['train']['ckpt_dir']
    os.makedirs(ckpt_dir, exist_ok=True)

    Model = import_class('models', config['model']['name'])
    model = Model()

    ops = {'loss': model.loss
          }

    train_dataset_position = np.load(config['dataset']['train_data_path']+'data.npy')[0:3]
    train_dataset_feature = np.load(config['dataset']['train_data_path']+'data.npy')[4:7]
    train_set = data.TensorDataset(train_dataset_position, train_dataset_feature)
    train_loader = data(train_set, batch_size, shuffle=True)

    test_dataset = np.load(config['dataset']['test_data_path']+'data.npy')
    test_set = data.TensorDataset(test_dataset)
    test_loader = data(test_set, shuffle=False)

    # Train with evaluation
    logging.info("***************** Start training *****************")
    with tqdm.trange(epoch_start, epochs, desc='epochs') as tbar, \
        tqdm.tqdm(total=len(train_loader), leave=False, desc='Training epoch %04d / %04d ' % (epoch, epochs)) as pbar:
        for epoch in tbar:
            train_one_epoch(train_loader, ops, epoch)

            pbar.close()
            eval_one_epoch(test_loader)

            # Save the checkpoints
            if epoch % ckpt_save_interval == 0:
                ckpt_name = os.path.join(ckpt_dir, 'checkpoint_epoch_%d' % epoch)
                save_checkpoint(
                    checkpoint_state(model, epoch), filename=ckpt_name,
                )

    pbar.close()
    pbar = tqdm.tqdm(total=len(train_loader), leave=False, desc='train')
    logging.info("***************** End training *******************")


def train_one_epoch(train_loader, ops, epoch):
    # Device is CPU
    pytorch_device = torch.device('cpu')

    for step, batch in enumerate(train_loader):
        


def eval_one_epoch(test_loader):
    eval_dict = {}
    total_loss = count = 0.0

    # eval one epoch
    for i, data in tqdm.tqdm(enumerate(test_loader, 0), total=len(test_loader), leave=False, desc='val'):

        loss, tb_dict, disp_dict = model_fn_eval(model, data)

        total_loss += loss.item()
        for k, v in tb_dict.items():
            eval_dict[k] = eval_dict.get(k, 0) + v

    # statistics this epoch
    for k, v in eval_dict.items():
        eval_dict[k] = eval_dict[k] / max(count, 1)


if __name__ == '__main__':
    log_dir = setup_logger()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to config file", metavar="FILE", required=False, default='./config/config.yaml')
    args = parser.parse_args()

    params = parser.parse_args()

    with open(params.config, 'r') as stream:
        try:
            config = yaml.load(stream,Loader=yaml.FullLoader)
            # backup config file
            shutil.copy(params.config, log_dir)

            main(config, log_dir)
            print(colored('Finised training.','green'))
        except yaml.YAMLError as exc:
            logging.error('Configuration file could not be read')
            print(colored('Configuration file could not be read.','Red'))
            exit(1)