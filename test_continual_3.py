# test_continual.py
import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.dataset_synapse import Synapse_dataset
from utils import test_single_volume
from networks.vision_transformer import CSwinUnet as ViT_seg
from config import get_config
from thop import profile, clever_format

# --- All arguments are the same as before ---
parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str,
                    default='../data/Synapse/test_vol_h5', help='root dir for validation volume data')
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name, determines which task to test')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--output_dir', type=str, help='output dir where the continual model is saved')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--is_savenii', action="store_true", help='whether to save results during inference')
parser.add_argument('--test_save_dir', type=str, default='../predictions', help='saving prediction as nii!')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
parser.add_argument('--num_classes_new', type=int,
                    default=4, help='output channel of the new task network')
parser.add_argument('--num_classes_old', type=int,
                    default=9, help='output channel of the old task network')
parser.add_argument('--model_name', type=str, default='continual_epoch_29.pth',
                    help='Name of the model file to load from output_dir')
parser.add_argument('--max_epochs', type=int, default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24, help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
parser.add_argument('--zip', action='store_true')
parser.add_argument('--cache-mode', type=str, default='part')
parser.add_argument('--resume', default=None)
parser.add_argument('--accumulation-steps', type=int, default=1)
parser.add_argument('--use-checkpoint', action='store_true')
parser.add_argument('--amp-opt-level', type=str, default='O1')
parser.add_argument('--tag', type=str, default='test')
parser.add_argument('--eval', action='store_true')
parser.add_argument('--throughput', action='store_true')

args = parser.parse_args()
args.num_classes = args.num_classes_old + args.num_classes_new
config = get_config(args)


class ContinualModelWrapper(nn.Module):
    def __init__(self, model, task_type, num_classes_old):
        super().__init__()
        self.model = model
        self.task_type = task_type
        self.num_classes_old = num_classes_old

    def forward(self, x):
        full_output = self.model(x)
        if self.task_type == 'old':
            return full_output[:, :self.num_classes_old, :, :]
        elif self.task_type == 'new':
            return full_output[:, self.num_classes_old:, :, :]
        else:
            raise ValueError("task_type must be 'old' or 'new'")


def inference(args, model, test_save_path=None):
    # The dataset loader is chosen based on the --dataset argument
    db_test = args.Dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir, is_kits=(args.dataset == 'kits23'))

    # --- DEBUGGING CHANGE 1: Add a sanity check for the dataset length ---
    if len(db_test) == 0:
        logging.error(f"No data found for dataset '{args.dataset}'. Please check the following paths:")
        logging.error(f"  - Volume Path: {os.path.abspath(args.volume_path)}")
        logging.error(f"  - List Dir: {os.path.abspath(args.list_dir)}")
        raise ValueError("Test dataset is empty. Aborting.")

    logging.info(f"Found {len(db_test)} volumes for testing.")

    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]

        metric_i = test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                      test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)
        metric_list += np.array(metric_i)
        logging.info('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))

    metric_list = metric_list / len(db_test)
    for i in range(1, args.num_classes):
        logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))

    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    logging.info('Testing performance for dataset %s: mean_dice : %f mean_hd95 : %f' % (args.dataset, performance, mean_hd95))
    return "Testing Finished!"


if __name__ == "__main__":
    cudnn.benchmark = True
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_config = {
        'Synapse': {
            'Dataset': Synapse_dataset,
            'volume_path': './datasets/Synapse/test_vol_h5',
            'list_dir': './lists/lists_Synapse',
            'num_classes': args.num_classes_old,
            'z_spacing': 1,
            'task_type': 'old'
        },
        'kits23': {
            'Dataset': Synapse_dataset,
            'volume_path': args.volume_path,
            'list_dir': './lists/kits23',
            'num_classes': args.num_classes_new,
            'z_spacing': 1,
            'task_type': 'new'
        },
    }

    dataset_name = args.dataset
    if dataset_name not in dataset_config:
        raise ValueError(f"Dataset {dataset_name} not configured for continual testing.")

    cfg_data = dataset_config[dataset_name]
    args.num_classes = cfg_data['num_classes']
    args.volume_path = cfg_data['volume_path']
    args.Dataset = cfg_data['Dataset']
    args.list_dir = cfg_data['list_dir']
    args.z_spacing = cfg_data['z_spacing']
    task_type = cfg_data['task_type']

    total_classes = args.num_classes_old + args.num_classes_new
    net = ViT_seg(config, img_size=args.img_size, num_classes=total_classes).cuda()

    snapshot_path = os.path.join(args.output_dir, args.model_name)
    if not os.path.exists(snapshot_path):
        raise FileNotFoundError(f"Model snapshot not found at {snapshot_path}")

    # --- DEBUGGING CHANGE 2: More robust logging setup ---
    log_folder = './test_log/test_log_continual'
    os.makedirs(log_folder, exist_ok=True)

    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO) # Set the minimum level to capture

    # Remove any existing handlers to avoid duplicate logs
    while logger.hasHandlers():
        logger.removeHandler(logger.handlers[0])

    # Create a handler for file output
    file_handler = logging.FileHandler(os.path.join(log_folder, f"log_{dataset_name}.txt"))
    file_handler.setFormatter(logging.Formatter('[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(file_handler)

    # Create a handler for console output
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter('%(message)s')) # Simple format for console
    logger.addHandler(stream_handler)

    net.load_state_dict(torch.load(snapshot_path))
    logging.info(f"Loaded model from {snapshot_path}")

    wrapped_model = ContinualModelWrapper(net, task_type, args.num_classes_old)

    snapshot_name = os.path.basename(snapshot_path)
    logging.info(f"Args: {str(args)}")
    logging.info(f"Testing model: {snapshot_name}")

    if args.is_savenii:
        test_save_path = os.path.join(args.output_dir, f"predictions_{dataset_name}")
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None

    inference(args, wrapped_model, test_save_path)

    dummy_input = torch.randn(1, 3, args.img_size, args.img_size).cuda()
    flops, params = profile(net, inputs=(dummy_input,))
    flops, params = clever_format([flops, params], "%.3f")
    logging.info(f'Full Model FLOPs: {flops}')
    logging.info(f'Full Model Params: {params}')
