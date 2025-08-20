import argparse
import logging
import os

#os.environ['CUDA_VISIBLE_DEVICES'] = '1'

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

# Define model wrapper for slicing outputs
class OutputSliceWrapper(nn.Module):
    def __init__(self, model, num_classes):
        super().__init__()
        self.model = model
        self.num_classes = num_classes

    def forward(self, x):
        output = self.model(x)
        return output[:, :self.num_classes, :, :]

parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str,
                    default='../data/Synapse/test_vol_h5', help='root dir for validation volume data')
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
parser.add_argument('--model_num_classes', type=int,
                    default=9, help='total number of classes in the model')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--output_dir', type=str, help='output dir')
parser.add_argument('--max_iterations', type=int,default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--is_savenii', action="store_true", help='whether to save results during inference')
parser.add_argument('--test_save_dir', type=str, default='../predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                            'full: cache all data, '
                            'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')

args = parser.parse_args()
if args.dataset == "Synapse":
    args.volume_path = os.path.join(args.volume_path, "test_vol_h5")
config = get_config(args)


def inference(args, model, test_save_path=None):
    if args.dataset == "Synapse":
        db_test = args.Dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir)
    elif args.dataset == "kits23":
        db_test = args.Dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir)
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
    for i in range(1, args.num_classes+1):
        logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    return "Testing Finished!"

def remove_base_model_prefix(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('base_model.'):
            new_key = key.replace('base_model.', '', 1)
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict


if __name__ == "__main__":

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_config = {
        'Synapse': {
            'Dataset': Synapse_dataset,
            'volume_path': args.volume_path,
            'list_dir': './lists/lists_Synapse',
            'num_classes': 9,
            'z_spacing': 1,
        },
        'kits23': {
            'Dataset': Synapse_dataset,
            'volume_path': args.volume_path,
            'list_dir': './lists/kits23',
            'num_classes': 4,
            'z_spacing': 1,
        },
    }
    dataset_name = args.dataset
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.volume_path = dataset_config[dataset_name]['volume_path']
    args.Dataset = dataset_config[dataset_name]['Dataset']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.z_spacing = dataset_config[dataset_name]['z_spacing']
    args.is_pretrain = True

    # IMPORTANT: Always initialize model with the full number of classes (9)
    # that it was trained with, regardless of current dataset
    FULL_MODEL_CLASSES = 9  # This should match the training configuration
    net = ViT_seg(config, img_size=args.img_size, num_classes=FULL_MODEL_CLASSES).cuda()

    # Find the best model checkpoint
    snapshot = os.path.join(args.output_dir, 'continual_surgical_tpgm_final.pth')
    if not os.path.exists(snapshot):
        # Try to find the last epoch checkpoint
        checkpoint_files = [f for f in os.listdir(args.output_dir) if f.startswith('finetuned_epoch_')]
        if checkpoint_files:
            # Find the highest epoch number
            epochs = [int(f.split('_')[-1].split('.')[0]) for f in checkpoint_files]
            max_epoch = max(epochs)
            snapshot = os.path.join(args.output_dir, f'finetuned_epoch_{max_epoch}.pth')
        else:
            raise FileNotFoundError(f"No suitable checkpoint found in {args.output_dir}")

    # Load model weights
    state_dict = torch.load(snapshot)
    state_dict = remove_base_model_prefix(state_dict=state_dict)
    net.load_state_dict(state_dict)
    print(f"Loaded model from {snapshot}")
    print(f"Model initialized with {FULL_MODEL_CLASSES} classes, testing on {args.num_classes} classes for {args.dataset}")

    # Wrap model to output only active classes for current dataset
    net = OutputSliceWrapper(net, args.num_classes)

    log_folder = './test_log/test_log_'
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=os.path.join(args.output_dir, "test_log.txt"), level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(f"Testing model: {snapshot}")
    logging.info(f"Model has {FULL_MODEL_CLASSES} total classes, testing {args.num_classes} classes for {args.dataset}")

    if args.is_savenii:
        args.test_save_dir = os.path.join(args.output_dir, "predictions")
        test_save_path = args.test_save_dir
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None

    # Calculate FLOPs and params on original model (without wrapper)
    net_without_wrapper = net.model
    net_without_wrapper.eval()
    dummy_input = torch.randn(1, 3, args.img_size, args.img_size).cuda()
    flops, params = profile(net_without_wrapper, inputs=(dummy_input,))
    flops, params = clever_format([flops, params], "%.3f")
    logging.info(f'FLOPs: {flops}')
    logging.info(f'Params: {params}')
    print('FLOPs:', flops)
    print('Params:', params)

    # Run inference
    inference(args, net, test_save_path)
