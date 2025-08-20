# test_multitask.py
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

parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str,
                    default='./datasets/Synapse/test_vol_h5', help='root dir for validation volume data')
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name, determines which task to test (e.g., Synapse, kits23, lits17)')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--output_dir', type=str, help='output dir where the final continual model is saved')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--is_savenii', action="store_true", help='whether to save results during inference')
parser.add_argument('--test_save_dir', type=str, default='../predictions', help='saving prediction as nii!')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )

# Arguments for continual learning model testing
parser.add_argument('--is_continual_model', action='store_true',
                    help='Whether the model to test is a continual learning model')
parser.add_argument('--model_num_classes', type=int, default=9,
                    help='Number of classes in the model architecture (usually 9 for your setup)')
parser.add_argument('--num_classes_synapse', type=int, default=9,
                    help='Number of classes in Synapse dataset')
parser.add_argument('--num_classes_kits', type=int, default=4,
                    help='Number of classes in KITS23 dataset')
parser.add_argument('--num_classes_lits', type=int, default=3,
                    help='Number of classes in LiTS17 dataset')

parser.add_argument('--model_name', type=str, default='epoch_149.pth',
                    help='Name of the model file to load from output_dir')

# Preserved all original arguments
parser.add_argument('--max_epochs', type=int, default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24, help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument("--opts", help="Modify config options by adding 'KEY VALUE' pairs. ", default=None, nargs='+')
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
config = get_config(args)


def load_model_state_dict(model_path):
    """
    Load a model state dict and handle different formats
    """
    state_dict = torch.load(model_path, map_location='cpu')

    # Handle different checkpoint formats
    if 'model' in state_dict:
        state_dict = state_dict['model']
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    # Remove common prefixes
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        if key.startswith('module.'):
            new_key = key[len('module.'):]
        elif key.startswith('base_model.'):
            new_key = key[len('base_model.'):]
        new_state_dict[new_key] = value

    return new_state_dict


class ContinualModelWrapper(nn.Module):
    """
    A wrapper that takes a full model and slices its output to isolate
    the logits for a specific task.
    """
    def __init__(self, model, class_start_index, num_classes_task):
        super().__init__()
        self.model = model
        self.start_idx = class_start_index
        self.end_idx = class_start_index + num_classes_task
        print(f"Wrapper created to slice model output from index {self.start_idx} to {self.end_idx-1} (total: {num_classes_task} classes)")

    def forward(self, x):
        full_output = self.model(x)
        # Return only the slice of the output corresponding to the current task
        sliced_output = full_output[:, self.start_idx:self.end_idx, :, :]
        return sliced_output


def inference(args, model, test_save_path=None):
    # Dataset loading logic
    if args.dataset == 'Synapse':
        db_test = Synapse_dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir)
        args.num_classes = args.num_classes_synapse
        args.z_spacing = 1
    elif args.dataset == 'kits23':
        db_test = Synapse_dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir, is_kits=True)
        args.num_classes = args.num_classes_kits
        args.z_spacing = 1
    elif args.dataset == 'lits17':
        db_test = Synapse_dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir, is_lits=True)
        args.num_classes = args.num_classes_lits
        args.z_spacing = 1
    else:
        raise ValueError(f"Dataset '{args.dataset}' not supported. Available: Synapse, kits23, lits17")

    if len(db_test) == 0:
        logging.error(f"No data found for dataset '{args.dataset}'. Please check paths.")
        raise ValueError("Test dataset is empty. Aborting.")

    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info(f"Testing on {args.dataset}: {len(testloader)} volumes.")
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

    # Setup logging
    log_folder = './test_log/test_log_multitask'
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=os.path.join(log_folder, f"log_{args.dataset}.txt"), level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    # FIXED: Always create the model with the original architecture size
    net = ViT_seg(config, img_size=args.img_size, num_classes=args.model_num_classes).cuda()

    # Load the model
    snapshot_path = os.path.join(args.output_dir, args.model_name)
    if not os.path.exists(snapshot_path):
        raise FileNotFoundError(f"Model snapshot not found at {snapshot_path}")

    # Load model state dict
    state_dict = load_model_state_dict(snapshot_path)

    # Check if the state dict matches the model architecture
    model_dict = net.state_dict()
    mismatched_keys = []
    for key in state_dict.keys():
        if key in model_dict:
            if state_dict[key].shape != model_dict[key].shape:
                mismatched_keys.append(f"{key}: checkpoint {state_dict[key].shape} vs model {model_dict[key].shape}")
        else:
            mismatched_keys.append(f"{key}: not found in model")

    if mismatched_keys:
        print("WARNING: Found mismatched keys:")
        for key in mismatched_keys:
            print(f"  {key}")

    net.load_state_dict(state_dict, strict=False)
    logging.info(f"Loaded model from {snapshot_path}")

    # Handle continual learning model testing
    if args.is_continual_model:
        # Define class mappings for continual learning
        # Based on your training script, the class allocation should be:
        # Synapse: classes 0-8 (9 classes total)
        # KiTS23: classes 0-3 (4 classes, reusing background class)
        # LiTS17: classes 0-2 (3 classes, reusing background class)

        dataset_class_mapping = {
            'Synapse': {'start_index': 0, 'num_classes': args.num_classes_synapse},
            'kits23': {'start_index': 0, 'num_classes': args.num_classes_kits},
            'lits17': {'start_index': 0, 'num_classes': args.num_classes_lits}
        }

        if args.dataset not in dataset_class_mapping:
            raise ValueError(f"Dataset '{args.dataset}' not configured for continual learning testing.")

        class_info = dataset_class_mapping[args.dataset]
        class_start_index = class_info['start_index']
        task_num_classes = class_info['num_classes']

        print(f"=== Continual Learning Model Testing ===")
        print(f"Dataset: {args.dataset}")
        print(f"Model architecture classes: {args.model_num_classes}")
        print(f"Task classes: {task_num_classes}")
        print(f"Class slice: {class_start_index} to {class_start_index + task_num_classes - 1}")
        print("=" * 45)

        # Wrap the model to test specific task
        model = ContinualModelWrapper(net, class_start_index, task_num_classes)
    else:
        model = net

    logging.info(f"--- Testing on dataset: {args.dataset} ---")
    logging.info(f"Args: {str(args)}")

    if args.is_savenii:
        test_save_path = os.path.join(args.output_dir, f"predictions_{args.dataset}")
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None

    inference(args, model, test_save_path)
