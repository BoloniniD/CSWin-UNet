# test_continual_model.py
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

# Define model wrapper for handling continual learning outputs
class ContinualTestWrapper(nn.Module):
    def __init__(self, model, test_dataset, model_task_level):
        super().__init__()
        self.model = model
        self.test_dataset = test_dataset
        self.model_task_level = model_task_level

        # Define class mappings based on continual learning structure
        self.class_mappings = {
            'synapse': {
                'classes': 9,
                'indices': list(range(9))  # 0-8
            },
            'kits23': {
                'classes': 4,
                'indices': [0] + list(range(9, 12))  # 0, 9-11
            },
            'lits17': {
                'classes': 3,
                'indices': [0] + list(range(12, 14))  # 0, 12-13
            }
        }

        if test_dataset not in self.class_mappings:
            raise ValueError(f"Unknown test dataset: {test_dataset}")

        self.target_classes = self.class_mappings[test_dataset]['classes']
        self.class_indices = self.class_mappings[test_dataset]['indices']

        print(f"Testing on {test_dataset} using classes at indices: {self.class_indices}")

    def forward(self, x):
        output = self.model(x)
        # Extract only the relevant classes for the test dataset
        selected_output = output[:, self.class_indices, :, :]
        return selected_output

parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str, required=True,
                    help='root dir for validation volume data')
parser.add_argument('--test_dataset', type=str, required=True,
                    choices=['synapse', 'kits23', 'lits17'],
                    help='dataset to test on')
parser.add_argument('--model_path', type=str, required=True,
                    help='path to trained model checkpoint')
parser.add_argument('--model_task_level', type=str,
                    choices=['task1', 'task2', 'task3'],
                    help='which task level the model was trained up to (will be auto-detected if not provided)')

parser.add_argument('--list_dir', type=str,
                    help='list dir (will be auto-set based on test_dataset if not provided)')
parser.add_argument('--output_dir', type=str, required=True, help='output dir for results')
parser.add_argument('--batch_size', type=int, default=1, help='batch_size for testing')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--is_savenii', action="store_true", help='whether to save results during inference')
parser.add_argument('--test_save_dir', type=str, default='./predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file')

# Additional arguments from original script
parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, full: cache all data, part: sharding the dataset into nonoverlapping pieces and only cache one piece')
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
config = get_config(args)


def get_dataset_config(test_dataset):
    """Get dataset configuration based on test dataset"""
    configs = {
        'synapse': {
            'Dataset': Synapse_dataset,
            'list_dir': './lists/lists_Synapse',
            'num_classes': 9,
            'z_spacing': 1,
            'is_kits': False,
        },
        'kits23': {
            'Dataset': Synapse_dataset,  # Assuming you use the same dataset class with is_kits=True
            'list_dir': './lists/kits23',
            'num_classes': 4,
            'z_spacing': 1,
            'is_kits': True,
        },
        'lits17': {
            'Dataset': Synapse_dataset,  # You might need to create a Lits_dataset class
            'list_dir': './lists/lits17',
            'num_classes': 3,
            'z_spacing': 1,
            'is_kits': False,  # Assuming lits17 uses different dataset class
        },
    }
    return configs[test_dataset]


def get_model_num_classes(task_level):
    """Get total number of classes in model based on task level"""
    task_classes = {
        'task1': 9,   # Base model: Synapse only
        'task2': 12,  # After task 2: Synapse + kits23 (9 + 4 - 1)
        'task3': 14,  # After task 3: Synapse + kits23 + lits17 (12 + 3 - 1)
    }
    return task_classes[task_level]


def detect_model_task_level(checkpoint_path):
    """Detect the task level of the model from the checkpoint"""
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    state_dict = remove_prefixes(state_dict)

    # Look for the output layer to determine number of classes
    output_layer_keys = [
        'output.weight',
        'cswin_unet.output.weight',
        'segmentation_head.weight',
        'final.weight',
        'classifier.weight'
    ]

    num_classes = None
    for key in output_layer_keys:
        if key in state_dict:
            num_classes = state_dict[key].shape[0]
            print(f"Detected {num_classes} classes from {key}")
            break

    if num_classes is None:
        # Search through all keys for potential output layers
        for key, value in state_dict.items():
            if 'output' in key and 'weight' in key and len(value.shape) == 4:
                num_classes = value.shape[0]
                print(f"Detected {num_classes} classes from {key}")
                break

    if num_classes is None:
        raise RuntimeError("Could not detect number of classes from checkpoint")

    # Map number of classes to task level
    class_to_task = {
        9: 'task1',
        12: 'task2',
        14: 'task3'
    }

    if num_classes not in class_to_task:
        raise RuntimeError(f"Unknown number of classes: {num_classes}. Expected 9, 12, or 14.")

    detected_task = class_to_task[num_classes]
    print(f"Auto-detected model task level: {detected_task}")
    return detected_task, num_classes


def find_checkpoint(model_path):
    """Find the appropriate checkpoint file"""
    if os.path.isfile(model_path):
        return model_path

    # If it's a directory, try to find the final model
    if os.path.isdir(model_path):
        # Look for common checkpoint patterns
        checkpoint_patterns = [
            '*_final.pth',
            'task*_final.pth',
            '*_epoch_*.pth'
        ]

        import glob
        for pattern in checkpoint_patterns:
            files = glob.glob(os.path.join(model_path, pattern))
            if files:
                # Return the most recent file
                return max(files, key=os.path.getctime)

    raise FileNotFoundError(f"Could not find checkpoint at {model_path}")


def remove_prefixes(state_dict):
    """Remove common prefixes from state dict keys"""
    prefixes_to_remove = ['base_model.', 'module.']
    new_state_dict = {}

    for key, value in state_dict.items():
        new_key = key
        for prefix in prefixes_to_remove:
            if new_key.startswith(prefix):
                new_key = new_key.replace(prefix, '', 1)
        new_state_dict[new_key] = value

    return new_state_dict


def inference(args, model, test_save_path=None):
    """Run inference on the test dataset"""
    dataset_config = get_dataset_config(args.test_dataset)

    # Create dataset instance
    if args.test_dataset == 'synapse':
        db_test = dataset_config['Dataset'](
            base_dir=args.volume_path,
            split="test_vol",
            list_dir=dataset_config['list_dir']
        )
    elif args.test_dataset == 'kits23':
        db_test = dataset_config['Dataset'](
            base_dir=args.volume_path,
            split="test_vol",
            list_dir=dataset_config['list_dir'],
            is_kits=True
        )
    elif args.test_dataset == 'lits17':
        # You might need to implement a specific Lits_dataset class
        db_test = dataset_config['Dataset'](
            base_dir=args.volume_path,
            split="test_vol",
            list_dir=dataset_config['list_dir']
        )

    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))

    model.eval()
    metric_list = 0.0
    num_classes = dataset_config['num_classes']

    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]

        metric_i = test_single_volume(
            image, label, model,
            classes=num_classes,
            patch_size=[args.img_size, args.img_size],
            test_save_path=test_save_path,
            case=case_name,
            z_spacing=dataset_config['z_spacing']
        )

        metric_list += np.array(metric_i)
        logging.info('idx %d case %s mean_dice %f mean_hd95 %f' %
                    (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))

    metric_list = metric_list / len(db_test)

    # Log per-class results
    for i in range(1, num_classes):  # Skip background class in detailed logging
        logging.info('Mean class %d mean_dice %f mean_hd95 %f' %
                    (i, metric_list[i-1][0], metric_list[i-1][1]))

    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]

    logging.info('Testing performance: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    logging.info(f"Dataset: {args.test_dataset}, Model: {args.model_task_level}")

    return "Testing Finished!"


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

    # Set up dataset configuration
    dataset_config = get_dataset_config(args.test_dataset)

    # Auto-set list_dir if not provided
    if args.list_dir is None:
        args.list_dir = dataset_config['list_dir']

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Find checkpoint
    checkpoint_path = find_checkpoint(args.model_path)

    # Auto-detect model task level if not provided
    if args.model_task_level is None:
        detected_task_level, model_num_classes = detect_model_task_level(checkpoint_path)
        args.model_task_level = detected_task_level
    else:
        model_num_classes = get_model_num_classes(args.model_task_level)
        # Verify that the specified task level matches the checkpoint
        try:
            _, detected_num_classes = detect_model_task_level(checkpoint_path)
            if detected_num_classes != model_num_classes:
                print(f"Warning: You specified {args.model_task_level} ({model_num_classes} classes) "
                      f"but the checkpoint has {detected_num_classes} classes.")
                response = input("Do you want to use auto-detected task level? (y/n): ")
                if response.lower() == 'y':
                    detected_task_level, model_num_classes = detect_model_task_level(checkpoint_path)
                    args.model_task_level = detected_task_level
        except Exception as e:
            print(f"Could not verify task level: {e}")

    # Set up logging
    logging.basicConfig(
        filename=os.path.join(args.output_dir, f"test_{args.test_dataset}_{args.model_task_level}.log"),
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S'
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    test_num_classes = dataset_config['num_classes']

    print(f"\n=== Testing Configuration ===")
    print(f"Test Dataset: {args.test_dataset}")
    print(f"Model Task Level: {args.model_task_level} (auto-detected)" if args.model_task_level else f"Model Task Level: {args.model_task_level}")
    print(f"Model Total Classes: {model_num_classes}")
    print(f"Test Classes: {test_num_classes}")
    print(f"Checkpoint Path: {checkpoint_path}")
    print("=" * 35)

    # Initialize model with correct number of classes
    net = ViT_seg(config, img_size=args.img_size, num_classes=model_num_classes).cuda()

    # Load checkpoint
    logging.info(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location='cuda')
    state_dict = remove_prefixes(state_dict)

    # Load the state dict
    try:
        net.load_state_dict(state_dict, strict=True)
        logging.info("Loaded model with strict=True")
    except RuntimeError as e:
        logging.warning(f"Strict loading failed: {e}")
        logging.info("Attempting to load with strict=False")
        net.load_state_dict(state_dict, strict=False)

    print(f"Successfully loaded model from {checkpoint_path}")

    # Wrap model for continual learning testing
    net = ContinualTestWrapper(net, args.test_dataset, args.model_task_level)

    logging.info(f"Model initialized with {model_num_classes} classes")
    logging.info(f"Testing on {args.test_dataset} with {test_num_classes} classes")

    # Set up save directory if needed
    if args.is_savenii:
        test_save_path = os.path.join(args.output_dir, f"predictions_{args.test_dataset}")
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None

    # Calculate FLOPs and params
    try:
        net_for_flops = net.model  # Use the underlying model
        net_for_flops.eval()
        dummy_input = torch.randn(1, 3, args.img_size, args.img_size).cuda()
        flops, params = profile(net_for_flops, inputs=(dummy_input,))
        flops, params = clever_format([flops, params], "%.3f")
        logging.info(f'FLOPs: {flops}')
        logging.info(f'Params: {params}')
        print('FLOPs:', flops)
        print('Params:', params)
    except Exception as e:
        logging.warning(f"Could not calculate FLOPs/params: {e}")

    # Run inference
    inference(args, net, test_save_path)
