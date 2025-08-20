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

# --- New Imports for Visualization ---
import matplotlib.pyplot as plt
import torch.nn.functional as F
# --- End New Imports ---


parser = argparse.ArgumentParser()
parser.add_argument('--volume_path', type=str,
                    default='./datasets/Synapse/test_vol_h5', help='root dir for validation volume data')
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
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


# --- New Function to Save Visuals ---
def save_visuals(image, label, prediction, case_name, slice_idx, save_dir):
    """Saves a side-by-side comparison of the image, ground truth, and prediction."""
    # Ensure inputs are 2D numpy arrays
    image = np.squeeze(image)
    label = np.squeeze(label)
    prediction = np.squeeze(prediction)

    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Input Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(label, cmap='jet', vmin=0, vmax=args.num_classes-1)
    plt.title('Ground Truth')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(prediction, cmap='jet', vmin=0, vmax=args.num_classes-1)
    plt.title('Model Prediction')
    plt.axis('off')

    save_path = os.path.join(save_dir, f"{case_name}_slice_{slice_idx}.png")
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()  # Close the figure to free up memory
# --- End New Function ---


def inference(args, model, test_save_path=None, visual_save_dir=None):
    if args.dataset == "Synapse":
        db_test = args.Dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir)
    elif args.dataset == "kits23":
        db_test = args.Dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir)
    elif args.dataset == "lits17":
        db_test = args.Dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir)
    else:
        raise ValueError(f"Dataset {args.dataset} not supported.")

    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0

    # --- New: Configuration for saving visuals ---
    max_visuals_to_save = 5  # Save visuals for the first 5 cases
    num_visuals_saved = 0
    # --- End New ---

    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]

        # --- New: Generate and save a visual prediction for a subset of images ---
        if visual_save_dir and num_visuals_saved < max_visuals_to_save:
            # The input from dataloader is a 3D volume, likely with shape (1, num_slices, H, W)
            # We select the middle slice for visualization.
            num_slices = image.shape[1]
            mid_slice_idx = num_slices // 2

            # Prepare the slice for the model: (B, C, H, W), C=3, H=W=img_size
            image_slice_tensor = image[:, mid_slice_idx, :, :].unsqueeze(1)  # Shape: (1, 1, H, W)
            image_slice_for_model = image_slice_tensor.repeat(1, 3, 1, 1)  # Shape: (1, 3, H, W)
            resized_slice_for_model = F.interpolate(image_slice_for_model,
                                                    size=(args.img_size, args.img_size),
                                                    mode='bilinear',
                                                    align_corners=False)
            with torch.no_grad():
                output = model(resized_slice_for_model.cuda())
                # Resize output back to original slice dimensions for accurate visualization
                output_resized = F.interpolate(output, size=(h, w), mode='bilinear', align_corners=False)
                prediction_mask = torch.argmax(output_resized, dim=1).squeeze(0).cpu().numpy()

            # Get original image and label slices for saving
            original_image_slice = image[0, mid_slice_idx, :, :].cpu().numpy()
            label_slice = label[0, mid_slice_idx, :, :].cpu().numpy()

            save_visuals(original_image_slice, label_slice, prediction_mask, case_name, mid_slice_idx, visual_save_dir)
            num_visuals_saved += 1
        # --- End New ---

        metric_i = test_single_volume(image, label, model, classes=args.num_classes, patch_size=[args.img_size, args.img_size],
                                      test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)
        metric_list += np.array(metric_i)
        logging.info('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
    metric_list = metric_list / len(db_test)
    for i in range(1, args.num_classes):
        logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    return "Testing Finished!"


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

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
        'lits17': {
            'Dataset': Synapse_dataset,
            'volume_path': args.volume_path,
            'list_dir': './lists/lits17',
            'num_classes': 3,
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

    net = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes).cuda()

    snapshot = os.path.join(args.output_dir, 'best_model.pth')
    if not os.path.exists(snapshot): snapshot = snapshot.replace('best_model', 'epoch_'+str(args.max_epochs-1))

    msg = net.load_state_dict(torch.load(snapshot))
    print("Loaded model from:", snapshot)
    snapshot_name = snapshot.split('/')[-1]

    log_folder = './test_log/test_log_'
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=os.path.join(log_folder, f"log_{dataset_name}.txt"), level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(snapshot_name)

    # --- New: Create directory for visual predictions ---
    visual_save_dir = "./test_visuals"
    os.makedirs(visual_save_dir, exist_ok=True)
    logging.info(f"Visualizations will be saved to {visual_save_dir}")
    # --- End New ---

    if args.is_savenii:
        args.test_save_dir = os.path.join(args.output_dir, "predictions")
        test_save_path = args.test_save_dir
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None

    inference(args, net, test_save_path, visual_save_dir)

    try:
        dummy_input = torch.randn(1, 3, args.img_size, args.img_size).cuda()
        flops, params = profile(net, inputs=(dummy_input,))
        flops, params = clever_format([flops, params], "%.3f")
        print('FLOPs:', flops)
        print('Params:', params)
    except Exception as e:
        print(f"Could not calculate FLOPs/Params: {e}")
