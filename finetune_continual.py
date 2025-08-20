# finetune_continual.py
import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.vision_transformer import CSwinUnet as ViT_seg
from config import get_config
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from utils import DiceLoss
from tqdm import tqdm
import torch.optim as optim
from torchvision import transforms
from datasets.dataset_synapse import Synapse_dataset, RandomGenerator

# --- Argument Parser Setup (Complete) ---
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/kits23/train_npz', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='kits23', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/kits23', help='list dir')
parser.add_argument('--output_dir', type=str, help='output dir')
parser.add_argument('--num_classes_new', type=int,
                    default=4, help='output channel of the new task network')
parser.add_argument('--num_classes_old', type=int,
                    default=9, help='output channel of the old task network')
parser.add_argument('--max_iterations', type=int,
                    default=10000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=50, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=12, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.001,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
parser.add_argument('--pretrained_path', type=str, required=True,
                    help='path to pretrained model checkpoint')
parser.add_argument('--data_fraction', type=float, default=0.1,
                    help='fraction of data to use for finetuning (default: 0.1)')
parser.add_argument('--freeze_layers', type=int, default=0,
                    help='number of transformer layers to freeze (0 = no freezing)')
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
args.num_classes = args.num_classes_new + args.num_classes_old
config = get_config(args)


# --- THE FIX: Moved worker_init_fn to the global scope ---
def worker_init_fn(worker_id):
    """
    Sets the random seed for each dataloader worker.
    This is essential for reproducible data loading.
    """
    random.seed(args.seed + worker_id)


def load_pretrained_weights_continual(model, pretrained_path, num_classes_old, num_classes_new):
    """
    Loads pretrained weights and adapts the final layer for continual learning.
    The new output layer will have (num_classes_old + num_classes_new) channels.
    """
    print(f"Loading pretrained model from {pretrained_path} for continual learning.")
    print(f"Old task classes: {num_classes_old}, New task classes: {num_classes_new}")

    pretrained_dict = torch.load(pretrained_path, map_location='cpu')

    if 'module.' in list(pretrained_dict.keys())[0]:
        print("Removing 'module.' prefix from pretrained weights.")
        pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}

    model_dict = model.state_dict()
    output_layer_keys = [k for k in pretrained_dict.keys() if 'output' in k or 'head' in k]
    if not output_layer_keys:
        raise ValueError("Could not identify the output layer in the pretrained model.")

    print(f"Identified output layer keys: {output_layer_keys}")
    pretrained_dict_filtered = {k: v for k, v in pretrained_dict.items() if k not in output_layer_keys}
    model.load_state_dict(pretrained_dict_filtered, strict=False)
    print(f"Loaded {len(pretrained_dict_filtered)} backbone layers from pretrained model.")

    with torch.no_grad():
        for key in output_layer_keys:
            if key in model_dict:
                old_weights = pretrained_dict[key]
                model_dict[key][:num_classes_old] = old_weights
                print(f"Copied pretrained weights for '{key}' for the first {num_classes_old} classes.")
            else:
                print(f"Warning: Key '{key}' from pretrained model not found in the new model.")

    model.load_state_dict(model_dict)
    print("Successfully adapted output layer for continual learning.")
    return model


def trainer_continual(args, model, snapshot_path):
    """
    Trainer function for the continual learning baseline.
    """
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    base_lr = args.base_lr
    num_classes_new = args.num_classes_new
    num_classes_old = args.num_classes_old
    total_num_classes = num_classes_old + num_classes_new
    batch_size = args.batch_size * args.n_gpu

    db_train_full = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                                    transform=transforms.Compose(
                                        [RandomGenerator(output_size=[args.img_size, args.img_size])]),
                                    is_kits=True)

    total_samples = len(db_train_full)
    subset_size = int(total_samples * args.data_fraction)
    random.seed(args.seed)
    indices = random.sample(range(total_samples), subset_size)
    db_train = Subset(db_train_full, indices)

    print(f"Using {subset_size}/{total_samples} samples ({args.data_fraction*100:.1f}%) for continual learning.")
    logging.info(f"Using {subset_size}/{total_samples} samples ({args.data_fraction*100:.1f}%) for continual learning.")

    # The worker_init_fn is now passed from the global scope
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                           num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(total_num_classes)
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    logging.info(f"{len(trainloader)} iterations per epoch.")

    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for _, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

            label_batch_shifted = label_batch + num_classes_old
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch_shifted.long())
            loss_dice = dice_loss(outputs, label_batch_shifted, softmax=True)
            loss = 0.4 * loss_ce + 0.6 * loss_dice

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num += 1
            writer.add_scalar('info/lr', optimizer.param_groups[0]['lr'], iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            if iter_num % 20 == 0:
                logging.info(f'iteration {iter_num} : loss : {loss.item():.4f}, loss_ce: {loss_ce.item():.4f}')
                image = image_batch[0, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                preds = torch.softmax(outputs, dim=1)
                preds_argmax = torch.argmax(preds, dim=1, keepdim=True)
                preds_new_task = torch.where(preds_argmax >= num_classes_old, preds_argmax - num_classes_old, 0)
                writer.add_image('train/Prediction_NewTask', preds_new_task[0, ...] * 50, iter_num)
                labs = label_batch[0, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth_NewTask', labs, iter_num)

        scheduler.step()

        if (epoch_num + 1) % 10 == 0 or epoch_num == max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, f'continual_epoch_{epoch_num}.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info(f"Saved model to {save_mode_path}")

    writer.close()
    return "Continual Learning Finished!"


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

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    total_num_classes = args.num_classes_old + args.num_classes_new
    net = ViT_seg(config, img_size=args.img_size, num_classes=total_num_classes).cuda()
    net.load_from(config)
    net = load_pretrained_weights_continual(net, args.pretrained_path,
                                            num_classes_old=args.num_classes_old,
                                            num_classes_new=args.num_classes_new)

    trainer_continual(args, net, args.output_dir)
