# finetune_tpgm.py
import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.vision_transformer import CSwinUnet as ViT_seg
from trainer import trainer_synapse
from config import get_config
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from utils import DiceLoss
from tqdm import tqdm
import torch.optim as optim
from torchvision import transforms
import copy
from collections import defaultdict

class temporary_parameter_replace:
    def __init__(self, model, params_dict):
        self.model = model
        self.params_dict = params_dict
        self.original_params = {}

    def __enter__(self):
        for name, param in self.model.named_parameters():
            if name in self.params_dict:
                self.original_params[name] = param.data.clone()
                param.data.copy_(self.params_dict[name])

    def __exit__(self, exc_type, exc_val, exc_tb):
        for name, param in self.model.named_parameters():
            if name in self.original_params:
                param.data.copy_(self.original_params[name])

class TPGM(nn.Module):
    def __init__(self, model, norm_mode, exclude_list=[]) -> None:
        super().__init__()
        self.norm_mode = norm_mode
        self.exclude_list = exclude_list
        self.threshold = torch.nn.Hardtanh(0,1)
        self.constraints_name = []
        self.constraints = []
        self.create_contraint(model)
        self.constraints = nn.ParameterList(self.constraints)
        self.ratio_vals = {}
        self.currently_recording = False

    def create_contraint(self, module):
        for name, para in module.named_parameters():
            if not para.requires_grad:
                continue
            if name not in self.exclude_list:
                self.constraints_name.append(name)
                temp = nn.Parameter(torch.Tensor([0.1]), requires_grad=True)
                self.constraints.append(temp)

    def apply_constraints(
        self,
        new,
        pre_trained,
        constraint_iterator,
        apply=False,
    ):
        projected_params = {}
        new_module = new.module if isinstance(new, nn.DataParallel) else new
        pre_trained_module = pre_trained.module if isinstance(pre_trained, nn.DataParallel) else pre_trained

        for (name, new_para), anchor_para in zip(
            new_module.named_parameters(), pre_trained_module.parameters()
        ):
            if not new_para.requires_grad:
                continue
            if name not in self.exclude_list:
                alpha = self._project_ratio(
                    new_para,
                    anchor_para,
                    constraint_iterator,
                    name,
                )
                v = (new_para.detach() - anchor_para.detach()) * alpha
                temp = v + anchor_para.detach()
                if apply:
                    with torch.no_grad():
                        new_para.copy_(temp)
                else:
                    projected_params[name] = temp
        if not apply:
            return projected_params

    def _project_ratio(self, new, anchor, constraint_iterator, name):
        t = new.detach() - anchor.detach()

        if "l2" in self.norm_mode:
            norms = torch.norm(t)
        else:
            norms = torch.sum(torch.abs(t))

        constraint_param = next(constraint_iterator)
        constraint = torch.clamp(constraint_param, min=1e-8, max=norms.item())
        ratio = constraint / (norms + 1e-8)
        ratio = self.threshold(ratio)

        if self.currently_recording:
            self.ratio_vals[name] = ratio.item()

        return ratio

    def forward(
        self,
        new=None,
        pre_trained=None,
        x=None,
        apply=False,
        active_classes=None,
    ):
        self.ratio_vals = {}
        self.currently_recording = not apply

        constraint_iterator = iter(self.constraints)

        if apply:
            self.apply_constraints(
                new,
                pre_trained,
                constraint_iterator,
                apply=apply,
            )
        else:
            projected_params = self.apply_constraints(new, pre_trained, constraint_iterator, apply=False)
            with temporary_parameter_replace(new, projected_params):
                out = new(x)
                if active_classes is not None:
                    out = out[:, :active_classes, :, :]
            return out

    def get_ratio_stats(self):
        """Get statistics about the projection ratios"""
        if not self.ratio_vals:
            return 0.0, 0.0, 0.0

        ratios = list(self.ratio_vals.values())
        min_ratio = min(ratios)
        max_ratio = max(ratios)
        mean_ratio = sum(ratios) / len(ratios)
        return min_ratio, max_ratio, mean_ratio

    ### MODIFICATION ###
    # This is the updated method with the correct parsing logic for your model.
    def get_per_block_ratio_stats(self):
        """Groups ratios by model block and calculates the mean for each."""
        if not self.ratio_vals:
            return {}

        block_ratios = defaultdict(list)
        for name, ratio in self.ratio_vals.items():
            parts = name.split('.')

            # Group by the first two parts of the name, e.g., 'cswin_unet.stage1'
            # This is robust for names like 'cswin_unet.norm.weight' as well.
            if len(parts) > 1:
                block_name = ".".join(parts[:2])
            else:
                # Handle cases with no dots, though unlikely for nested models
                block_name = parts[0]

            block_ratios[block_name].append(ratio)

        # Calculate the mean for each block
        mean_block_ratios = {name: sum(ratios) / len(ratios) for name, ratios in block_ratios.items()}
        return mean_block_ratios


class tpgm_trainer(object):
    def __init__(
        self,
        model,
        pgmloader,
        norm_mode,
        proj_lr,
        max_iters,
        ce_loss,
        dice_loss,
        snapshot_path,
        active_classes=None,
        exclude_list = []
    ) -> None:
        self.device = torch.device("cuda")
        self.proj_lr = proj_lr
        self.max_iters = max_iters
        self.active_classes = active_classes
        self.tpgm = TPGM(model, norm_mode=norm_mode, exclude_list=exclude_list).to(self.device)

        if isinstance(model, nn.DataParallel):
            self.pre_trained = copy.deepcopy(model.module)
        else:
            self.pre_trained = copy.deepcopy(model)

        self.pgm_optimizer = torch.optim.Adam(self.tpgm.parameters(), lr=self.proj_lr)
        self.pgmloader = pgmloader
        self.dataset_iterator = iter(self.pgmloader)
        self.ce_loss = ce_loss
        self.dice_loss = dice_loss
        self.ratio_stats = []

        self.ratio_logger = logging.getLogger('tpgm_ratios')
        self.ratio_logger.setLevel(logging.INFO)
        self.ratio_logger.propagate = False
        handler = logging.FileHandler(os.path.join(snapshot_path, "tpgm_ratios.log"), mode='w')
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        handler.setFormatter(formatter)
        if (self.ratio_logger.hasHandlers()):
            self.ratio_logger.handlers.clear()
        self.ratio_logger.addHandler(handler)
        self.ratio_logger.info("TPGM Ratio Logger Initialized.")


    def tpgm_iters(self, model, apply=False):
        if not apply:
            self.count = 0
            while self.count < self.max_iters:
                try:
                    data = next(self.dataset_iterator)
                except StopIteration:
                    self.dataset_iterator = iter(self.pgmloader)
                    data = next(self.dataset_iterator)

                pgm_image = data['image'].to(self.device)
                pgm_target = data['label'].to(self.device)

                outputs = self.tpgm(model, self.pre_trained, x=pgm_image, active_classes=self.active_classes)

                loss_ce = self.ce_loss(outputs, pgm_target[:].long())
                loss_dice = self.dice_loss(outputs, pgm_target, softmax=True)
                pgm_loss = 0.4 * loss_ce + 0.6 * loss_dice

                self.pgm_optimizer.zero_grad()
                pgm_loss.backward()
                self.pgm_optimizer.step()
                self.count += 1

                if (self.count+1) % 20 == 0:
                    min_ratio, max_ratio, mean_ratio = self.tpgm.get_ratio_stats()
                    per_block_stats = self.tpgm.get_per_block_ratio_stats()

                    print(f"{self.count}/{self.max_iters} TPGM iterations completed. See tpgm_ratios.log for details.")

                    self.ratio_logger.info(f"--- Iteration {self.count}/{self.max_iters} ---")
                    self.ratio_logger.info(f"Global Projection Ratios - Min: {min_ratio:.4f}, Max: {max_ratio:.4f}, Mean: {mean_ratio:.4f}")
                    self.ratio_logger.info("Mean Ratios per Model Block:")
                    # Increased padding for longer names
                    for block_name in sorted(per_block_stats.keys()):
                        self.ratio_logger.info(f"  - {block_name:<40}: {per_block_stats[block_name]:.4f}")
                    self.ratio_logger.info("-" * (20 + len(str(self.count)) + len(str(self.max_iters))))


        self.tpgm(model, self.pre_trained, apply=True)


# Argument parser setup
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/kits23/train_npz', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='kits23', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/kits23', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=4, help='output channel of network')
parser.add_argument('--model_num_classes', type=int,
                    default=9, help='total number of classes in the model')
parser.add_argument('--output_dir', type=str, help='output dir')
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
# TPGM specific arguments
parser.add_argument('--tpgm_norm_mode', type=str, default='l2', choices=['l2', 'mars'],
                   help='Norm mode for TPGM: l2 or mars')
parser.add_argument('--tpgm_lr', type=float, default=0.001,
                   help='Learning rate for TPGM constraints')
parser.add_argument('--tpgm_iters', type=int, default=100,
                   help='Number of TPGM iterations per epoch')
parser.add_argument('--tpgm_exclude', nargs='+', default=[],
                   help='List of layer names to exclude from TPGM projection')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use')

args = parser.parse_args()
config = get_config(args)

def load_pretrained_weights(model, pretrained_path):
    """Load pretrained weights without modifying output layers"""
    print(f"Loading pretrained model from {pretrained_path}")
    pretrained_dict = torch.load(pretrained_path, map_location='cpu')
    pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}
    pretrained_dict = {k.replace('base_model.', ''): v for k, v in pretrained_dict.items()}
    model_dict = model.state_dict()

    model.load_state_dict(pretrained_dict, strict=True)
    print(f"Loaded full pretrained model with all {len(pretrained_dict)} layers")
    return model

def freeze_layers(model, num_layers_to_freeze):
    """Freeze specified number of transformer layers"""
    if num_layers_to_freeze == 0:
        return

    for param in model.patch_embed.parameters():
        param.requires_grad = False

    if hasattr(model, 'absolute_pos_embed'):
        model.absolute_pos_embed.requires_grad = False

    layer_count = 0
    for name, module in model.named_modules():
        if 'layers' in name and 'block' in name:
            if layer_count < num_layers_to_freeze:
                for param in module.parameters():
                    param.requires_grad = False
                layer_count += 1
                print(f"Frozen layer: {name}")

    print(f"Total frozen layers: {layer_count}")

def worker_init_fn(worker_id):
    random.seed(1234 + worker_id)

def trainer_finetune(args, model, snapshot_path):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    from torch.utils.data import Subset

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    base_lr = args.base_lr
    num_classes = args.num_classes
    model_num_classes = args.model_num_classes
    batch_size = args.batch_size * args.n_gpu

    db_train_full = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                                    transform=transforms.Compose(
                                        [RandomGenerator(output_size=[args.img_size, args.img_size])]),
                                    is_kits=True)

    total_samples = len(db_train_full)
    total_fraction = int(total_samples * args.data_fraction)

    finetune_size = int(total_fraction * 0.9)
    tpgm_size = total_fraction - finetune_size

    indices = list(range(total_samples))
    random.shuffle(indices)
    finetune_indices = indices[:finetune_size]
    tpgm_indices = indices[finetune_size:finetune_size + tpgm_size]

    db_train = Subset(db_train_full, finetune_indices)
    db_tpgm = Subset(db_train_full, tpgm_indices)

    print(f"Using {finetune_size}/{total_samples} samples ({finetune_size/total_samples*100:.1f}%) for finetuning")
    print(f"Using {tpgm_size}/{total_samples} samples ({tpgm_size/total_samples*100:.1f}%) for TPGM")
    logging.info(f"Using {finetune_size}/{total_samples} samples for finetuning")
    logging.info(f"Using {tpgm_size}/{total_samples} samples for TPGM")
    logging.info(f"Model has {model_num_classes} total classes, training on {num_classes} classes")

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                           num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    tpgm_loader = DataLoader(db_tpgm, batch_size=batch_size, shuffle=True,
                           num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=base_lr, weight_decay=0.01)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)

    tpgm = tpgm_trainer(
        model=model,
        pgmloader=tpgm_loader,
        norm_mode=args.tpgm_norm_mode,
        proj_lr=args.tpgm_lr,
        max_iters=args.tpgm_iters,
        ce_loss=ce_loss,
        dice_loss=dice_loss,
        snapshot_path=snapshot_path,
        active_classes=num_classes,
        exclude_list=args.tpgm_exclude
    )

    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))

    iterator = tqdm(range(max_epoch), ncols=70)

    for epoch_num in iterator:
        model.train()
        for _, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

            outputs = model(image_batch)
            outputs_active = outputs[:, :num_classes, :, :]

            loss_ce = ce_loss(outputs_active, label_batch[:].long())
            loss_dice = dice_loss(outputs_active, label_batch, softmax=True)
            loss = 0.4 * loss_ce + 0.6 * loss_dice

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', optimizer.param_groups[0]['lr'], iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            if iter_num % 10 == 0:
                logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))

            if iter_num % 20 == 0:
                image = image_batch[0, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs_vis = torch.argmax(torch.softmax(outputs_active, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs_vis[0, ...] * 50, iter_num)
                labs = label_batch[0, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        print(f"Running TPGM projection after epoch {epoch_num}")
        tpgm.tpgm_iters(model, apply=False)
        scheduler.step()

        if (epoch_num + 1) % 10 == 0 or epoch_num == max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'finetuned_epoch_' + str(epoch_num) + '.pth')
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), save_mode_path)
            else:
                torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

    print("Applying final TPGM projection")
    tpgm.tpgm_iters(model, apply=True)

    save_mode_path = os.path.join(snapshot_path, 'finetuned_final.pth')
    if isinstance(model, nn.DataParallel):
        torch.save(model.module.state_dict(), save_mode_path)
    else:
        torch.save(model.state_dict(), save_mode_path)
    logging.info("save final model to {}".format(save_mode_path))

    writer.close()
    return "Finetuning Finished!"

if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    torch.cuda.set_device(args.gpu_id)
    device = torch.device(f'cuda:{args.gpu_id}')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    args.dataset = 'kits23'
    args.num_classes = 4
    args.model_num_classes = 9

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    net = ViT_seg(config, img_size=args.img_size, num_classes=args.model_num_classes).cuda()
    net.load_from(config)
    net = load_pretrained_weights(net, args.pretrained_path)

    if args.freeze_layers > 0:
        freeze_layers(net, args.freeze_layers)

    trainer_finetune(args, net, args.output_dir)
