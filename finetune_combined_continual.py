# finetune_surgical_tpgm_continual.py
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
from collections import defaultdict
import torch.nn.functional as F
import itertools
import copy

parser = argparse.ArgumentParser()
# Data and model arguments
parser.add_argument('--root_path', type=str,
                    default='../data/kits23/train_npz', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='kits23', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/kits23', help='list dir')
parser.add_argument('--num_classes_old', type=int,
                    default=9, help='number of classes in the old model')
parser.add_argument('--num_classes_new', type=int,
                    default=4, help='number of classes in the new dataset')
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
parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file')
parser.add_argument('--pretrained_path', type=str, required=True,
                    help='path to pretrained model checkpoint')
parser.add_argument('--data_fraction', type=float, default=1.0,
                    help='fraction of data to use for finetuning (default: 1.0)')
parser.add_argument('--freeze_layers', type=int, default=0,
                    help='number of transformer layers to freeze (0 = no freezing)')

# Continual learning arguments
parser.add_argument('--kd_temperature', type=float, default=3.0,
                    help='temperature for knowledge distillation')
parser.add_argument('--kd_weight', type=float, default=0.5,
                    help='weight for knowledge distillation loss')
parser.add_argument('--freeze_old_classes', action='store_true',
                    help='freeze weights for old class predictions')

# Surgical fine-tuning arguments
parser.add_argument('--auto_tune', type=str, default='RGN',
                    choices=['none', 'RGN', 'eb-criterion'],
                    help='Auto-tuning method for surgical fine-tuning')
parser.add_argument('--gradient_batches', type=int, default=5,
                    help='Number of batches to use for gradient analysis')

# TPGM specific arguments
parser.add_argument('--tpgm_norm_mode', type=str, default='l2', choices=['l2', 'mars'],
                   help='Norm mode for TPGM: l2 or mars')
parser.add_argument('--tpgm_lr', type=float, default=0.001,
                   help='Learning rate for TPGM constraints')
parser.add_argument('--tpgm_iters', type=int, default=100,
                   help='Number of TPGM iterations per epoch')
parser.add_argument('--tpgm_exclude', nargs='+', default=[],
                   help='List of layer names to exclude from TPGM projection')
parser.add_argument('--tpgm_weight', type=float, default=0.3,
                   help='Weight for TPGM in combined approach')
parser.add_argument('--tpgm_data_fraction', type=float, default=0.1,
                   help='Fraction of training data to use for TPGM projection')

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

parser.add_argument('--num_classes', type=int,
                    default=4, help='number of classes in the new dataset')
parser.add_argument('--model_num_classes', type=int,
                    default=9, help='total number of classes in the model (old classes)')

parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use')

args = parser.parse_args()
config = get_config(args)


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
        self.ratio_vals = []
        self.currently_recording = False

    def create_contraint(self, module):
        for name, para in module.named_parameters():
            if not para.requires_grad:
                continue
            if name not in self.exclude_list:
                self.constraints_name.append(name)
                temp = nn.Parameter(torch.Tensor([0.1]), requires_grad=True)
                self.constraints.append(temp)

    def apply_constraints(self, new, pre_trained, constraint_iterator, apply=False):
        projected_params = {}
        new_module = new.module if isinstance(new, nn.DataParallel) else new
        pre_trained_module = pre_trained.module if isinstance(pre_trained, nn.DataParallel) else pre_trained

        for (name, new_para), anchor_para in zip(
            new_module.named_parameters(), pre_trained_module.parameters()
        ):
            if not new_para.requires_grad:
                continue
            if name not in self.exclude_list:
                alpha = self._project_ratio(new_para, anchor_para, constraint_iterator)
                v = (new_para.detach() - anchor_para.detach()) * alpha
                temp = v + anchor_para.detach()
                if apply:
                    with torch.no_grad():
                        new_para.copy_(temp)
                else:
                    projected_params[name] = temp
        if not apply:
            return projected_params

    def _project_ratio(self, new, anchor, constraint_iterator):
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
            self.ratio_vals.append(ratio.item())

        return ratio

    def forward(self, new=None, pre_trained=None, x=None, apply=False, active_classes=None):
        self.ratio_vals = []
        self.currently_recording = not apply
        constraint_iterator = iter(self.constraints)

        if apply:
            self.apply_constraints(new, pre_trained, constraint_iterator, apply=apply)
        else:
            projected_params = self.apply_constraints(new, pre_trained, constraint_iterator, apply=False)
            with temporary_parameter_replace(new, projected_params):
                out = new(x)
                if active_classes is not None:
                    out = out[:, :active_classes, :, :]
            return out

    def get_ratio_stats(self):
        if not self.ratio_vals:
            return 0.0, 0.0, 0.0
        min_ratio = min(self.ratio_vals)
        max_ratio = max(self.ratio_vals)
        mean_ratio = sum(self.ratio_vals) / len(self.ratio_vals)
        return min_ratio, max_ratio, mean_ratio


class ContinualLearningModel(nn.Module):
    """Wrapper model for continual learning that expands the output layer"""
    def __init__(self, base_model, num_classes_old, num_classes_new):
        super().__init__()
        self.base_model = base_model
        self.num_classes_old = num_classes_old
        self.num_classes_new = num_classes_new
        self.num_classes_total = num_classes_old + num_classes_new - 1
        self._expand_final_layer()

    def _expand_final_layer(self):
        final_layer_candidates = ['output', 'final', 'classifier', 'head', 'segmentation_head']
        final_layer_found = False

        for candidate_name in final_layer_candidates:
            if hasattr(self.base_model, candidate_name):
                module = getattr(self.base_model, candidate_name)
                if self._is_classification_layer(module):
                    new_layer = self._create_expanded_layer(module)
                    setattr(self.base_model, candidate_name, new_layer)
                    print(f"Expanded final layer '{candidate_name}': {self.num_classes_old} -> {self.num_classes_total} classes")
                    final_layer_found = True
                    break

        if not final_layer_found:
            final_layer_found = self._search_and_replace_final_layer()

        if not final_layer_found:
            raise RuntimeError("Could not find the final classification layer to expand")

    def _is_classification_layer(self, module):
        if isinstance(module, nn.Conv2d):
            return module.out_channels == self.num_classes_old
        elif isinstance(module, nn.Linear):
            return module.out_features == self.num_classes_old
        return False

    def _create_expanded_layer(self, old_module):
        if isinstance(old_module, nn.Conv2d):
            new_conv = nn.Conv2d(
                in_channels=old_module.in_channels,
                out_channels=self.num_classes_total,
                kernel_size=old_module.kernel_size,
                stride=old_module.stride,
                padding=old_module.padding,
                bias=old_module.bias is not None
            )

            with torch.no_grad():
                new_conv.weight[:self.num_classes_old] = old_module.weight
                if old_module.bias is not None:
                    new_conv.bias[:self.num_classes_old] = old_module.bias
                    nn.init.constant_(new_conv.bias[self.num_classes_old:], 0)
                nn.init.kaiming_normal_(new_conv.weight[self.num_classes_old:])
            return new_conv

        elif isinstance(old_module, nn.Linear):
            new_linear = nn.Linear(
                in_features=old_module.in_features,
                out_features=self.num_classes_total,
                bias=old_module.bias is not None
            )

            with torch.no_grad():
                new_linear.weight[:self.num_classes_old] = old_module.weight
                if old_module.bias is not None:
                    new_linear.bias[:self.num_classes_old] = old_module.bias
                    nn.init.constant_(new_linear.bias[self.num_classes_old:], 0)
                nn.init.kaiming_normal_(new_linear.weight[self.num_classes_old:])
            return new_linear
        else:
            raise TypeError(f"Unsupported layer type for expansion: {type(old_module)}")

    def _search_and_replace_final_layer(self):
        for name, module in self.base_model.named_modules():
            if self._is_classification_layer(module):
                new_layer = self._create_expanded_layer(module)
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]

                if parent_name:
                    parent = self.base_model
                    for part in parent_name.split('.'):
                        parent = getattr(parent, part)
                    setattr(parent, child_name, new_layer)
                else:
                    setattr(self.base_model, child_name, new_layer)

                print(f"Expanded final layer '{name}': {self.num_classes_old} -> {self.num_classes_total} classes")
                return True
        return False

    def forward(self, x):
        return self.base_model(x)


class tpgm_trainer(object):
    def __init__(self, model, pgmloader, norm_mode, proj_lr, max_iters, ce_loss, dice_loss,
                 active_classes=None, exclude_list=[]):
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

                # Map labels for continual learning
                pgm_target_mapped = pgm_target.clone()
                pgm_target_mapped[pgm_target > 0] += args.num_classes_old - 1

                outputs = self.tpgm(model, self.pre_trained, x=pgm_image, active_classes=self.active_classes)

                min_ratio, max_ratio, mean_ratio = self.tpgm.get_ratio_stats()
                self.ratio_stats.append((min_ratio, max_ratio, mean_ratio))

                loss_ce = self.ce_loss(outputs, pgm_target_mapped[:].long())
                loss_dice = self.dice_loss(outputs, pgm_target_mapped, softmax=True)
                pgm_loss = 0.4 * loss_ce + 0.6 * loss_dice

                self.pgm_optimizer.zero_grad()
                pgm_loss.backward()
                self.pgm_optimizer.step()
                self.count += 1

                if (self.count+1) % 20 == 0:
                    print("{}/{} TPGM iterations completed".format(self.count, self.max_iters))
                    print(f"  Projection ratios - Min: {min_ratio:.4f}, Max: {max_ratio:.4f}, Mean: {mean_ratio:.4f}")

        self.tpgm(model, self.pre_trained, apply=True)


def knowledge_distillation_loss(outputs, old_outputs, temperature=3.0):
    """Calculate knowledge distillation loss"""
    log_p = F.log_softmax(outputs / temperature, dim=1)
    q = F.softmax(old_outputs / temperature, dim=1)
    kd_loss = F.kl_div(log_p, q, reduction='batchmean') * (temperature ** 2)
    return kd_loss


def get_layer_names(model):
    """Get names of non-batch-norm layers for gradient analysis"""
    layer_names = []
    for name, _ in model.named_parameters():
        if "bn" not in name.lower() and "norm" not in name.lower():
            layer_names.append(name)
    return layer_names


def get_lr_weights(model, loader, args, criterion, num_classes_new):
    """Calculate learning rate weights for different layers based on gradients"""
    layer_names = get_layer_names(model)
    metrics = defaultdict(list)
    average_metrics = defaultdict(float)

    partial_loader = itertools.islice(loader, args.gradient_batches)
    xent_grads = []

    model.eval()

    for batch_data in partial_loader:
        image_batch, label_batch = batch_data['image'], batch_data['label']
        image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

        outputs = model(image_batch)
        label_batch_mapped = label_batch.clone()
        label_batch_mapped[label_batch > 0] += args.num_classes_old - 1

        loss_ce = criterion(outputs, label_batch_mapped.long())
        grad_xent = torch.autograd.grad(
            outputs=loss_ce, inputs=model.parameters(), retain_graph=True, allow_unused=True
        )
        xent_grads.append([g.detach() if g is not None else None for g in grad_xent])

    def get_grad_norms(model, grads, args):
        _metrics = defaultdict(list)
        for (name, param), grad in zip(model.named_parameters(), grads):
            if name not in layer_names or grad is None:
                continue

            if args.auto_tune == "eb-criterion":
                grad_var = torch.var(grad, dim=0, keepdim=True)
                tmp = (grad * grad) / (grad_var + 1e-8)
                _metrics[name] = tmp.mean().item()
            else:  # RGN
                param_norm = torch.norm(param).item()
                if param_norm > 1e-8:
                    _metrics[name] = torch.norm(grad).item() / param_norm
                else:
                    _metrics[name] = 0.0
        return _metrics

    for xent_grad in xent_grads:
        xent_grad_metrics = get_grad_norms(model, xent_grad, args)
        for k, v in xent_grad_metrics.items():
            metrics[k].append(v)

    for k, v in metrics.items():
        if len(v) > 0:
            average_metrics[k] = np.array(v).mean(0)

    return average_metrics


def create_surgical_optimizer(model, base_lr, weights, args):
    """Create optimizer with different learning rates for different layers"""
    param_groups = []
    param_dict = dict(model.named_parameters())
    layer_info = []

    if args.auto_tune == "none":
        return optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=base_lr, weight_decay=0.01), []

    for name, param in param_dict.items():
        if not param.requires_grad:
            continue

        if name in weights:
            lr = weights[name] * base_lr
        else:
            lr = 0.0

        param_groups.append({
            'params': [param],
            'lr': lr,
            'name': name
        })

        layer_info.append({
            'name': name,
            'weight': weights.get(name, 0.0),
            'lr': lr
        })

    return optim.AdamW(param_groups, lr=base_lr, weight_decay=0.01), layer_info


def log_layer_learning_rates(layer_info, args):
    """Log the learning rates assigned to each layer"""
    logging.info("\n" + "="*80)
    logging.info(f"SURGICAL + TPGM CONTINUAL LEARNING - {args.auto_tune.upper()} METHOD")
    logging.info("="*80)
    logging.info(f"{'Layer Name':<50} {'Weight':<12} {'Learning Rate':<15}")
    logging.info("-"*80)

    sorted_layers = sorted(layer_info, key=lambda x: x['lr'], reverse=True)
    active_layers = 0

    for layer in sorted_layers:
        status = "ACTIVE" if layer['lr'] > 0 else "FROZEN"
        if layer['lr'] > 0:
            active_layers += 1
        logging.info(f"{layer['name']:<50} {layer['weight']:<12.6f} {layer['lr']:<15.8f} [{status}]")

    logging.info("-"*80)
    logging.info(f"Total layers: {len(layer_info)}, Active layers: {active_layers}, Frozen layers: {len(layer_info) - active_layers}")
    logging.info("="*80 + "\n")


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


def trainer_surgical_tpgm_continual(args, model, old_model, snapshot_path):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    from torch.utils.data import Subset

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    base_lr = args.base_lr
    num_classes_total = args.num_classes_old + args.num_classes_new - 1
    batch_size = args.batch_size * args.n_gpu

    # Load dataset
    db_train_full = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                                    transform=transforms.Compose(
                                        [RandomGenerator(output_size=[args.img_size, args.img_size])]),
                                    is_kits=True)

    total_samples = len(db_train_full)
    subset_size = int(total_samples * args.data_fraction)

    if args.data_fraction < 1.0:
        random.seed(args.seed)
        indices = random.sample(range(total_samples), subset_size)
        db_train = Subset(db_train_full, indices)
    else:
        db_train = db_train_full
        subset_size = total_samples

    # Split data for TPGM
    tpgm_size = int(subset_size * args.tpgm_data_fraction)
    train_size = subset_size - tpgm_size

    if args.data_fraction < 1.0:
        train_indices = indices[:train_size]
        tpgm_indices = indices[train_size:train_size + tpgm_size]
        db_train = Subset(db_train_full, train_indices)
        db_tpgm = Subset(db_train_full, tpgm_indices)
    else:
        all_indices = list(range(total_samples))
        random.shuffle(all_indices)
        train_indices = all_indices[:train_size]
        tpgm_indices = all_indices[train_size:train_size + tpgm_size]
        db_train = Subset(db_train_full, train_indices)
        db_tpgm = Subset(db_train_full, tpgm_indices)

    print(f"Using {train_size}/{total_samples} samples for training")
    print(f"Using {tpgm_size}/{total_samples} samples for TPGM")
    logging.info(f"Training samples: {train_size}, TPGM samples: {tpgm_size}")
    logging.info(f"Old classes: {args.num_classes_old}, New classes: {args.num_classes_new}, Total: {num_classes_total}")

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                           num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    tpgm_loader = DataLoader(db_tpgm, batch_size=batch_size, shuffle=True,
                           num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    if args.n_gpu > 1:
        model = nn.DataParallel(model)
        old_model = nn.DataParallel(old_model)

    # Loss functions
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes_total)

    # Initialize TPGM trainer
    tpgm = tpgm_trainer(
        model=model,
        pgmloader=tpgm_loader,
        norm_mode=args.tpgm_norm_mode,
        proj_lr=args.tpgm_lr,
        max_iters=args.tpgm_iters,
        ce_loss=ce_loss,
        dice_loss=dice_loss,
        active_classes=num_classes_total,
        exclude_list=args.tpgm_exclude
    )

    # Initialize optimizer
    if args.auto_tune == "none":
        optimizer, _ = create_surgical_optimizer(model, base_lr, {}, args)
    else:
        optimizer = None

    scheduler = None
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)

    logging.info(f"Combined Surgical + TPGM Continual Learning Configuration:")
    logging.info(f"KD Temperature: {args.kd_temperature}")
    logging.info(f"KD Weight: {args.kd_weight}")
    logging.info(f"TPGM Weight: {args.tpgm_weight}")
    logging.info(f"Auto-tune method: {args.auto_tune}")
    logging.info(f"TPGM norm mode: {args.tpgm_norm_mode}")
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))

    iterator = tqdm(range(max_epoch), ncols=70)

    for epoch_num in iterator:
        model.train()
        old_model.eval()

        # Surgical fine-tuning: Calculate gradient weights
        if args.auto_tune != "none":
            logging.info(f"\n[EPOCH {epoch_num + 1}] Calculating gradient weights for surgical fine-tuning...")
            weights = get_lr_weights(model, trainloader, args, ce_loss, args.num_classes_new)

            if args.auto_tune == "RGN":
                if weights:
                    max_weight = max(weights.values()) if weights.values() else 1.0
                    logging.info(f"RGN: Max weight before normalization: {max_weight:.6f}")
                    for k in weights:
                        weights[k] = weights[k] / max_weight if max_weight > 0 else 0.0

            elif args.auto_tune == "eb-criterion":
                if weights:
                    threshold = 0.95
                    min_weight = min(weights.values())
                    max_weight = max(weights.values())
                    logging.info(f"EB-Criterion: Weight range before thresholding: {min_weight:.6f} - {max_weight:.6f}")
                    for k in weights:
                        weights[k] = 1.0 if weights[k] >= threshold else 0.0

            optimizer, layer_info = create_surgical_optimizer(model, base_lr, weights, args)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
            log_layer_learning_rates(layer_info, args)

        # Training loop
        for batch_idx, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

            # Forward pass through both models
            outputs = model(image_batch)
            with torch.no_grad():
                old_outputs = old_model(image_batch)

            # Map new dataset labels to the expanded label space
            label_batch_mapped = label_batch.clone()
            label_batch_mapped[label_batch > 0] += args.num_classes_old - 1

            # Calculate losses
            loss_ce = ce_loss(outputs, label_batch_mapped.long())
            loss_dice = dice_loss(outputs, label_batch_mapped, softmax=True)

            # Knowledge distillation loss (only for old classes)
            loss_kd = knowledge_distillation_loss(
                outputs[:, :args.num_classes_old],
                old_outputs,
                temperature=args.kd_temperature
            )

            # Combined loss
            loss_seg = 0.4 * loss_ce + 0.6 * loss_dice
            loss = (1 - args.kd_weight - args.tpgm_weight) * loss_seg + args.kd_weight * loss_kd

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', optimizer.param_groups[0]['lr'], iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)
            writer.add_scalar('info/loss_kd', loss_kd, iter_num)

            if iter_num % 10 == 0:
                logging.info('iteration %d : loss : %f, loss_ce: %f, loss_kd: %f' %
                           (iter_num, loss.item(), loss_ce.item(), loss_kd.item()))

            if iter_num % 20 == 0:
                image = image_batch[0, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs_vis = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs_vis[0, ...] * 20, iter_num)
                labs = label_batch_mapped[0, ...].unsqueeze(0) * 20
                writer.add_image('train/GroundTruth', labs, iter_num)

        # Apply TPGM projection after each epoch
        if args.tpgm_weight > 0:
            print(f"Running TPGM projection after epoch {epoch_num}")
            tpgm.tpgm_iters(model, apply=False)

        if scheduler is not None:
            scheduler.step()

        # Save model periodically
        if (epoch_num + 1) % 5 == 0 or epoch_num == max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, f'surgical_tpgm_continual_epoch_{epoch_num}.pth')
            if args.n_gpu > 1:
                torch.save(model.module.state_dict(), save_mode_path)
            else:
                torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

    # Apply final TPGM projection
    if args.tpgm_weight > 0:
        print("Applying final TPGM projection")
        tpgm.tpgm_iters(model, apply=True)

    writer.close()
    return "Surgical + TPGM Continual Learning Finished!"


def worker_init_fn(worker_id):
    random.seed(1234 + worker_id)


if __name__ == "__main__":
    torch.cuda.set_device(args.gpu_id)
    device = torch.device(f'cuda:{args.gpu_id}')

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

    # Create base model with old number of classes
    net = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes_old).cuda()
    net.load_from(config)

    # Load pretrained weights
    print(f"Loading pretrained model from {args.pretrained_path}")
    pretrained_dict = torch.load(args.pretrained_path, map_location='cpu')
    pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}
    net.load_state_dict(pretrained_dict, strict=True)

    # Apply layer freezing if specified
    if args.freeze_layers > 0:
        freeze_layers(net, args.freeze_layers)

    # Create old model for knowledge distillation
    old_net = copy.deepcopy(net)
    old_net.eval()
    for param in old_net.parameters():
        param.requires_grad = False

    # Create continual learning model with expanded output
    cl_model = ContinualLearningModel(net, args.num_classes_old, args.num_classes_new).cuda()

    # Print configuration
    print(f"\n=== Surgical + TPGM Continual Learning Configuration ===")
    print(f"Old model classes: {args.num_classes_old}")
    print(f"New dataset classes: {args.num_classes_new}")
    print(f"Total classes: {args.num_classes_old + args.num_classes_new - 1}")
    print(f"KD Temperature: {args.kd_temperature}")
    print(f"KD Weight: {args.kd_weight}")
    print(f"TPGM Weight: {args.tpgm_weight}")
    print(f"Auto-tune method: {args.auto_tune}")
    print(f"TPGM norm mode: {args.tpgm_norm_mode}")
    print(f"Data fraction: {args.data_fraction}")
    print(f"TPGM data fraction: {args.tpgm_data_fraction}")
    print(f"Base learning rate: {args.base_lr}")
    print(f"TPGM learning rate: {args.tpgm_lr}")
    print(f"Max epochs: {args.max_epochs}")
    print(f"Frozen layers: {args.freeze_layers}")
    print("=" * 55)

    trainer_surgical_tpgm_continual(args, cl_model, old_net, args.output_dir)
