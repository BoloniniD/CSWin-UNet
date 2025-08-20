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
parser.add_argument('--root_path', type=str,
                    default='./datasets/kits23/train_npz', help='root dir for data')
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
parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
parser.add_argument('--pretrained_path', type=str, required=True,
                    help='path to pretrained model checkpoint')
parser.add_argument('--data_fraction', type=float, default=1.0,
                    help='fraction of data to use for finetuning (default: 1.0)')

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
config = get_config(args)


class ContinualLearningModel(nn.Module):
    """Wrapper model for continual learning that expands the output layer"""
    def __init__(self, base_model, num_classes_old, num_classes_new):
        super().__init__()
        self.base_model = base_model
        self.num_classes_old = num_classes_old
        self.num_classes_new = num_classes_new
        # Total classes = old + new - 1 (shared background)
        self.num_classes_total = num_classes_old + num_classes_new - 1

        # Find and replace the final classification layer
        self._expand_final_layer()

    def _expand_final_layer(self):
        """Find and expand the final classification layer"""
        # Common names for final layers in segmentation models
        final_layer_candidates = ['output', 'final', 'classifier', 'head', 'segmentation_head']

        # First, try to find by common names
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

        # If not found by name, search through all modules
        if not final_layer_found:
            final_layer_found = self._search_and_replace_final_layer()

        if not final_layer_found:
            raise RuntimeError("Could not find the final classification layer to expand")

    def _is_classification_layer(self, module):
        """Check if a module is likely the final classification layer"""
        if isinstance(module, nn.Conv2d):
            return module.out_channels == self.num_classes_old
        elif isinstance(module, nn.Linear):
            return module.out_features == self.num_classes_old
        return False

    def _create_expanded_layer(self, old_module):
        """Create an expanded version of the classification layer"""
        if isinstance(old_module, nn.Conv2d):
            # Handle Conv2d layer
            new_conv = nn.Conv2d(
                in_channels=old_module.in_channels,
                out_channels=self.num_classes_total,
                kernel_size=old_module.kernel_size,
                stride=old_module.stride,
                padding=old_module.padding,
                bias=old_module.bias is not None
            )

            # Copy old weights for existing classes
            with torch.no_grad():
                new_conv.weight[:self.num_classes_old] = old_module.weight

                # Handle bias if it exists
                if old_module.bias is not None:
                    new_conv.bias[:self.num_classes_old] = old_module.bias
                    # Initialize new class biases
                    nn.init.constant_(new_conv.bias[self.num_classes_old:], 0)

                # Initialize new class weights
                nn.init.kaiming_normal_(new_conv.weight[self.num_classes_old:])

            return new_conv

        elif isinstance(old_module, nn.Linear):
            # Handle Linear layer
            new_linear = nn.Linear(
                in_features=old_module.in_features,
                out_features=self.num_classes_total,
                bias=old_module.bias is not None
            )

            # Copy old weights for existing classes
            with torch.no_grad():
                new_linear.weight[:self.num_classes_old] = old_module.weight

                # Handle bias if it exists
                if old_module.bias is not None:
                    new_linear.bias[:self.num_classes_old] = old_module.bias
                    # Initialize new class biases
                    nn.init.constant_(new_linear.bias[self.num_classes_old:], 0)

                # Initialize new class weights
                nn.init.kaiming_normal_(new_linear.weight[self.num_classes_old:])

            return new_linear

        else:
            raise TypeError(f"Unsupported layer type for expansion: {type(old_module)}")

    def _search_and_replace_final_layer(self):
        """Search through all modules to find and replace the final classification layer"""
        for name, module in self.base_model.named_modules():
            if self._is_classification_layer(module):
                # Found potential final layer
                new_layer = self._create_expanded_layer(module)

                # Replace the module
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

    # Use limited batches for gradient calculation
    partial_loader = itertools.islice(loader, args.gradient_batches)
    xent_grads = []

    model.eval()  # Set to eval mode for gradient calculation

    # Calculate gradients for cross-entropy loss
    for batch_data in partial_loader:
        image_batch, label_batch = batch_data['image'], batch_data['label']
        image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

        outputs = model(image_batch)

        # Map new dataset labels to the expanded label space
        # New dataset classes go after old classes (except background stays at 0)
        label_batch_mapped = label_batch.clone()
        label_batch_mapped[label_batch > 0] += args.num_classes_old - 1

        loss_ce = criterion(outputs, label_batch_mapped.long())

        grad_xent = torch.autograd.grad(
            outputs=loss_ce, inputs=model.parameters(), retain_graph=True, allow_unused=True
        )
        xent_grads.append([g.detach() if g is not None else None for g in grad_xent])

    def get_grad_norms(model, grads, args):
        """Helper function to compute gradient norms"""
        _metrics = defaultdict(list)

        for (name, param), grad in zip(model.named_parameters(), grads):
            if name not in layer_names or grad is None:
                continue

            if args.auto_tune == "eb-criterion":
                # Compute evidence-based criterion
                grad_var = torch.var(grad, dim=0, keepdim=True)
                tmp = (grad * grad) / (grad_var + 1e-8)
                _metrics[name] = tmp.mean().item()
            else:  # RGN
                # Compute relative gradient norm
                param_norm = torch.norm(param).item()
                if param_norm > 1e-8:
                    _metrics[name] = torch.norm(grad).item() / param_norm
                else:
                    _metrics[name] = 0.0
        return _metrics

    # Average metrics across batches
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

    # Store layer info for logging
    layer_info = []

    if args.auto_tune == "none":
        # Standard optimizer
        return optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=base_lr, weight_decay=0.01), []

    # Create parameter groups with different learning rates
    for name, param in param_dict.items():
        if not param.requires_grad:
            continue

        if name in weights:
            lr = weights[name] * base_lr
        else:
            lr = 0.0  # Don't update parameters not in weights

        param_groups.append({
            'params': [param],
            'lr': lr,
            'name': name
        })

        # Store info for logging
        layer_info.append({
            'name': name,
            'weight': weights.get(name, 0.0),
            'lr': lr
        })

    return optim.AdamW(param_groups, lr=base_lr, weight_decay=0.01), layer_info


def log_layer_learning_rates(layer_info, args):
    """Log the learning rates assigned to each layer"""
    logging.info("\n" + "="*80)
    logging.info(f"CONTINUAL LEARNING - SURGICAL FINE-TUNING - {args.auto_tune.upper()} METHOD")
    logging.info("="*80)
    logging.info(f"{'Layer Name':<50} {'Weight':<12} {'Learning Rate':<15}")
    logging.info("-"*80)

    # Sort by learning rate (descending) to see which layers get highest LR
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


def trainer_continual_learning(args, model, old_model, snapshot_path):
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

    print(f"Using {subset_size}/{total_samples} samples ({args.data_fraction*100:.1f}%) for continual learning")
    logging.info(f"Using {subset_size}/{total_samples} samples ({args.data_fraction*100:.1f}%) for continual learning")
    logging.info(f"Old classes: {args.num_classes_old}, New classes: {args.num_classes_new}, Total: {num_classes_total}")

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                           num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    if args.n_gpu > 1:
        model = nn.DataParallel(model)
        old_model = nn.DataParallel(old_model)

    # Loss functions
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes_total)

    # Initialize optimizer
    if args.auto_tune == "none":
        optimizer, _ = create_surgical_optimizer(model, base_lr, {}, args)
    else:
        optimizer = None  # Will be created dynamically

    scheduler = None
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)

    logging.info(f"Continual Learning Configuration:")
    logging.info(f"KD Temperature: {args.kd_temperature}")
    logging.info(f"KD Weight: {args.kd_weight}")
    logging.info(f"Auto-tune method: {args.auto_tune}")
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))

    iterator = tqdm(range(max_epoch), ncols=70)

    for epoch_num in iterator:
        model.train()
        old_model.eval()  # Keep old model in eval mode

        # Surgical fine-tuning: Calculate gradient weights at the beginning of each epoch
        if args.auto_tune != "none":
            logging.info(f"\n[EPOCH {epoch_num + 1}] Calculating gradient weights for continual learning...")

            # Calculate gradient-based weights
            weights = get_lr_weights(model, trainloader, args, ce_loss, args.num_classes_new)

            if args.auto_tune == "RGN":
                # Normalize weights by maximum weight
                if weights:
                    max_weight = max(weights.values()) if weights.values() else 1.0
                    logging.info(f"RGN: Max weight before normalization: {max_weight:.6f}")
                    for k in weights:
                        weights[k] = weights[k] / max_weight if max_weight > 0 else 0.0

            elif args.auto_tune == "eb-criterion":
                # Apply threshold-based selection
                if weights:
                    threshold = 0.95
                    min_weight = min(weights.values())
                    max_weight = max(weights.values())
                    logging.info(f"EB-Criterion: Weight range before thresholding: {min_weight:.6f} - {max_weight:.6f}")

                    for k in weights:
                        weights[k] = 1.0 if weights[k] >= threshold else 0.0

            # Create new optimizer with updated weights
            optimizer, layer_info = create_surgical_optimizer(model, base_lr, weights, args)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)

            # Log detailed layer information
            log_layer_learning_rates(layer_info, args)

        for batch_idx, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

            # Forward pass through both models
            outputs = model(image_batch)
            with torch.no_grad():
                old_outputs = old_model(image_batch)

            # Map new dataset labels to the expanded label space
            # Background (0) stays 0, new classes (1,2,3) become (9,10,11)
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
            loss = (1 - args.kd_weight) * loss_seg + args.kd_weight * loss_kd

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

        if scheduler is not None:
            scheduler.step()

        # Save model periodically
        if (epoch_num + 1) % 5 == 0 or epoch_num == max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, f'continual_epoch_{epoch_num}.pth')
            if args.n_gpu > 1:
                torch.save(model.module.state_dict(), save_mode_path)
            else:
                torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))

    writer.close()
    return "Continual Learning Finished!"


def worker_init_fn(worker_id):
    random.seed(1234 + worker_id)


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

    # Create base model with old number of classes
    net = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes_old).cuda()
    net.load_from(config)

    # Load pretrained weights
    print(f"Loading pretrained model from {args.pretrained_path}")
    pretrained_dict = torch.load(args.pretrained_path, map_location='cpu')
    pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}
    net.load_state_dict(pretrained_dict, strict=True)

    # Create old model for knowledge distillation
    old_net = copy.deepcopy(net)
    old_net.eval()
    for param in old_net.parameters():
        param.requires_grad = False

    # Create continual learning model with expanded output
    cl_model = ContinualLearningModel(net, args.num_classes_old, args.num_classes_new).cuda()

    # Print continual learning configuration
    print(f"\n=== Continual Learning Configuration ===")
    print(f"Old model classes: {args.num_classes_old}")
    print(f"New dataset classes: {args.num_classes_new}")
    print(f"Total classes: {args.num_classes_old + args.num_classes_new - 1}")
    print(f"KD Temperature: {args.kd_temperature}")
    print(f"KD Weight: {args.kd_weight}")
    print(f"Auto-tune method: {args.auto_tune}")
    print(f"Data fraction: {args.data_fraction}")
    print(f"Base learning rate: {args.base_lr}")
    print(f"Max epochs: {args.max_epochs}")
    print("=" * 45)

    trainer_continual_learning(args, cl_model, old_net, args.output_dir)
