# finetune_surgical.py
import argparse
from loguru import logger
import os
import random
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module='timm.models.layers')
import torch
import torch.backends.cudnn as cudnn
from collections import defaultdict
from networks.vision_transformer import CSwinUnet as ViT_seg
from trainer import trainer_synapse
from config import get_config
from torch import nn, optim
from utils import DiceLoss
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import copy
from torch.nn import CrossEntropyLoss


def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_nested_parameters(module):
    """Get parameters from nested modules"""
    params = []
    for submodule in module:
        if isinstance(submodule, nn.ModuleList):
            for layer in submodule:
                params += list(layer.parameters())
        else:
            params += list(submodule.parameters())
    return params


def get_lr_weights(model, dataloader, criterion, device, num_batches=5):
    model.train()
    grad_norms = defaultdict(float)

    for _ in range(num_batches):
        model.zero_grad()
        sample = next(iter(dataloader))
        if 'label' in sample:
            images, labels = sample['image'].to(device), sample['label'].to(device)
        else:
            images, labels = sample['image'].to(device), sample['segmentation'].to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        current_param_groups = get_parameter_groups(model, 0)
        for name, params in current_param_groups.items():
            grad_norm = 0
            for p in params:
                if p.grad is not None:
                    grad_norm += p.grad.data.norm(2).item() ** 2
            grad_norms[name] += grad_norm ** 0.5

    total_norm = sum(grad_norms.values())
    return {k: v/total_norm for k, v in grad_norms.items()}

def get_nested_params(module):
    """Get parameters from nested ModuleLists"""
    params = []
    if isinstance(module, nn.ModuleList):
        for layer in module:
            params += list(layer.parameters())
    else:
        params += list(module.parameters())
    return params

def get_parameter_groups(model, _):
    """Parameter groups based on the actual CSWin-UNet architecture"""
    cswin = model.cswin_unet

    # Create comprehensive parameter groups for surgical fine-tuning
    return {
        'stem': list(cswin.stage1_conv_embed.parameters()),

        # Encoder blocks
        'encoder1': get_nested_params(cswin.stage1),
        'merge1': list(cswin.merge1.parameters()),
        'encoder2': get_nested_params(cswin.stage2),
        'merge2': list(cswin.merge2.parameters()),
        'encoder3': get_nested_params(cswin.stage3),
        'merge3': list(cswin.merge3.parameters()),
        'encoder4': get_nested_params(cswin.stage4),
        'bottleneck': list(cswin.norm.parameters()),

        # Decoder blocks
        'decoder4': get_nested_params(cswin.stage_up4),
        'upsample4': list(cswin.upsample4.parameters()),
        'concat4': list(cswin.concat_linear4.parameters()),

        'decoder3': get_nested_params(cswin.stage_up3),
        'upsample3': list(cswin.upsample3.parameters()),
        'concat3': list(cswin.concat_linear3.parameters()),

        'decoder2': get_nested_params(cswin.stage_up2),
        'upsample2': list(cswin.upsample2.parameters()),
        'concat2': list(cswin.concat_linear2.parameters()),

        'decoder1': get_nested_params(cswin.stage_up1),
        'upsample1': list(cswin.upsample1.parameters()),

        # Final layers
        'norm_up': list(cswin.norm_up.parameters()),
        'output': list(cswin.output.parameters())
    }

def get_lr_weights(model, dataloader, criterion, current_params, device):
    """Calculate relative gradient norms for each parameter group"""
    model.train()
    model.zero_grad()

    # Get a batch of data
    sample = next(iter(dataloader))
    if 'label' in sample:
        images, labels = sample['image'].to(device), sample['label'].to(device)
    else:
        images, labels = sample['image'].to(device), sample['segmentation'].to(device)

    # Forward + backward pass
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()

    # Calculate gradient norms
    grad_norms = {}
    for name, params in current_params.items():
        grad_norm = 0
        for p in params:
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item() ** 2
        grad_norms[name] = grad_norm ** 0.5

    # Calculate relative weights
    total_norm = sum(grad_norms.values())
    return {k: v/total_norm for k, v in grad_norms.items()}

def surgical_trainer(args, model, snapshot_path):
    """Modified training loop with surgical fine-tuning"""
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    logger.info(str(args))

    # Initialize surgical tuning parameters
    lr_wd_grid = [
        (1e-3, 1e-4),
        (1e-4, 1e-4),
        (1e-5, 1e-4)
    ]

    tune_metrics = defaultdict(list)
    param_groups = get_parameter_groups(model,0)

    full_db = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                             transform=transforms.Compose(
                                 [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    
    # Create 1/10 subset
    dataset_size = len(full_db)
    subset_size = dataset_size // 5
    indices = torch.randperm(dataset_size, generator=torch.Generator().manual_seed(args.seed))[:subset_size]
    db_train = torch.utils.data.Subset(full_db, indices)

    print(f"Using subset of {len(db_train)} samples (1/10 of original {dataset_size})")

    # Rest of the code remains the same
    trainloader = DataLoader(db_train, batch_size=24, shuffle=True,
                            num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)

    for lr, wd in lr_wd_grid:
        # Clone original model
        orig_model = copy.deepcopy(model)
        model = model.cuda()

        # Initialize optimizer with all parameters
        opt = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

        # Training setup
        ce_loss = CrossEntropyLoss()
        dice_loss = DiceLoss(args.num_classes)

        # Surgical tuning loop
        for epoch in range(args.max_epochs):
            logger.info(f"Epoch: {epoch}")
            model.train()
            for i_batch, sampled_batch in enumerate(trainloader):
                # Forward pass
                if 'label' in sampled_batch:
                    images, labels = sampled_batch['image'].to(device), sampled_batch['label'].to(device)
                else:
                    images, labels = sampled_batch['image'].to(device), sampled_batch['segmentation'].to(device)
                images, labels = images.cuda(), labels.cuda()
                outputs = model(images)

                # Calculate loss
                loss_ce = ce_loss(outputs, labels.long())
                loss_dice = dice_loss(outputs, labels, softmax=True)
                loss = 0.2 * loss_ce + 0.8 * loss_dice

                if epoch == 0 and i_batch == 0:
                    import matplotlib.pyplot as plt
                    plt.figure(figsize=(12,4))
                    plt.subplot(131); plt.imshow(images[0].cpu().permute(1,2,0).mean(dim=-1))
                    plt.title('Input Image')
                    plt.subplot(132); plt.imshow(labels[0].cpu().squeeze())
                    plt.title('Ground Truth')
                    plt.subplot(133); plt.imshow(torch.argmax(outputs, dim=1)[0].cpu().squeeze())
                    plt.title('Prediction')
                    plt.savefig('debug_sample.png')
                    plt.close()

                # Backward pass
                opt.zero_grad()
                loss.backward()

                # RGN Auto-tuning
                criterion = lambda outputs, labels: 0.2 * ce_loss(outputs, labels.long()) + 0.8 * dice_loss(outputs, labels, softmax=True)
                grad_weights = get_lr_weights(model, trainloader, criterion, param_groups, args.device)
                max_weight = max(grad_weights.values())

                # Update optimizer with new learning rates
                params = []
                current_param_groups = get_parameter_groups(model, 0)
                for name in current_param_groups:
                    params.append({
                        'params': current_param_groups[name],
                        'lr': grad_weights[name] * lr / max_weight
                    })
                    logger.info(f"===============================\nRGN chosen parameters: {name} = {grad_weights[name] * lr / max_weight}\n===============================")
                opt = optim.Adam(params, weight_decay=wd)

                opt.step()

                # logger and metrics tracking
                tune_metrics[f'lr_{lr}_wd_{wd}'].append({
                    'epoch': epoch,
                    'loss': loss.item(),
                    'grad_weights': grad_weights
                })
                logger.info(f"Loss: {loss.item()}")

            # Save checkpoints and metrics
            if epoch % args.save_interval == 0:
                torch.save(model.state_dict(), os.path.join(snapshot_path, f'model_lr{lr}_wd{wd}_epoch{epoch}.pth'))
                torch.save(tune_metrics, os.path.join(snapshot_path, 'tune_metrics.pt'))

    return "Surgical Training Finished!"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_ckpt', type=str,
                        default='./pretrain/epoch_149.pth',
                        help='Path to pre-trained checkpoint')
    parser.add_argument('--root_path', type=str,
                        default='./datasets/Synapse_blurred/train_npz',
                        help='Root dir for data')
    parser.add_argument('--dataset', type=str,
                        default='Synapse', help='Dataset name')
    parser.add_argument('--list_dir', type=str,
                        default='./lists/lists_Synapse_blurred',
                        help='List dir')
    parser.add_argument('--num_classes', type=int,
                        default=9, help='Output channel of network')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for finetuned models')
    parser.add_argument('--max_epochs', type=int, default=51,
                        help='Max finetuning epochs')
    parser.add_argument('--batch_size', type=int,
                        default=24, help='Batch size per GPU')
    parser.add_argument('--base_lr', type=float, default=0.001,
                        help='Base learning rate')
    parser.add_argument('--seed', type=int, default=1234,
                        help='Random seed')
    parser.add_argument('--cfg', type=str, required=True,
                        metavar="FILE", help='Path to config file')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--n_gpu', type=int, default=2,
                        help='Total GPU count')
    parser.add_argument('--deterministic', type=int, default=1,
                        help='Use deterministic training')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='Checkpoint save interval')
    parser.add_argument('--auto_tune', type=str, default='RGN',
                        choices=['RGN', 'eb-criterion', 'none'],
                        help='Auto-tuning method')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                            'full: cache all data, '
                            'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
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
    config.defrost()
    config.MODEL.PRETRAIN_CKPT = args.pretrained_ckpt
    config.freeze()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

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
            'root_path': args.root_path,
            'list_dir': args.list_dir,
            'num_classes': args.num_classes,
        },
        'kits23': {
            'root_path': args.root_path,
            'list_dir': './lists/kits23',
            'num_classes': args.num_classes,
        }
    }
    args.num_classes = dataset_config[args.dataset]['num_classes']
    args.root_path = dataset_config[args.dataset]['root_path']
    args.list_dir = dataset_config[args.dataset]['list_dir']

    model = ViT_seg(config, img_size=args.img_size,
                    num_classes=args.num_classes).to(device)
    model.load_from(config)
    ### DEBUG
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, args.img_size, args.img_size).to(device)
        output = model(dummy_input)
        print("Output shape:", output.shape)
    ### DEBUG
    print(f"Loaded pretrained weights from {args.pretrained_ckpt}")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    surgical_trainer(args, model, args.output_dir)

