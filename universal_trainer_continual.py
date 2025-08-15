# trainer_continual.py (updated version)
import logging
import os
import random
import sys
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms
from collections import defaultdict
import numpy as np
import torch.nn.functional as F
import itertools


def worker_init_fn(worker_id):
    random.seed(1234 + worker_id)


class ContinualCrossEntropyLoss(nn.Module):
    """Custom CE loss for continual learning that only computes loss on relevant classes"""
    def __init__(self, task_offset, num_new_classes, ignore_index=-100):
        super().__init__()
        self.task_offset = task_offset  # Where new task classes start
        self.num_new_classes = num_new_classes  # Number of classes in new task
        self.ignore_index = ignore_index
        self.ce_loss = CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, predictions, targets):
        # Map new task labels to expanded model output space
        # Background (0) stays as 0
        # New task classes (1 to num_new_classes-1) map to (task_offset to task_offset+num_new_classes-2)
        mapped_targets = targets.clone()
        mask = targets > 0  # Non-background pixels
        mapped_targets[mask] = targets[mask] + self.task_offset - 1

        return self.ce_loss(predictions, mapped_targets.long())


class ContinualDiceLoss(nn.Module):
    """Custom Dice loss for continual learning"""
    def __init__(self, task_offset, num_new_classes, num_total_classes):
        super().__init__()
        self.task_offset = task_offset
        self.num_new_classes = num_new_classes
        self.num_total_classes = num_total_classes
        self.dice_loss = DiceLoss(num_total_classes)

    def forward(self, predictions, targets, softmax=True):
        # Create one-hot encoding for the expanded class space
        mapped_targets = targets.clone()
        mask = targets > 0
        mapped_targets[mask] = targets[mask] + self.task_offset - 1

        return self.dice_loss(predictions, mapped_targets, softmax=softmax)


class TPGM(nn.Module):
    """Task-aware Projection Gradient Method for constraining weight updates"""
    def __init__(self, model, norm_mode='mars', exclude_list=[]) -> None:
        super().__init__()
        self.norm_mode = norm_mode
        self.exclude_list = exclude_list
        self.threshold = torch.nn.Hardtanh(0, 1)
        self.constraints_name = []
        self.constraints = []
        self.create_constraint(model)
        self.constraints = nn.ParameterList(self.constraints)
        self.init = True

    def create_constraint(self, module):
        for name, para in module.named_parameters():
            if not para.requires_grad:
                continue
            if not any(exc in name for exc in self.exclude_list):
                self.constraints_name.append(name)
                temp = nn.Parameter(torch.tensor(1e-6), requires_grad=True)
                self.constraints.append(temp)

    def apply_constraints(self, new, pre_trained, constraint_iterator, apply=False):
        for (name, new_para), (_, anchor_para) in zip(
            new.named_parameters(), pre_trained.named_parameters()
        ):
            if not new_para.requires_grad:
                continue
            if not any(exc in name for exc in self.exclude_list):
                # Handle size mismatch for expanded layers
                if new_para.shape != anchor_para.shape:
                    # Only constrain the old part of the weights for output layers
                    if 'output' in name or 'head' in name or 'final' in name:
                        if len(new_para.shape) >= 2:
                            old_size = anchor_para.shape[0]
                            if old_size <= new_para.shape[0]:
                                alpha = self._project_ratio(
                                    new_para[:old_size], anchor_para, constraint_iterator
                                )
                                v = (new_para[:old_size] - anchor_para) * alpha
                                temp = v + anchor_para
                                if apply:
                                    with torch.no_grad():
                                        new_para[:old_size].copy_(temp.detach())
                                else:
                                    # Don't break gradient flow when not applying
                                    new_para[:old_size].data = temp.detach()
                            else:
                                _ = next(constraint_iterator)
                        else:
                            # For 1D parameters (bias)
                            old_size = anchor_para.shape[0]
                            if old_size <= new_para.shape[0]:
                                alpha = self._project_ratio(
                                    new_para[:old_size], anchor_para, constraint_iterator
                                )
                                v = (new_para[:old_size] - anchor_para) * alpha
                                temp = v + anchor_para
                                if apply:
                                    with torch.no_grad():
                                        new_para[:old_size].copy_(temp.detach())
                                else:
                                    new_para[:old_size].data = temp.detach()
                            else:
                                _ = next(constraint_iterator)
                    else:
                        # Skip constraint for non-output layers with shape mismatch
                        _ = next(constraint_iterator)
                        continue
                else:
                    alpha = self._project_ratio(new_para, anchor_para, constraint_iterator)
                    v = (new_para - anchor_para) * alpha
                    temp = v + anchor_para
                    if apply:
                        with torch.no_grad():
                            new_para.copy_(temp.detach())
                    else:
                        # Don't break gradient flow - just update data
                        new_para.data = temp.detach()

        # Only set init to False when actually applying constraints
        if apply:
            self.init = False

    def _project_ratio(self, new, anchor, constraint_iterator):
        t = new - anchor  # Keep gradients when not applying

        # Compute norms
        if "l2" in self.norm_mode:
            norms = torch.norm(t)
        else:
            norms = torch.sum(torch.abs(t), dim=tuple(range(1, t.dim())), keepdim=True)

        # Ensure norms is always positive
        norms = torch.clamp(norms, min=1e-8)

        constraint = next(constraint_iterator)

        if self.init:
            with torch.no_grad():
                init_val = torch.clamp(norms.detach().min() / 2 if norms.numel() > 0 else torch.tensor(1e-6),
                                     min=1e-8, max=1e-3)
                constraint.copy_(init_val)

        with torch.no_grad():
            max_norm = torch.clamp(norms.detach().max() if norms.numel() > 0 else torch.tensor(1e-3), min=1e-7)
            clipped_constraint = self._clip(constraint, max_norm)
            constraint.copy_(clipped_constraint)

        # Compute ratio with numerical stability
        ratio = self.threshold(constraint / (norms + 1e-8))
        return ratio

    def _clip(self, constraint, max_norm):
        min_val = 1e-8
        max_norm = torch.clamp(max_norm, min=min_val * 10)
        return torch.clamp(constraint, min=min_val, max=max_norm)

    def _safe_model_copy(self, model):
        """Safely create a model copy without gradient graph issues"""
        try:
            # Get state dict and clone all tensors
            state_dict = {}
            for k, v in model.state_dict().items():
                state_dict[k] = v.clone()

            # Create new model instance
            from networks.vision_transformer import CSwinUnet as ViT_seg

            if hasattr(model, 'module'):  # DataParallel case
                original_model = model.module
                config = getattr(original_model, 'config', None)
                img_size = getattr(original_model, 'img_size', 224)
                num_classes = getattr(original_model, 'num_classes', 9)

                new_model = ViT_seg(config, img_size=img_size, num_classes=num_classes)
                new_model = new_model.to(next(model.parameters()).device)
                new_model = torch.nn.DataParallel(new_model)
            else:
                config = getattr(model, 'config', None)
                img_size = getattr(model, 'img_size', 224)
                num_classes = getattr(model, 'num_classes', 9)

                new_model = ViT_seg(config, img_size=img_size, num_classes=num_classes)
                new_model = new_model.to(next(model.parameters()).device)

            # Load state dict
            new_model.load_state_dict(state_dict)

            # Ensure the new model requires gradients
            for param in new_model.parameters():
                param.requires_grad = True

            return new_model

        except Exception as e:
            logging.error(f"Failed to create model copy: {e}")
            raise e

    def forward(self, new=None, pre_trained=None, x=None, apply=False):
        constraint_iterator = iter(self.constraints)

        if apply:
            # When applying, we modify the original model in-place
            self.apply_constraints(new, pre_trained, constraint_iterator, apply=True)
            return None
        else:
            # When not applying, we work with a copy for constraint learning
            new_copy = self._safe_model_copy(new)
            new_copy.train()  # Keep in training mode for gradient computation

            # Apply constraints to the copy (this updates the constraint parameters)
            constraint_iterator = iter(self.constraints)
            self.apply_constraints(new_copy, pre_trained, constraint_iterator, apply=False)

            # Forward pass through the constrained model
            out = new_copy(x)
            return out


class TPGMTrainer:
    """TPGM trainer for projection-based fine-tuning in continual learning"""
    def __init__(self, model, pgmloader, norm_mode, proj_lr, max_iters,
                 task_offset, num_new_classes, num_total_classes, exclude_list=[]):
        self.device = torch.device("cuda")
        self.proj_lr = proj_lr
        self.max_iters = max_iters
        self.task_offset = task_offset
        self.num_new_classes = num_new_classes
        self.num_total_classes = num_total_classes

        self.tpgm = TPGM(model, norm_mode=norm_mode, exclude_list=exclude_list).to(self.device)
        self.pre_trained = copy.deepcopy(model)
        self.pgm_optimizer = torch.optim.Adam(self.tpgm.parameters(), lr=self.proj_lr)
        self.pgmloader = pgmloader
        self.dataset_iterator = iter(self.pgmloader)
        self.criterion = ContinualCrossEntropyLoss(task_offset, num_new_classes)

    def tpgm_iters(self, model, apply=False):
        if not apply:
            self.count = 0
            # Enable gradient computation for TPGM constraint learning
            self.tpgm.train()

            while self.count < self.max_iters:
                try:
                    data = next(self.dataset_iterator)
                except StopIteration:
                    self.dataset_iterator = iter(self.pgmloader)
                    data = next(self.dataset_iterator)

                pgm_image, pgm_target = data['image'], data['label']
                pgm_image = pgm_image.to(self.device)
                pgm_target = pgm_target.to(self.device)

                # Clear gradients
                self.pgm_optimizer.zero_grad()

                # Forward pass through TPGM
                outputs = self.tpgm(model, self.pre_trained, x=pgm_image, apply=False)

                # Compute loss
                pgm_loss = self.criterion(outputs, pgm_target)

                # Backward pass
                pgm_loss.backward()

                # Update TPGM constraints
                self.pgm_optimizer.step()

                self.count += 1

                if (self.count + 1) % 20 == 0:
                    logging.info(f"TPGM iteration {self.count}/{self.max_iters} completed, loss: {pgm_loss.item():.4f}")

        # Apply the learned constraints to the actual model
        with torch.no_grad():
            self.tpgm(model, self.pre_trained, apply=True)


def identify_output_layer_keys(model):
    """Identify the output/final layer keys in the model"""
    state_dict = model.state_dict()
    output_keys = []

    # Common patterns for output layers in segmentation models
    patterns = ['output', 'head', 'final', 'seg', 'cls', 'last_layer', 'decoder.4']

    for key in state_dict.keys():
        for pattern in patterns:
            if pattern in key.lower():
                output_keys.append(key)
                break

    # If no pattern matches, try to find the last conv/linear layer
    if not output_keys:
        conv_keys = [k for k in state_dict.keys() if 'conv' in k.lower() or 'linear' in k.lower()]
        if conv_keys:
            # Assuming the last conv/linear layer is the output
            output_keys = [conv_keys[-1]]
            # Also check for associated bias
            bias_key = conv_keys[-1].replace('.weight', '.bias')
            if bias_key in state_dict:
                output_keys.append(bias_key)

    return output_keys


def load_pretrained_weights_continual(model, pretrained_path, num_classes_old, num_classes_new):
    """
    Loads pretrained weights and adapts the final layer for continual learning.
    The model should already be initialized with the correct total number of classes.
    """
    logging.info(f"Loading pretrained model from {pretrained_path} for continual learning.")
    logging.info(f"Old task classes: {num_classes_old}, New task classes: {num_classes_new}")

    # Load pretrained state dict
    checkpoint = torch.load(pretrained_path, map_location='cpu')

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            pretrained_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            pretrained_dict = checkpoint['state_dict']
        else:
            pretrained_dict = checkpoint
    else:
        pretrained_dict = checkpoint

    # Remove 'module.' prefix if present
    if 'module.' in list(pretrained_dict.keys())[0]:
        logging.info("Removing 'module.' prefix from pretrained weights.")
        pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}

    # Get current model state dict
    model_dict = model.state_dict()

    # Identify output layer keys
    output_layer_keys = identify_output_layer_keys(model)
    if not output_layer_keys:
        # Fallback to common patterns
        output_layer_keys = [k for k in pretrained_dict.keys() if any(pattern in k for pattern in ['output', 'head', 'final'])]

    logging.info(f"Identified output layer keys: {output_layer_keys}")

    # Filter out output layers from pretrained dict (we'll handle them separately)
    pretrained_dict_filtered = {k: v for k, v in pretrained_dict.items() if k not in output_layer_keys}

    # Load backbone weights
    missing_keys = []
    for k in pretrained_dict_filtered:
        if k in model_dict and pretrained_dict_filtered[k].shape == model_dict[k].shape:
            model_dict[k] = pretrained_dict_filtered[k]
        else:
            missing_keys.append(k)

    logging.info(f"Loaded {len(pretrained_dict_filtered) - len(missing_keys)} backbone layers from pretrained model.")
    if missing_keys:
        logging.info(f"Missing or mismatched keys: {len(missing_keys)}")

    # Handle output layer weights - copy old class weights to the expanded layer
    with torch.no_grad():
        for key in output_layer_keys:
            if key in model_dict and key in pretrained_dict:
                old_weights = pretrained_dict[key]
                new_weights = model_dict[key]

                if 'weight' in key:
                    # Copy old class weights
                    new_weights[:num_classes_old] = old_weights
                    logging.info(f"Copied pretrained weights for '{key}' for the first {num_classes_old} classes.")

                    # Initialize new class weights
                    if len(new_weights.shape) == 4:  # Conv2d
                        nn.init.xavier_normal_(new_weights[num_classes_old:])
                    elif len(new_weights.shape) == 2:  # Linear
                        nn.init.xavier_normal_(new_weights[num_classes_old:])

                elif 'bias' in key:
                    # Copy old class biases
                    new_weights[:num_classes_old] = old_weights
                    # New class biases are already initialized to zero
                    logging.info(f"Copied pretrained bias for '{key}' for the first {num_classes_old} classes.")

                model_dict[key] = new_weights
            else:
                logging.warning(f"Key '{key}' from pretrained model not found in the new model or size mismatch.")

    # Load the updated state dict
    model.load_state_dict(model_dict)
    logging.info("Successfully adapted model for continual learning.")
    return model


def get_surgical_weights_continual(model, loader, args, layer_names, task_offset, num_new_classes):
    """Calculate RGN-based weights for surgical fine-tuning in continual learning"""
    metrics = defaultdict(list)
    average_metrics = defaultdict(float)

    # Use continual loss for gradient calculation
    criterion = ContinualCrossEntropyLoss(task_offset, num_new_classes)

    partial_loader = itertools.islice(loader, 5)
    xent_grads = []

    for sampled_batch in partial_loader:
        image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
        image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

        outputs = model(image_batch)
        loss_xent = criterion(outputs, label_batch)
        grad_xent = torch.autograd.grad(
            outputs=loss_xent, inputs=model.parameters(), retain_graph=True
        )
        xent_grads.append([g.detach() for g in grad_xent])

    def get_grad_norms(model, grads, surgical_mode):
        _metrics = defaultdict(list)
        for (name, param), grad in zip(model.named_parameters(), grads):
            if name not in layer_names:
                continue
            if surgical_mode == "eb-criterion":
                tmp = (grad * grad) / (torch.var(grad, dim=0, keepdim=True) + 1e-8)
                _metrics[name] = tmp.mean().item()
            else:  # RGN mode
                _metrics[name] = torch.norm(grad).item() / (torch.norm(param).item() + 1e-8)
        return _metrics

    for xent_grad in xent_grads:
        xent_grad_metrics = get_grad_norms(model, xent_grad, args.surgical_mode)
        for k, v in xent_grad_metrics.items():
            metrics[k].append(v)

    for k, v in metrics.items():
        average_metrics[k] = np.array(v).mean(0)

    return average_metrics


def debug_predictions(model, dataloader, epoch, task_offset, num_classes, max_batches=2):
    """Debug function to check if model is making reasonable predictions during training"""
    model.eval()
    print(f"\n=== DEBUG PREDICTIONS - Epoch {epoch} ===")

    ce_loss = ContinualCrossEntropyLoss(task_offset, num_classes)

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break

            image, label = batch['image'].cuda(), batch['label'].cuda()
            outputs = model(image)

            # Check raw outputs
            print(f"Batch {i}:")
            print(f"  Input shape: {image.shape}")
            print(f"  Output shape: {outputs.shape}")
            print(f"  Output classes range: {outputs.shape[1]}")

            # Check ground truth labels
            unique_labels = torch.unique(label)
            print(f"  Unique GT labels: {unique_labels.cpu().numpy()}")

            # Map labels like in training
            mapped_targets = label.clone()
            mask = label > 0
            mapped_targets[mask] = label[mask] + task_offset - 1
            unique_mapped = torch.unique(mapped_targets)
            print(f"  Unique mapped labels: {unique_mapped.cpu().numpy()}")

            # Check predictions
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            unique_preds = torch.unique(predictions)
            print(f"  Unique predictions: {unique_preds.cpu().numpy()}")

            # Check if model is predicting the right classes
            class_probs = probabilities.mean(dim=(0, 2, 3))
            active_classes = []
            for idx, prob in enumerate(class_probs):
                if prob > 0.01:  # Classes with >1% average probability
                    active_classes.append((idx, prob.item()))
            print(f"  Active classes (>1% prob): {active_classes}")

            # Compute loss for verification
            loss = ce_loss(outputs, label)
            print(f"  CE Loss: {loss.item():.4f}")

            print()

    model.train()
    print("=== END DEBUG PREDICTIONS ===\n")


def trainer_continual(args, model, snapshot_path):
    """Continual learning trainer with optional Surgical Fine-tuning and TPGM"""
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator
    from torch.utils.data import Subset

    # Setup logging
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    # Basic configuration
    base_lr = args.base_lr
    num_classes = args.num_classes  # New task classes
    old_num_classes = args.old_num_classes  # Previous model classes
    total_classes = old_num_classes + num_classes - 1  # Total classes after expansion
    task_offset = old_num_classes  # Where new classes start
    batch_size = args.batch_size * args.n_gpu

    logging.info(f"Continual Learning: {old_num_classes} -> {total_classes} classes")
    logging.info(f"Task offset: {task_offset}, New classes: {num_classes}")
    logging.info(f"Dataset fraction: {args.dataset_fraction}")

    # Load pretrained weights if specified
    if args.pretrained_ckpt:
        model = load_pretrained_weights_continual(
            model, args.pretrained_ckpt, old_num_classes, num_classes
        )
        logging.info("Pretrained weights loaded and adapted for continual learning")

    # Multi-GPU support
    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    # Save original model state for TPGM (after loading pretrained weights)
    original_model = None
    if args.enable_tpgm:
        original_model = copy.deepcopy(model)
        logging.info("TPGM enabled - saved model state")

    # Setup dataset for new task
    db_train_full = Synapse_dataset(
        base_dir=args.root_path,
        list_dir=args.list_dir,
        split="train",
        transform=transforms.Compose([
            RandomGenerator(output_size=[args.img_size, args.img_size])
        ])
    )

    # Apply dataset fraction if specified
    if args.dataset_fraction < 1.0:
        # Calculate the number of samples to use
        total_samples = len(db_train_full)
        num_samples = int(total_samples * args.dataset_fraction)

        # Create random indices for subset
        np.random.seed(args.seed)  # Ensure reproducibility
        indices = np.random.choice(total_samples, num_samples, replace=False)
        indices = sorted(indices.tolist())  # Sort for deterministic behavior

        # Create subset dataset
        db_train = Subset(db_train_full, indices)

        logging.info(f"Using {num_samples}/{total_samples} samples ({args.dataset_fraction:.2%} of dataset)")
    else:
        db_train = db_train_full
        logging.info(f"Using full dataset: {len(db_train)} samples")

    print(f"The length of train set is: {len(db_train)}")

    trainloader = DataLoader(
        db_train, batch_size=batch_size, shuffle=True,
        num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn
    )

    model.train()

    # Setup continual learning losses
    ce_loss = ContinualCrossEntropyLoss(task_offset, num_classes)
    dice_loss = ContinualDiceLoss(task_offset, num_classes, total_classes)

    # Setup optimizer
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    # Setup TPGM if enabled
    tpgm_trainer = None
    if args.enable_tpgm:
        # For TPGM, create a separate smaller loader if needed
        if args.dataset_fraction < 1.0:
            # Use same subset for TPGM but with different batch size
            pgm_dataset = Subset(db_train_full, indices)
        else:
            pgm_dataset = db_train_full

        pgm_loader = DataLoader(
            pgm_dataset, batch_size=args.tpgm_batch_size, shuffle=True,
            num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn
        )

        # Identify output layers to exclude from TPGM constraints
        output_keys = identify_output_layer_keys(model)
        exclude_list = output_keys
        if args.n_gpu > 1:
            # Add module prefix for DataParallel
            exclude_list = [f"module.{key}" for key in output_keys]

        tpgm_trainer = TPGMTrainer(
            model=model,
            pgmloader=pgm_loader,
            norm_mode=args.tpgm_norm_mode,
            proj_lr=args.tpgm_proj_lr,
            max_iters=args.tpgm_max_iters,
            task_offset=task_offset,
            num_new_classes=num_classes,
            num_total_classes=total_classes,
            exclude_list=exclude_list
        )
        logging.info(f"TPGM initialized for continual learning with exclude list: {exclude_list}")

    # Setup surgical fine-tuning if enabled
    layer_weights = None
    layer_names = []
    if args.enable_surgical:
        layer_names = [n for n, _ in model.named_parameters() if "bn" not in n]
        logging.info(f"Surgical fine-tuning enabled for continual learning")

    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # This will be adjusted based on subset size
    logging.info(f"{len(trainloader)} iterations per epoch. {max_iterations} max iterations")

    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)

    for epoch_num in iterator:
        # Apply surgical fine-tuning weight calculation
        if args.enable_surgical and epoch_num % args.surgical_update_freq == 0:
            logging.info(f"Updating surgical weights at epoch {epoch_num}")
            weights = get_surgical_weights_continual(
                model, trainloader, args, layer_names, task_offset, num_classes
            )

            if weights:  # Only proceed if weights were calculated
                # Adjust weights
                if args.surgical_mode == "RGN":
                    max_weight = max(weights.values())
                    for k in weights:
                        weights[k] = weights[k] / max_weight
                        # Boost output layer weights
                        if any(pattern in k for pattern in ['output', 'head', 'final']):
                            weights[k] = min(weights[k] * 1.5, 1.0)
                elif args.surgical_mode == "eb-criterion":
                    for k in weights:
                        weights[k] = 0.0 if weights[k] < 0.95 else 1.0

                # Create parameter groups
                params_weights = []
                for n, p in model.named_parameters():
                    if "bn" not in n:
                        if n in weights:
                            params_weights.append({'params': p, 'lr': weights[n] * base_lr})
                        else:
                            params_weights.append({'params': p, 'lr': base_lr})

                optimizer = optim.SGD(params_weights, momentum=0.9, weight_decay=0.0001)

                logging.info(f"Surgical weights range: [{min(weights.values()):.4f}, "
                           f"{max(weights.values()):.4f}]")

        # Apply TPGM projection update
        if args.enable_tpgm and args.tpgm_proj_freq > 0 and epoch_num % args.tpgm_proj_freq == 0:
            logging.info(f"Applying TPGM projection update at epoch {epoch_num}")
            tpgm_trainer.tpgm_iters(model, apply=False)

        epoch_loss = 0.0
        epoch_ce_loss = 0.0
        epoch_dice_loss = 0.0

        for i_batch, sampled_batch in enumerate(trainloader):
            if i_batch == 0:  # Debug first batch every 10 epochs
                debug_predictions(model, trainloader, epoch_num, task_offset, num_classes, max_batches=1)
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

            # Forward pass
            outputs = model(image_batch)

            # Compute losses using continual learning losses
            loss_ce = ce_loss(outputs, label_batch)
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.4 * loss_ce + 0.6 * loss_dice

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Apply TPGM projection after weight update
            if args.enable_tpgm and tpgm_trainer:
                tpgm_trainer.tpgm_iters(model, apply=True)

            # Learning rate scheduling
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                if args.enable_surgical and 'lr' in param_group:
                    original_lr = param_group.get('lr', base_lr)
                    lr_ratio = original_lr / base_lr if base_lr > 0 else 1.0
                    param_group['lr'] = lr_ * lr_ratio
                else:
                    param_group['lr'] = lr_

            iter_num = iter_num + 1
            epoch_loss += loss.item()
            epoch_ce_loss += loss_ce.item()
            epoch_dice_loss += loss_dice.item()

            # Logging
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            if i_batch % 10 == 0:
                logging.info(f'Epoch [{epoch_num}/{max_epoch}] Iteration [{i_batch}/{len(trainloader)}]: '
                           f'Loss: {loss.item():.4f}, CE: {loss_ce.item():.4f}')

            # Visualization
            if iter_num % 20 == 0:
                image = image_batch[0, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)

                # Visualize predictions
                outputs_vis = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs_vis[0, ...] * 20, iter_num)

                # Map labels for visualization
                label_vis = label_batch[0, ...].clone()
                mask = label_vis > 0
                label_vis[mask] = label_vis[mask] + task_offset - 1
                writer.add_image('train/GroundTruth', label_vis.unsqueeze(0) * 20, iter_num)

        # Log epoch statistics
        avg_epoch_loss = epoch_loss / len(trainloader)
        avg_epoch_ce = epoch_ce_loss / len(trainloader)
        avg_epoch_dice = epoch_dice_loss / len(trainloader)

        logging.info(f'Epoch [{epoch_num}/{max_epoch}] Average Loss: {avg_epoch_loss:.4f}, '
                   f'CE: {avg_epoch_ce:.4f}, Dice: {avg_epoch_dice:.4f}')

        # Save checkpoints
        save_interval = 5
        if epoch_num > int(max_epoch / 2) and (epoch_num + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, f'epoch_{epoch_num}_continual.pth')

            # Save model state and metadata
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'epoch': epoch_num,
                'old_num_classes': old_num_classes,
                'new_num_classes': num_classes,
                'total_classes': total_classes,
                'task_offset': task_offset,
                'optimizer_state_dict': optimizer.state_dict(),
                'dataset_fraction': args.dataset_fraction,  # Save fraction info
            }
            torch.save(checkpoint, save_mode_path)
            logging.info(f"Saved continual learning checkpoint to {save_mode_path}")

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, f'final_continual_model.pth')
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'epoch': epoch_num,
                'old_num_classes': old_num_classes,
                'new_num_classes': num_classes,
                'total_classes': total_classes,
                'task_offset': task_offset,
                'optimizer_state_dict': optimizer.state_dict(),
                'dataset_fraction': args.dataset_fraction,  # Save fraction info
            }
            torch.save(checkpoint, save_mode_path)
            logging.info(f"Saved final continual model to {save_mode_path}")
            iterator.close()
            break

    writer.close()
    return "Continual Learning Training Finished!"