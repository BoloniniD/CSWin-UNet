# utils/tpgm.py
import copy
import torch
import torch.nn as nn

class TPGM(nn.Module):
    def __init__(self, model, norm_mode, exclude_list=[]) -> None:
        super().__init__()
        self.norm_mode = norm_mode
        self.exclude_list = exclude_list
        self.threshold = torch.nn.Hardtanh(0,1)
        self.constraints_name = []
        self.constraints = []
        self.create_contraint(model) # Create constraint place holders
        self.constraints = nn.ParameterList(self.constraints)
        self.init = True

    def create_contraint(self, module):
        for name, para in module.named_parameters():
            if not para.requires_grad:
                continue
            if name not in self.exclude_list:
                self.constraints_name.append(name)
                temp = nn.Parameter(torch.Tensor([0]), requires_grad=True)
                self.constraints.append(temp)

    def apply_constraints(
        self,
        new,
        pre_trained,
        constraint_iterator,
        apply=False,
    ):
        # apply: A flag for whether in Projection Update or Projection stage (Sec.3.3)
        for (name, new_para), anchor_para in zip(
            new.named_parameters(), pre_trained.parameters()
        ):

            if not new_para.requires_grad:
                continue
            if name not in self.exclude_list:
                alpha = self._project_ratio(
                    new_para,
                    anchor_para,
                    constraint_iterator,
                )
                v = (new_para.detach() - anchor_para.detach()) * alpha
                temp = v + anchor_para.detach()
                if apply:
                    # When apply=True, copy the projected weights into the original tensor with no gradient.
                    with torch.no_grad():
                        new_para.copy_(temp)
                else:
                    # When apply=False, copy the projected weights into the original tensor.
                    new_para.requires_grad = False # Need to set requires_grad=False s.t. the original tensor is no longer a leaf node.
                    new_para.copy_(temp)

        self.init = False


    def _project_ratio(self, new, anchor, constraint_iterator):
        t = new.detach() - anchor.detach()

        if "l2" in self.norm_mode:
            norms = torch.norm(t) # L2 norm
        else:
            norms = torch.sum(torch.abs(t), dim=tuple(range(1,t.dim())), keepdim=True) # MARS norm

        constraint = next(constraint_iterator)

        if self.init:
            # Initialize the constraints to a small value, i.e., norms.min()/2, the first time.
            with torch.no_grad():
                temp = norms.min()/2
                constraint.copy_(temp)
        with torch.no_grad():
            constraint.copy_(self._clip(constraint, norms)) # Clip constraint to be within (1e-8, norms.max)

        ratio = self.threshold(constraint / (norms + 1e-8))
        return ratio

    def _clip(self, constraint, norms):
        return torch.nn.functional.hardtanh(constraint,1e-8,norms.max())

    def forward(
        self,
        new=None,
        pre_trained=None,
        x=None,
        apply=False,
    ):
        constraint_iterator = iter(self.constraints)

        if apply:
            self.apply_constraints(
                new,
                pre_trained,
                constraint_iterator,
                apply=apply,
            )
        else:
            new_copy = new #copy.deepcopy(new)
            new_copy.eval()
            self.apply_constraints(new_copy, pre_trained, constraint_iterator)
            out = new_copy(x)
            return out


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
        exclude_list = []
    ) -> None:
        #####################################################################
        # model: The pre-trained model weights .
        # pgmloader: Dataloader for training TPGM
        # norm_mode ["l2_norm","mars_norm"]: Norm used for calculating projection in TPGM.
        # proj_lr: Learning rate for TPGM.
        # max_iters: Number of iterations for running TPGM Projection Update each time.
        # ce_loss: Cross entropy loss function
        # dice_loss: Dice loss function
        # exclude_list: Specify the list of weights to exclude from TPGM projection
        #####################################################################
        self.device = torch.device("cuda")
        self.proj_lr = proj_lr
        self.max_iters = max_iters
        self.tpgm = TPGM(model, norm_mode=norm_mode, exclude_list=exclude_list).to(self.device)
        self.pre_trained = copy.deepcopy(model)
        self.pgm_optimizer = torch.optim.Adam(self.tpgm.parameters(), lr=self.proj_lr)
        self.pgmloader = pgmloader
        self.dataset_iterator = iter(self.pgmloader)
        self.ce_loss = ce_loss
        self.dice_loss = dice_loss

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

                outputs = self.tpgm(model, self.pre_trained, x=pgm_image)

                # Use the same loss combination as in finetuning
                loss_ce = self.ce_loss(outputs, pgm_target[:].long())
                loss_dice = self.dice_loss(outputs, pgm_target, softmax=True)
                pgm_loss = 0.4 * loss_ce + 0.6 * loss_dice

                self.pgm_optimizer.zero_grad()
                pgm_loss.backward()
                self.pgm_optimizer.step()
                self.count += 1

                if (self.count+1)%20 == 0:
                    print("{}/{} TPGM iterations completed".format(self.count, self.max_iters))

        self.tpgm(model, self.pre_trained, apply=True)
