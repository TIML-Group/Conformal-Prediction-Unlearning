from copy import deepcopy
import os
import math
import time
import torch
import torch.nn as nn
import utils


class UnlearnMethod:
    def __init__(self, model, loss_function, save_path, args) -> None:
        self.unlearn_dataloaders = None
        self.model = model
        self.loss_function = loss_function
        self.save_path = save_path
        self.args = args
        self.params = {}

    def prepare_unlearn(self, unlearn_dataloaders: dict) -> None:
        self.unlearn_dataloaders = unlearn_dataloaders
        return

    def get_unlearned_model(self) -> nn.Module:
        return self.model

    def get_params(self) -> dict:
        return self.params

def cycle(dl):
    while True:
        for data in dl:
            yield data


def calc_sparsity(tensor):
    # Count zero elements
    num_zero_elements = tensor.numel() - torch.count_nonzero(tensor)

    # Total number of elements
    total_elements = tensor.numel()

    # Compute sparsity
    sparsity = num_zero_elements / total_elements
    return sparsity.item(), total_elements, num_zero_elements


@torch.no_grad()
def update_parameters(source_model, input_model, avg_fn):
    source_param = (source_model.parameters())
    input_param = (input_model.parameters())
    for p_swa, p_model in zip(source_param, input_param):
        device = p_swa.device
        p_model_ = p_model.detach().to(device)
        p_swa.detach().copy_(avg_fn(p_swa.detach(), p_model_))


def expdecay_lr_scheduler(base_lr, current_epoch, T_max, base=2):
    return base_lr * (1 - current_epoch / T_max) ** base


def linear_lr_scheduler(base_lr, current_epoch, T_max, base=1):
    return base_lr * (1 - current_epoch / T_max) ** base


def cosine_lr_scheduler(base_lr, current_epoch, T_max):
    return base_lr * (1 + math.cos(math.pi * current_epoch / T_max)) / 2


class AdaptiveLoss(torch.nn.Module):
    def __init__(self, loss_function, lambd=1, reduction='mean'):
        super(AdaptiveLoss, self).__init__()
        self.loss_function = loss_function  # reduction=none
        self.lambd = lambd
        self.reduction = reduction

    def forward(self, predict, target):
        ori_loss = self.loss_function(predict, target)
        coef = 1 / (torch.pow(ori_loss.detach().clone(), self.lambd) + 1e-15)
        # print((coef / coef.sum()).sum(), ((coef / coef.sum())* predict.shape[0]).sum()) 
        ad_loss = (coef / coef.sum()) * ori_loss * predict.shape[0]

        if self.reduction == 'mean':
            ad_loss = ad_loss.mean()
        elif self.reduction == 'sum':
            ad_loss = ad_loss.sum()
        return ad_loss


class SFRon(UnlearnMethod):
    def __init__(self, model, loss_function, save_path, args) -> None:
        super().__init__(model, loss_function, save_path, args)
        self.num_classes = args.num_classes
        self.seed = args.seed
        self.eval = True
        self.forget_loss_function = None
        self.retain_loss_function = None
        self.weight_saliency_mask = None
        self.avg_fn = None

        # params

        self.lambd = 0.5  # lambda in the paper
        if args.data_name == 'tiny_imagenet' and args.unlearn_type == 'random' and args.retain_ratio==0.9:
            # tinyimage 10%
            self.opt = 'adamw'
            self.momentum = 0.9
            self.weight_decay = 0.05
            self.retain_lr = 2e-5

            self.n_iters = 800  # 500
            self.unlearn_loss = "adaga"
            self.forget_freq = 1

            self.forget_alpha = 800.0  # 500
            self.max_norm = 7.0

            self.ema_enabled = True
            self.ema_beta = 1.0

            self.sched = 'cosine'
            self.gamma = 0.6

            self.mask = True
            self.th = 1

            self.log_freq = 100
        elif args.data_name == 'tiny_imagenet' and args.unlearn_type == 'random' and args.retain_ratio==0.5:
            # tinyimage 10%
            self.opt = 'adamw'
            self.momentum = 0.9
            self.weight_decay = 0.05
            self.retain_lr = 2e-5

            self.n_iters = 1000
            self.unlearn_loss = "adaga"
            self.forget_freq = 1

            self.forget_alpha = 1000.0
            self.max_norm = 7.0

            self.ema_enabled = True
            self.ema_beta = 1.0

            self.sched = 'cosine'
            self.gamma = 0.6

            self.mask = True
            self.th = 1

            self.log_freq = 100
        elif args.data_name == 'tiny_imagenet' and args.unlearn_type == 'class':
            # tinyimage 10%
            self.opt = 'adamw'
            self.momentum = 0.9
            self.weight_decay = 0.05
            self.retain_lr = 2e-5

            self.n_iters = 500  # 500
            self.unlearn_loss = "adaga"
            self.forget_freq = 1

            self.forget_alpha = 500.0  # 500.0
            self.max_norm = 7.0

            self.ema_enabled = True
            self.ema_beta = 1.0

            self.sched = 'cosine'
            self.gamma = 0.6

            self.mask = True
            self.th = 1

            self.log_freq = 100
        elif args.data_name == 'cifar10' and args.unlearn_type == 'random' and args.retain_ratio==0.9:
            # CIFAR10 10%
            self.opt = 'sgd'
            self.momentum = 0.9
            self.weight_decay = 5e-4
            self.retain_lr = 0.01

            self.n_iters = 3000
            self.unlearn_loss = "adaga"
            self.forget_freq = 1  # 5

            self.forget_alpha = 40 # 25
            self.max_norm = 7.0

            self.ema_enabled = True
            self.ema_beta = 1.0

            self.sched = 'cosine'
            self.lambd = 0.5  # lambda in the paper

            self.mask = True
            self.th = 1  # gamma in the paper

            self.log_freq = 100
        elif args.data_name == 'cifar10' and args.unlearn_type == 'random' and args.retain_ratio==0.5:
            # CIFAR10 10%
            self.opt = 'sgd'
            self.momentum = 0.9
            self.weight_decay = 5e-4
            self.retain_lr = 0.01

            self.n_iters = 1300  # 1500
            self.unlearn_loss = "adaga"
            self.forget_freq = 1  # 5

            self.forget_alpha = 80. # 25
            self.max_norm = 7.0

            self.ema_enabled = True
            self.ema_beta = 1.0

            self.sched = 'cosine'
            self.lambd = 0.5  # lambda in the paper

            self.mask = True
            self.th = 1  # gamma in the paper

            self.log_freq = 100
        elif args.data_name == 'cifar10' and args.unlearn_type == 'class':
            # CIFAR10 10%
            self.opt = 'sgd'
            self.momentum = 0.9
            self.weight_decay = 5e-4
            self.retain_lr = 0.01

            self.n_iters = 1500
            self.unlearn_loss = "adaga"
            self.forget_freq = 5  # 5

            self.forget_alpha = 25 # 25
            self.max_norm = 7.0

            self.ema_enabled = True
            self.ema_beta = 1.0

            self.sched = 'cosine'
            self.lambd = 0.5  # lambda in the paper

            self.mask = True
            self.th = 1  # gamma in the paper

            self.log_freq = 100



    def prepare_unlearn(self, unlearn_dataloaders: dict) -> None:
        self.unlearn_dataloaders = unlearn_dataloaders

        def avg_fn(averaged_model_parameter, model_parameter):
            return (
                    1 - self.ema_beta) * model_parameter + self.ema_beta * averaged_model_parameter

        self.avg_fn = avg_fn
        # adaptive gradient ascent
        if self.unlearn_loss == "adaga":
            # adaptive ce loss
            ce_loss = nn.CrossEntropyLoss(reduction='none')
            self.forget_loss_function = AdaptiveLoss(ce_loss, lambd=self.lambd)
        elif self.unlearn_loss == "ga":
            self.forget_loss_function = self.loss_function
        self.retain_loss_function = self.loss_function
        # forget-remain balanced weight saliency mask
        forget_trainloader = self.unlearn_dataloaders['forget_train']
        retain_trainloader = self.unlearn_dataloaders['retain_train']
        if self.mask:
            self.weight_saliency_mask = self.get_weight_saliency_mask(
                forget_loader=forget_trainloader,
                remain_loader=retain_trainloader,
                threshold=self.th
            )
        else:
            self.weight_saliency_mask = None

    def get_unlearned_model(self) -> nn.Module:
        retain_trainloader = self.unlearn_dataloaders['retain_train']
        forget_trainloader = self.unlearn_dataloaders['forget_train']
        retain_train_iter = cycle(retain_trainloader)
        forget_train_iter = cycle(forget_trainloader)

        if self.sched == 'cosine':
            lr_scheduler = cosine_lr_scheduler
        elif self.sched == 'linear':
            lr_scheduler = linear_lr_scheduler
        elif self.sched == 'expdecay':
            lr_scheduler = expdecay_lr_scheduler

        if self.opt == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), self.retain_lr, momentum=self.momentum,
                                        weight_decay=self.weight_decay)
        elif self.opt == "adamw":
            optimizer = torch.optim.AdamW(self.model.parameters(), self.retain_lr, weight_decay=self.weight_decay)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.n_iters
        )

        if self.ema_enabled:
            ori_model = deepcopy(self.model)

        # Variables for monitoring/logging purposes:
        train_steps = 0
        log_forget_steps = 0
        log_remain_steps = 0
        running_forget_loss = 0
        running_forget_acc1 = 0
        running_remain_loss = 0
        running_remain_acc1 = 0
        start_time = time.time()

        for step in range(0, self.n_iters):
            self.model.train()
            if step % self.forget_freq == 0:
                cur_forget_alpha = lr_scheduler(self.forget_alpha, step, self.n_iters)
                # adaptive weighted gradient ascent on forget train
                x_forget, y_forget = next(forget_train_iter)
                x_forget, y_forget = x_forget.cuda(), y_forget.cuda()
                optimizer.zero_grad()
                outputs = self.model(x_forget)
                ori_forget_loss = -self.forget_loss_function(outputs, y_forget)
                forget_loss = cur_forget_alpha * ori_forget_loss
                forget_loss.backward()
                if self.mask:
                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            param.grad *= self.weight_saliency_mask[name].to(param.grad.device)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_norm)
                optimizer.step()
                forget_acc1 = utils.accuracy(outputs.data, y_forget)[0]
                running_forget_loss += ori_forget_loss
                running_forget_acc1 += forget_acc1
                log_forget_steps += 1

            # train on retain train
            self.model.train()
            x_retain, y_retain = next(retain_train_iter)
            x_retain, y_retain = x_retain.cuda(), y_retain.cuda()

            optimizer.zero_grad()
            outputs = self.model(x_retain)
            ori_remain_loss = self.retain_loss_function(outputs, y_retain)
            remain_loss = ori_remain_loss
            remain_loss.backward()
            optimizer.step()
            remain_acc1 = utils.accuracy(outputs.data, y_retain)[0]
            running_remain_loss += ori_remain_loss
            running_remain_acc1 += remain_acc1
            log_remain_steps += 1

            lr = optimizer.param_groups[0]['lr']
            train_steps += 1
            if self.eval and train_steps % self.log_freq == 0:
                # Measure training speed:
                end_time = time.time()
                steps_per_sec = log_remain_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_forget_loss = torch.tensor(running_forget_loss / log_forget_steps).item()
                avg_forget_acc1 = torch.tensor(running_forget_acc1 / log_forget_steps).item()
                avg_remain_loss = torch.tensor(running_remain_loss / log_remain_steps).item()
                avg_remain_acc1 = torch.tensor(running_remain_acc1 / log_remain_steps).item()
                print(
                    f"step={train_steps} Forget L:{avg_forget_loss:.4f} Acc:{avg_forget_acc1:.4f} Remain L:{avg_remain_loss:.4f} Acc:{avg_remain_acc1:.4f} LR:{lr}")
                # Reset monitoring variables:
                log_forget_steps = 0
                log_remain_steps = 0
                running_forget_loss = 0
                running_forget_acc1 = 0
                running_remain_loss = 0
                running_remain_acc1 = 0
                start_time = time.time()

            if self.ema_enabled:
                update_parameters(self.model, ori_model, self.avg_fn)
                ori_model = deepcopy(self.model)

            scheduler.step()
        return self.model

    def get_weight_saliency_mask(self, forget_loader, remain_loader, threshold):
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=0,
        )
        criterion = self.loss_function

        # forget fisher 
        forget_fisher_path = os.path.join(self.save_path, "forget_fisher.pt")
        if os.path.exists(forget_fisher_path):
            forget_gradients = torch.load(forget_fisher_path)
        else:
            forget_gradients = {}
            for name, param in self.model.named_parameters():
                forget_gradients[name] = 0
            self.model.eval()
            for i, (image, target) in enumerate(forget_loader):
                image = image.cuda()
                target = target.cuda()

                # compute output
                output_clean = self.model(image)
                loss = criterion(output_clean, target)

                optimizer.zero_grad()
                loss.backward()

                with torch.no_grad():
                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            forget_gradients[name] += param.grad.data.cpu() ** 2 / len(forget_loader)

            torch.save(forget_gradients, forget_fisher_path)

        # remain fisher 
        remain_fisher_path = os.path.join(self.save_path, "remain_fisher.pt")
        if os.path.exists(remain_fisher_path):
            remain_gradients = torch.load(remain_fisher_path)
        else:
            remain_gradients = {}
            for name, param in self.model.named_parameters():
                remain_gradients[name] = 0
            self.model.eval()
            for i, (image, target) in enumerate(remain_loader):
                image = image.cuda()
                target = target.cuda()

                # compute output
                output_clean = self.model(image)
                loss = criterion(output_clean, target)

                optimizer.zero_grad()
                loss.backward()

                with torch.no_grad():
                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            remain_gradients[name] += param.grad.data.cpu() ** 2 / len(remain_loader)

            torch.save(remain_gradients, remain_fisher_path)

        total_cnt = 0
        w_cnt = 0
        weight_saliency_mask = {}
        for name in forget_gradients.keys():
            weight_saliency_mask[name] = 0
            try:
                weight_saliency = (forget_gradients[name] + 1e-15) / (remain_gradients[name] + 1e-15)
                w = weight_saliency >= threshold
                w_sparsity, total_elements, w_num_zero_elements = calc_sparsity(w)
                total_cnt += total_elements
                w_cnt += w_num_zero_elements
                weight_saliency_mask[name] = w
            except:
                pass
        print(f"Total sparsity th:{threshold} weight:{w_cnt / total_cnt * 100}")
        return weight_saliency_mask

    def get_params(self) -> dict:
        self.params = {
            'opt': self.opt,
            'momentum': self.momentum,
            'weight_decay': self.weight_decay,
            'retain_lr': self.retain_lr,
            'n_iters': self.n_iters,
            'forget_freq': self.forget_freq,
            'forget_alpha': self.forget_alpha,
            'max_norm': self.max_norm,
            'ema_beta': self.ema_beta,
            'sched': self.sched,
            'lambd': self.lambd,
            'mask': self.mask,
            'threshold': self.th,
        }
        return self.params