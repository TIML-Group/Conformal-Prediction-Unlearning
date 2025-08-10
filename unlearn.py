import torch
from copy import deepcopy
from torch.utils.data import DataLoader, Subset, dataset, Dataset, random_split
from tqdm import tqdm
import utils
import torch.nn as nn
import wandb
import time
import numpy as np
from torch.nn import functional as F
from typing import Dict, List
from utils import evaluate_acc
from SFRon import SFRon
from collections import OrderedDict
import random


'''basic func'''
def training_step(model, batch, criterion, device):
    images, clabels = batch

    images, clabels = images.to(device), clabels.to(device)
    out = model(images)  # Generate predictions
    loss = criterion(out, clabels)  # Calculate loss
    _, pred = torch.max(out, 1)
    num_correct = (pred == clabels).sum()
    acc = num_correct.item() / len(clabels)

    return loss, acc, num_correct

def fit_one_cycle(
    epochs, model, train_loader, forget_loader, test_loader, device, lr, milestones, mask=None
):
    train_acc = evaluate_acc(model, train_loader, device)
    forget_acc = evaluate_acc(model, forget_loader, device)
    test_acc = evaluate_acc(model, test_loader, device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.9, weight_decay=5e-4)
    if milestones is None:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=0.1, last_epoch=-1
        )


    test_size = len(train_loader.dataset)
    model.train()
    for epoch in range(epochs):
        start = time.time()
        pbar = tqdm(train_loader, total=len(train_loader))
        correct_num = 0
        acc_list = []
        for batch_i, batch in enumerate(pbar):
            loss, acc, correct = training_step(model, batch, criterion, device)
            correct_num += correct
            acc_list.append(acc)
            loss.backward()

            if mask:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name]

            optimizer.step()
            optimizer.zero_grad()
        scheduler.step()

        forget_acc = evaluate_acc(model, forget_loader, device)
        test_acc = evaluate_acc(model, test_loader, device)
        wandb.log(
            {'epoch': epoch, 'train_acc': correct_num/test_size, 'forget_acc': forget_acc, 'test_acc': test_acc,
             "lr": optimizer.param_groups[0]["lr"],
             "epoch_time": time.time() - start})
    train_acc = evaluate_acc(model, train_loader, device)
    return train_acc, forget_acc, test_acc


def baseline(
    model,
    train_retain_dl,
    train_forget_dl,
    test_retain_dl,
    test_forget_dl,
    device,
    **kwargs,
):
    return (utils.evaluate_acc(model, train_retain_dl, device),
            utils.evaluate_acc(model, train_forget_dl, device),
            utils.evaluate_acc(model, test_retain_dl, device))


def retrain(
    model,
    train_retain_dl,
    train_forget_dl,
    test_retain_dl,
    test_forget_dl,
    device,
    num_epochs,
    milestones,
    learning_rate,
    **kwargs,
):
    train_acc, forget_acc, test_acc = fit_one_cycle(
        num_epochs, model, train_retain_dl, train_forget_dl, test_retain_dl, lr=learning_rate, milestones=milestones, device=device
    )

    return train_acc, forget_acc, test_acc


def finetune(
    model,
    train_retain_dl,
    train_forget_dl,
    test_retain_dl,
    test_forget_dl,
    device,
    num_epochs,
    learning_rate,
    milestones,
    **kwargs,
):
    train_acc, forget_acc, test_acc = fit_one_cycle(
        num_epochs, model, train_retain_dl, train_forget_dl, test_retain_dl, lr=learning_rate, device=device, milestones=milestones
    )

    return train_acc, forget_acc, test_acc


def RL(
    model,
    train_retain_dl,
    train_forget_dl,
    test_retain_dl,
    test_forget_dl,
    num_classes,
    device,
    num_epochs,
    batch_size,
    learning_rate,
    milestones,
    **kwargs,
):
    unlearninglabels = list(range(num_classes))
    unlearning_trainset = []

    for x, clabel in train_forget_dl.dataset:
        rnd = random.choice(unlearninglabels)
        while rnd == clabel:
            rnd = random.choice(unlearninglabels)
        unlearning_trainset.append((x, rnd))

    for x, y in train_retain_dl.dataset:
        unlearning_trainset.append((x, y))

    unlearning_train_set_dl = DataLoader(
        unlearning_trainset, batch_size=batch_size, pin_memory=True, shuffle=True
    )
    train_acc, forget_acc, test_acc = fit_one_cycle(
        num_epochs, model, unlearning_train_set_dl, train_forget_dl, test_retain_dl, lr=learning_rate, device=device, milestones=milestones
    )

    return train_acc, forget_acc, test_acc


def FisherForgetting(
    model,
    train_retain_dl,
    train_forget_dl,
    test_retain_dl,
    test_forget_dl,
    num_classes,
    device,
    unlearn_type,
    forget_class,
    **kwargs,
):
    def hessian(dataset, model):
        model.eval()
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
        criterion = nn.CrossEntropyLoss()

        for p in model.parameters():
            p.grad_acc = 0
            p.grad2_acc = 0

        for data, orig_target in tqdm(train_loader):
            data, orig_target = data.to(device), orig_target.to(device)
            output = model(data)
            prob = F.softmax(output, dim=-1).data

            for y in range(output.shape[1]):
                target = torch.empty_like(orig_target).fill_(y)
                loss = criterion(output, target)
                model.zero_grad()
                loss.backward(retain_graph=True)
                for p in model.parameters():
                    if p.requires_grad:
                        p.grad_acc += (orig_target == target).float() * p.grad.data
                        p.grad2_acc += prob[:, y] * p.grad.data.pow(2)

        for p in model.parameters():
            p.grad_acc /= len(train_loader)
            p.grad2_acc /= len(train_loader)


    def get_mean_var(p, is_base_dist=False, alpha=3e-6, unlearn_type='random'):
        var = deepcopy(1.0 / (p.grad2_acc + 1e-8))
        var = var.clamp(max=1e3)
        if p.size(0) == num_classes:
            var = var.clamp(max=1e2)
        var = alpha * var

        if p.ndim > 1:
            var = var.mean(dim=1, keepdim=True).expand_as(p).clone()
        if not is_base_dist:
            mu = deepcopy(p.data0.clone())
        else:
            mu = deepcopy(p.data0.clone())

        if unlearn_type == 'class':
            if p.size(0) == num_classes:
                mu[forget_class] = 0
                var[forget_class] = 0.0001
            if p.size(0) == num_classes:
                var *= 10
            elif p.ndim == 1:
                var *= 10
        elif unlearn_type == 'random':
            if p.ndim == 1:
                var *= 10
        return mu, var


    for p in model.parameters():
        p.data0 = deepcopy(p.data.clone())

    hessian(train_retain_dl.dataset, model)

    fisher_dir = []
    alpha = 1e-6
    for i, p in enumerate(model.parameters()):
        mu, var = get_mean_var(p, False, alpha, unlearn_type)
        p.data = mu + var.sqrt() * torch.empty_like(p.data0).normal_()
        fisher_dir.append(var.sqrt().view(-1).cpu().detach().numpy())
    return (utils.evaluate_acc(model, train_retain_dl, device),
            utils.evaluate_acc(model, train_forget_dl, device),
            utils.evaluate_acc(model, test_retain_dl, device))

def l1_regularization(model):
    params_vec = []
    for param in model.parameters():
        params_vec.append(param.view(-1))
    return torch.linalg.norm(torch.cat(params_vec), ord=1)

def GA(
    model,
    train_retain_dl,
    train_forget_dl,
    test_retain_dl,
    test_forget_dl,
    device,
    num_epochs,
    learning_rate,
    **kwargs,
):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    test_size = len(train_forget_dl.dataset)

    model.train()
    for epoch in range(num_epochs):
        start = time.time()
        correct_num = 0
        for i, (image, target) in enumerate(train_forget_dl):
            image = image.to(device)
            target = target.to(device)

            # compute output
            output_clean = model(image)
            loss = - criterion(output_clean, target)
            # loss = -criterion(output_clean, target) + 0.2 * l1_regularization(model)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, pred = torch.max(output_clean, 1)
            num_correct = (pred == target).sum()
            correct_num += num_correct

        forget_acc = evaluate_acc(model, train_forget_dl, device)
        test_acc = evaluate_acc(model, test_retain_dl, device)
        wandb.log(
            {'epoch': epoch, 'train_acc': correct_num/test_size, 'forget_acc': forget_acc, 'test_acc': test_acc,
             "lr": optimizer.param_groups[0]["lr"],
             "epoch_time": time.time() - start})
    return evaluate_acc(model, train_retain_dl, device), forget_acc, test_acc


def ga_plus(
    model,
    train_retain_dl,
    train_forget_dl,
    test_retain_dl,
    test_forget_dl,
    device,
    num_epochs,
    learning_rate,
    args,
    **kwargs
):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs)

    unlearning_data = LossData(forget_data=train_forget_dl.dataset, retain_data=train_retain_dl.dataset)
    training_loader = DataLoader(
        unlearning_data, batch_size=args.batch_size, shuffle=True, pin_memory=True
    )

    model.train()
    for epoch in range(args.num_epochs):
        start = time.time()
        for i, batch in enumerate(training_loader):
            loss, _, _ = training_step_ga_plus(model, batch, criterion)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        retain_acc = evaluate_acc(model, train_retain_dl, device)
        forget_acc = evaluate_acc(model, train_forget_dl, device)
        test_acc = evaluate_acc(model, test_retain_dl, device)
        wandb.log(
            {'epoch': epoch, 'train_acc': retain_acc, 'forget_acc': forget_acc, 'test_acc': test_acc,
             "lr": optimizer.param_groups[0]["lr"],
             "epoch_time": time.time() - start})
        scheduler.step()

    return retain_acc, forget_acc, test_acc


'''teacher'''
def UnlearnerLoss(
    output, labels, full_teacher_logits, unlearn_teacher_logits, KL_temperature
):
    labels = torch.unsqueeze(labels, dim=1)

    f_teacher_out = F.softmax(full_teacher_logits / KL_temperature, dim=1)
    u_teacher_out = F.softmax(unlearn_teacher_logits / KL_temperature, dim=1)

    overall_teacher_out = labels * u_teacher_out + (1 - labels) * f_teacher_out
    student_out = F.log_softmax(output / KL_temperature, dim=1)
    return F.kl_div(student_out, overall_teacher_out)


def unlearning_step(
    model,
    unlearning_teacher,
    full_trained_teacher,
    unlearn_data_loader,
    optimizer,
    device,
    KL_temperature,
):
    losses = []
    pbar = tqdm(unlearn_data_loader, total=len(unlearn_data_loader))
    for batch_i, batch in enumerate(pbar):
        x, y = batch
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            full_teacher_logits = full_trained_teacher(x)
            unlearn_teacher_logits = unlearning_teacher(x)
        output = model(x)
        optimizer.zero_grad()
        loss = UnlearnerLoss(
            output=output,
            labels=y,
            full_teacher_logits=full_teacher_logits,
            unlearn_teacher_logits=unlearn_teacher_logits,
            KL_temperature=KL_temperature,
        )
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().cpu().numpy())
    return np.mean(losses)


def blindspot_unlearner(
    model,
    unlearning_teacher,
    full_trained_teacher,
    retain_data,
    forget_data,
    num_epochs,
    batch_size,
    device,
    learning_rate,
    KL_temperature,
    test_retain_dl,
    test_forget_dl,
    train_retain_dl,
    train_forget_dl
):
    unlearning_data = UnLearningData(forget_data=forget_data, retain_data=retain_data)
    unlearning_loader = DataLoader(
        unlearning_data, batch_size=batch_size, shuffle=True, pin_memory=True
    )

    unlearning_teacher.eval()
    full_trained_teacher.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        start = time.time()
        loss = unlearning_step(
            model=model,
            unlearning_teacher=unlearning_teacher,
            full_trained_teacher=full_trained_teacher,
            unlearn_data_loader=unlearning_loader,
            optimizer=optimizer,
            device=device,
            KL_temperature=KL_temperature,
        )

        train_acc = evaluate_acc(model, train_retain_dl, device)
        forget_acc = evaluate_acc(model, train_forget_dl, device)
        test_acc = evaluate_acc(model, test_retain_dl, device)
        wandb.log(
            {'epoch': epoch, 'train_acc': train_acc, 'forget_acc': forget_acc, 'test_acc': test_acc,
             "lr": optimizer.param_groups[0]["lr"],
             "epoch_time": time.time() - start})
    return train_acc, forget_acc, test_acc


def teacher(
    model,
    unlearning_teacher,
    train_retain_dl,
    train_forget_dl,
    test_retain_dl,
    test_forget_dl,
    device,
    learning_rate,  # 0.0001
    batch_size,
    num_epochs,
    **kwargs,
):
    student_model = deepcopy(model)
    # retain_train_subset = random.sample(
    #     train_retain_dl.dataset, int(0.3 * len(train_retain_dl.dataset))
    # )
    # len_retain_train_subset = int(0.3 * len(train_retain_dl.dataset))
    len_retain_train_subset = int(len(train_forget_dl.dataset))
    retain_train_subset, _ = random_split(train_retain_dl.dataset, [len_retain_train_subset, len(train_retain_dl.dataset) - len_retain_train_subset])

    train_acc, forget_acc, test_acc = blindspot_unlearner(
        model=model,
        unlearning_teacher=unlearning_teacher,
        full_trained_teacher=student_model,
        retain_data=retain_train_subset,
        forget_data=train_forget_dl.dataset,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        device=device,
        KL_temperature=1,
        test_retain_dl=test_retain_dl,
        test_forget_dl=test_forget_dl,
        train_retain_dl=train_retain_dl,
        train_forget_dl=train_forget_dl,
    )
    return train_acc, forget_acc, test_acc


def ssd(
    model,
    train_retain_dl,
    train_forget_dl,
    test_retain_dl,
    test_forget_dl,
    dampening_constant,
    selection_weighting,
    full_train_dl,
    device,
    num_epochs,
    learning_rate,
    **kwargs,
):
    parameters = {
        "lower_bound": 1,
        "exponent": 1,
        "magnitude_diff": None,
        "min_layer": -1,
        "max_layer": -1,
        "forget_threshold": 1,
        "dampening_constant": dampening_constant,
        "selection_weighting": selection_weighting,
    }

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    ssd = ParameterPerturber(model, optimizer, device, parameters)
    model = model.eval()

    sample_importances = ssd.calc_importance(train_forget_dl)
    original_importances = ssd.calc_importance(full_train_dl)

    ssd.modify_weight(original_importances, sample_importances)

    return (utils.evaluate_acc(model, train_retain_dl, device),
            utils.evaluate_acc(model, train_forget_dl, device),
            utils.evaluate_acc(model, test_retain_dl, device))



def scrub(
    model,
    train_retain_dl,
    train_forget_dl,
    test_retain_dl,
    test_forget_dl,
    device,
    num_epochs,
    learning_rate,
    args,
    **kwargs,
):
    model_t = deepcopy(model)
    module_list = nn.ModuleList([model, model_t])


    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(args.kd_T)
    criterion_kd = DistillKL(args.kd_T)

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)
    criterion_list.append(criterion_div)
    criterion_list.append(criterion_kd)


    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

    for epoch in range(num_epochs):
        start = time.time()
        if epoch <= args.msteps:
            maximize_loss = train_distill(epoch, train_forget_dl, module_list, criterion_list, optimizer, args.gamma, args.beta, "maximize", quiet=False)
        train_acc, train_loss = train_distill(epoch, train_retain_dl, module_list, criterion_list, optimizer, args.gamma, args.beta,"minimize", quiet=False)

        forget_acc, test_acc = utils.evaluate_acc(model, train_forget_dl, device), utils.evaluate_acc(model, test_retain_dl, device)
        wandb.log(
            {'epoch': epoch, 'train_acc': train_acc, 'forget_acc': forget_acc, 'test_acc': test_acc,
             "lr": optimizer.param_groups[0]["lr"],
             "epoch_time": time.time() - start})


    return train_acc, forget_acc, test_acc




def salun(
    model,
    train_retain_dl,
    train_forget_dl,
    test_retain_dl,
    test_forget_dl,
    num_classes,
    device,
    num_epochs,
    batch_size,
    learning_rate,
    milestones,
    mask,
    **kwargs,
):
    unlearninglabels = list(range(num_classes))
    unlearning_trainset = []

    for x, clabel in train_forget_dl.dataset:
        rnd = random.choice(unlearninglabels)
        while rnd == clabel:
            rnd = random.choice(unlearninglabels)
        unlearning_trainset.append((x, rnd))

    for x, y in train_retain_dl.dataset:
        unlearning_trainset.append((x, y))

    unlearning_train_set_dl = DataLoader(
        unlearning_trainset, batch_size=batch_size, pin_memory=True, shuffle=True
    )
    train_acc, forget_acc, test_acc = fit_one_cycle(
        num_epochs, model, unlearning_train_set_dl, train_forget_dl, test_retain_dl, lr=learning_rate, device=device, milestones=milestones, mask=mask
    )

    return train_acc, forget_acc, test_acc


def sfron(
    model,
    train_retain_dl,
    train_forget_dl,
    test_retain_dl,
    test_forget_dl,
    num_classes,
    device,
    save_dir,
    args,
    **kwargs,
):

    acc_r, acc_f, acc_t = (utils.evaluate_acc(model, train_retain_dl, device),
     utils.evaluate_acc(model, train_forget_dl, device),
     utils.evaluate_acc(model, test_retain_dl, device))

    loss_function = nn.CrossEntropyLoss()
    unlearn_dataloaders = OrderedDict(
        forget_train=train_forget_dl,
        retain_train=train_retain_dl,
        forget_valid=test_forget_dl,
        retain_valid=test_retain_dl
    )

    unlearn_method = SFRon(model, loss_function, save_dir, args)
    unlearn_method.prepare_unlearn(unlearn_dataloaders)
    model = unlearn_method.get_unlearned_model()

    acc_r, acc_f, acc_t = (utils.evaluate_acc(model, train_retain_dl, device),
     utils.evaluate_acc(model, train_forget_dl, device),
     utils.evaluate_acc(model, test_retain_dl, device))

    return acc_r, acc_f, acc_t, model




"""
This file is used for the Scrub method
"""
###############################################
# SCRUB ParameterPerturber
###############################################
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        # correct_k = correct[:k].view(-1).float().sum(0)
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def train_distill(epoch, train_loader, module_list, criterion_list, optimizer, gamma, beta, split,
                  print_freq=12, quiet=False):
    """One epoch distillation"""
    # set modules as train()
    # for module in module_list:
    #     module.train()
    # # set teacher as eval()
    # module_list[-1].eval()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]

    model_s = module_list[0]
    model_t = module_list[-1]
    model_s.train()
    model_t.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    kd_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    acc_max_top1 = AverageMeter()

    end = time.time()
    for idx, (input, target) in enumerate(train_loader):

        input = input.cuda()
        target = target.cuda()
        data_time.update(time.time() - end)

        input = torch.Tensor(input).float()
        # target = torch.squeeze(torch.Tensor(target).long())

        # ===================forward=====================
        logit_s = model_s(input)
        with torch.no_grad():
            logit_t = model_t(input)

        # cls + kl div
        loss_cls = criterion_cls(logit_s, target)
        loss_div = criterion_div(logit_s, logit_t)
        loss_kd = 0

        if split == "minimize":
            loss = gamma * loss_cls + beta * loss_div

        elif split == "maximize":
            loss = -loss_div

        if split == "minimize" and not quiet:
            acc1, acc5 = accuracy(logit_s, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1.item(), input.size(0))
            top5.update(acc5.item(), input.size(0))
        elif split == "maximize" and not quiet:
            kd_losses.update(loss.item(), input.size(0))
            acc_max, _ = accuracy(logit_s, target, topk=(1, 5))
            acc_max_top1.update(acc_max.item(), input.size(0))


    # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

    if split == "maximize":
        if not quiet:
            # if idx % print_freq == 0:
            print('*** Maximize step ***')
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Forget_Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, idx, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=kd_losses, top1=acc_max_top1))
            # sys.stdout.flush()
    elif split == "minimize":
        if not quiet:
            print('*** Minimize step ***')
            # print(' * Acc@1 {top1.avg:.3f} '.format(top1=top1))
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Retain_Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, idx, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1))

        return top1.avg, losses.avg
    else:
        # module_list[0] = model_s
        # module_list[-1] = model_t
        return kd_losses.avg


"""
This file is used for the Selective Synaptic Dampening method
Strategy files use the methods from here
"""
###############################################
# SSD ParameterPerturber
###############################################

class ParameterPerturber:
    def __init__(
        self,
        model,
        opt,
        device="cuda" if torch.cuda.is_available() else "cpu",
        parameters=None,
    ):
        self.model = model
        self.opt = opt
        self.device = device
        self.alpha = None
        self.xmin = None

        print(parameters)
        self.lower_bound = parameters["lower_bound"]
        self.exponent = parameters["exponent"]
        self.magnitude_diff = parameters["magnitude_diff"]  # unused
        self.min_layer = parameters["min_layer"]
        self.max_layer = parameters["max_layer"]
        self.forget_threshold = parameters["forget_threshold"]
        self.dampening_constant = parameters["dampening_constant"]
        self.selection_weighting = parameters["selection_weighting"]

    def get_layer_num(self, layer_name: str) -> int:
        layer_id = layer_name.split(".")[1]
        if layer_id.isnumeric():
            return int(layer_id)
        else:
            return -1

    def zerolike_params_dict(self, model: torch.nn) -> Dict[str, torch.Tensor]:
        """
        Taken from: Avalanche: an End-to-End Library for Continual Learning - https://github.com/ContinualAI/avalanche
        Returns a dict like named_parameters(), with zeroed-out parameter valuse
        Parameters:
        model (torch.nn): model to get param dict from
        Returns:
        dict(str,torch.Tensor): dict of zero-like params
        """
        return dict(
            [
                (k, torch.zeros_like(p, device=p.device))
                for k, p in model.named_parameters()
            ]
        )

    def fulllike_params_dict(
        self, model: torch.nn, fill_value, as_tensor: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Returns a dict like named_parameters(), with parameter values replaced with fill_value

        Parameters:
        model (torch.nn): model to get param dict from
        fill_value: value to fill dict with
        Returns:
        dict(str,torch.Tensor): dict of named_parameters() with filled in values
        """

        def full_like_tensor(fillval, shape: list) -> list:
            """
            recursively builds nd list of shape shape, filled with fillval
            Parameters:
            fillval: value to fill matrix with
            shape: shape of target tensor
            Returns:
            list of shape shape, filled with fillval at each index
            """
            if len(shape) > 1:
                fillval = full_like_tensor(fillval, shape[1:])
            tmp = [fillval for _ in range(shape[0])]
            return tmp

        dictionary = {}

        for n, p in model.named_parameters():
            _p = (
                torch.tensor(full_like_tensor(fill_value, p.shape), device=self.device)
                if as_tensor
                else full_like_tensor(fill_value, p.shape)
            )
            dictionary[n] = _p
        return dictionary

    def subsample_dataset(self, dataset: dataset, sample_perc: float) -> Subset:
        """
        Take a subset of the dataset

        Parameters:
        dataset (dataset): dataset to be subsampled
        sample_perc (float): percentage of dataset to sample. range(0,1)
        Returns:
        Subset (float): requested subset of the dataset
        """
        sample_idxs = np.arange(0, len(dataset), step=int((1 / sample_perc)))
        return Subset(dataset, sample_idxs)

    def split_dataset_by_class(self, dataset: dataset) -> List[Subset]:
        """
        Split dataset into list of subsets
            each idx corresponds to samples from that class

        Parameters:
        dataset (dataset): dataset to be split
        Returns:
        subsets (List[Subset]): list of subsets of the dataset,
            each containing only the samples belonging to that class
        """
        n_classes = len(set([target for _, target in dataset]))
        subset_idxs = [[] for _ in range(n_classes)]
        for idx, (x, y) in enumerate(dataset):
            subset_idxs[y].append(idx)

        return [Subset(dataset, subset_idxs[idx]) for idx in range(n_classes)]

    def calc_importance(self, dataloader: DataLoader) -> Dict[str, torch.Tensor]:
        """
        Adapated from: Avalanche: an End-to-End Library for Continual Learning - https://github.com/ContinualAI/avalanche
        Calculate per-parameter, importance
            returns a dictionary [param_name: list(importance per parameter)]
        Parameters:
        DataLoader (DataLoader): DataLoader to be iterated over
        Returns:
        importances (dict(str, torch.Tensor([]))): named_parameters-like dictionary containing list of importances for each parameter
        """
        criterion = nn.CrossEntropyLoss()
        importances = self.zerolike_params_dict(self.model)
        for batch in dataloader:
            x, y = batch
            x, y = x.to(self.device), y.to(self.device)
            self.opt.zero_grad()
            out = self.model(x)
            loss = criterion(out, y)
            loss.backward()

            for (k1, p), (k2, imp) in zip(
                self.model.named_parameters(), importances.items()
            ):
                if p.grad is not None:
                    imp.data += p.grad.data.clone().pow(2)

        # average over mini batch length
        for _, imp in importances.items():
            imp.data /= float(len(dataloader))
        return importances

    def modify_weight(
        self,
        original_importance: List[Dict[str, torch.Tensor]],
        forget_importance: List[Dict[str, torch.Tensor]],
    ) -> None:
        """
        Perturb weights based on the SSD equations given in the paper
        Parameters:
        original_importance (List[Dict[str, torch.Tensor]]): list of importances for original dataset
        forget_importance (List[Dict[str, torch.Tensor]]): list of importances for forget sample
        threshold (float): value to multiply original imp by to determine memorization.

        Returns:
        None

        """

        with torch.no_grad():
            for (n, p), (oimp_n, oimp), (fimp_n, fimp) in zip(
                self.model.named_parameters(),
                original_importance.items(),
                forget_importance.items(),
            ):
                # print(f"{n} before: {p.sum()}")

                # Synapse Selection with parameter alpha
                oimp_norm = oimp.mul(self.selection_weighting)
                locations = torch.where(fimp > oimp_norm)

                # Synapse Dampening with parameter lambda
                weight = ((oimp.mul(self.dampening_constant)).div(fimp)).pow(
                    self.exponent
                )
                update = weight[locations]
                # Bound by 1 to prevent parameter values to increase.
                min_locs = torch.where(update > self.lower_bound)
                update[min_locs] = self.lower_bound
                p[locations] = p[locations].mul(update)
                # print(f"{n} after: {p.sum()}")


###############################################

class UnLearningData(Dataset):
    def __init__(self, forget_data, retain_data):
        super().__init__()
        self.forget_data = forget_data
        self.retain_data = retain_data
        self.forget_len = len(forget_data)
        self.retain_len = len(retain_data)

    def __len__(self):
        return self.retain_len + self.forget_len

    def __getitem__(self, index):
        if index < self.forget_len:
            x = self.forget_data[index][0]
            y = 1
            return x, y
        else:
            x = self.retain_data[index - self.forget_len][0]
            y = 0
            return x, y

class LossData(Dataset):
    def __init__(self, forget_data, retain_data):
        super().__init__()
        self.forget_data = forget_data
        self.retain_data = retain_data
        self.forget_len = len(forget_data)
        self.retain_len = len(retain_data)

    def __len__(self):
        return self.retain_len + self.forget_len

    def __getitem__(self, index):
        if index < self.forget_len:
            x, y = self.forget_data[index]
            label = 1
            return x, y, label
        else:
            x, y = self.retain_data[index - self.forget_len]
            label = 0
            return x, y, label

### ga_plus
def training_step_ga_plus(model, batch, criterion):
    device = next(model.parameters()).device
    images, clabels, labels = batch
    images, clabels, labels = images.to(device), clabels.to(device), labels.to(device)

    out = model(images)

    retain_mask = (labels == 0)
    forget_mask = (labels == 1)

    retain_logits = out[retain_mask]
    retain_clabels = clabels[retain_mask]
    loss_retain = criterion(retain_logits, retain_clabels)

    forget_logits = out[forget_mask]
    forget_clabels = clabels[forget_mask]
    loss_forget = criterion(forget_logits, forget_clabels)

    loss = loss_retain - 0.001*loss_forget  # Calculate loss
    return loss, loss_retain, loss_forget