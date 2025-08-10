import torch
from torch.utils.data import DataLoader, Subset, dataset, Dataset, random_split
from tqdm import tqdm
import utils
import torch.nn as nn
import wandb
import time
import numpy as np
from torch.nn import functional as F
import random


'''cpu loss'''
def get_cpu_loss(out_sofmax, clabels, q_hat, delta, device, unlearn_type):
    score = 1 - out_sofmax[:, clabels]
    loss = torch.clamp(q_hat - score + delta, min=0)
    return loss.mean()


'''calibration set'''
def find_quantile(cal_scores, n, alpha):
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    qhat = np.quantile(cal_scores, q_level, method='higher')
    return qhat


def get_calibration(net, alpha, cal_loader, device, unlearn_type):
    cal_size = len(cal_loader.dataset)
    '''calibration set'''
    output_batch_list = []
    labels_list = []
    net.eval()
    for (inputs, labels) in cal_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        cal_outputs = net(inputs)
        cal_outputs = F.softmax(cal_outputs, dim=1)
        output_batch_list.append(cal_outputs)
        labels_list.append(labels)

    all_cal_outputs = torch.cat(output_batch_list).to(device)
    all_labels = torch.cat(labels_list)
    cal_scores = 1 - all_cal_outputs[np.arange(cal_size), all_labels]

    q_hat = find_quantile(cal_scores.cpu().detach().numpy(), cal_size, alpha)
    return q_hat


'''basic func'''
def training_step(model, batch, criterion, device):
    images, clabels = batch
    images, clabels = images.to(device), clabels.to(device)
    out = model(images)  # Generate predictions
    loss = criterion(out, clabels)  # Calculate loss
    _, pred = torch.max(out, 1)
    num_correct = (pred == clabels).sum()
    acc = num_correct.item() / len(clabels)
    return loss, acc


def training_step_cpu_loss(model, batch, criterion, q_hat, delta, lamda, device, unlearn_type, unlearn_name):
    images, clabels, labels = batch
    images, clabels, labels = images.to(device), clabels.to(device), labels.to(device)
    out = model(images)  # Generate predictions

    retain_mask = (labels == 0)
    forget_mask = (labels == 1)

    # original loss
    retain_logits = out[retain_mask]
    retain_clabels = clabels[retain_mask]
    loss_ori = criterion(retain_logits, retain_clabels)

    # forget loss
    forget_logits = out[forget_mask]
    forget_clabels = clabels[forget_mask]

    forget_softmax = F.softmax(forget_logits, dim=1)
    loss_forget = get_cpu_loss(forget_softmax, forget_clabels, q_hat, delta, device, unlearn_type)

    loss = loss_ori + lamda*loss_forget

    _, pred = torch.max(out, 1)
    num_correct = (pred == clabels).sum()
    acc = num_correct.item() / len(clabels)
    return loss, acc


def fit_one_cycle(
    epochs, model, train_loader, forget_loader, test_loader, cal_dl, device, delta, alpha, lamda, lr, unlearn_type, milestones, unlearn_name='retrain', mask=None
):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.9, weight_decay=5e-4)
    if milestones is None:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=0.1, last_epoch=-1
        )
    batch_size = forget_loader.batch_size
    cp_loss_data = LossData(forget_data=forget_loader.dataset, retain_data=train_loader.dataset)
    cp_loss_dl = DataLoader(
        cp_loss_data, batch_size=batch_size, shuffle=True, pin_memory=True
    )

    model.train()

    train_list = []
    test_list = []
    forget_list = []
    for epoch in range(epochs):
        start = time.time()
        pbar = tqdm(cp_loss_dl, total=len(cp_loss_dl))
        acc_list = []
        q_hat = get_calibration(model, alpha, cal_dl, device, unlearn_type)

        for batch_i, batch in enumerate(pbar):
            loss, acc = training_step_cpu_loss(model, batch, criterion, q_hat,
                                              delta=delta, lamda=lamda, device=device,
                                              unlearn_type=unlearn_type, unlearn_name=unlearn_name)

            loss.backward()

            if mask:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name]

            optimizer.step()
            optimizer.zero_grad()
            acc_list.append(acc)

        scheduler.step()

        forget_acc = utils.evaluate_acc(model, forget_loader, device)
        test_acc = utils.evaluate_acc(model, test_loader, device)
        wandb.log(
            {'epoch': epoch, 'train_acc': np.mean(acc_list), 'forget_acc': forget_acc, 'test_acc': test_acc,
             "lr": optimizer.param_groups[0]["lr"],
             "epoch_time": time.time() - start})
        train_list.append(np.mean(acc_list))
        test_list.append(test_acc)
        forget_list.append(forget_acc)

        torch.cuda.empty_cache()

    return

def retrain(
    model,
    train_retain_dl,
    train_forget_dl,
    test_retain_dl,
    test_forget_dl,
    cal_dl,
    device,
    num_epochs,
    learning_rate,
    delta,
    alpha,
    lamda,
    unlearn_type,
    milestones,
    **kwargs,
):

    fit_one_cycle(
        num_epochs, model, train_retain_dl, train_forget_dl, test_retain_dl, cal_dl, device, delta, alpha, lamda,
        lr=learning_rate,
        unlearn_type=unlearn_type,
        milestones=milestones,
        unlearn_name='retrain',
    )

    return utils.evaluate_acc(model, test_retain_dl, device)

def finetune(
    model,
    train_retain_dl,
    train_forget_dl,
    test_retain_dl,
    test_forget_dl,
    cal_dl,
    device,
    num_epochs,
    learning_rate,
    delta,
    alpha,
    lamda,
    unlearn_type,
    milestones,
    **kwargs,
):
    fit_one_cycle(
        num_epochs, model, train_retain_dl, train_forget_dl, test_retain_dl, cal_dl, device, delta, alpha, lamda,
        lr=learning_rate,
        unlearn_type=unlearn_type,
        milestones=milestones,
        unlearn_name='finetune',
    )

    return utils.evaluate_acc(model, test_retain_dl, device)


def RL(
    model,
    train_retain_dl,
    train_forget_dl,
    test_retain_dl,
    test_forget_dl,
    cal_dl,
    num_classes,
    device,
    num_epochs,
    batch_size,
    learning_rate,
    delta,
    alpha,
    lamda,
    unlearn_type,
    milestones,
    unlearn_name,
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

    unlearning_forgetset = DataLoader(
        unlearning_trainset, batch_size=batch_size, pin_memory=True, shuffle=True
    )

    fit_one_cycle(
        num_epochs, model, unlearning_forgetset, train_forget_dl,
        test_retain_dl, cal_dl, device, delta, alpha, lamda,
        lr=learning_rate,
        unlearn_type=unlearn_type,
        milestones=milestones,
        unlearn_name=unlearn_name,
    )

    return utils.evaluate_acc(model, test_retain_dl, device)


def salun(
    model,
    train_retain_dl,
    train_forget_dl,
    test_retain_dl,
    test_forget_dl,
    cal_dl,
    num_classes,
    device,
    num_epochs,
    batch_size,
    learning_rate,
    delta,
    alpha,
    lamda,
    unlearn_type,
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

    unlearning_forgetset = DataLoader(
        unlearning_trainset, batch_size=batch_size, pin_memory=True, shuffle=True
    )
    fit_one_cycle(
        num_epochs, model, unlearning_forgetset, train_forget_dl, test_retain_dl,
        cal_dl, device, delta, alpha, lamda,
        lr=learning_rate,
        unlearn_type=unlearn_type,
        milestones=milestones,
        mask=mask,
        unlearn_name='salun',
    )

    return utils.evaluate_acc(model, test_retain_dl, device)


###############################################

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
            return x, y, 1
        else:
            x, y = self.retain_data[index - self.forget_len]
            return x, y, 0
