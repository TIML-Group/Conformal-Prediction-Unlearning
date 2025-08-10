import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import torch
from torch.utils.data import Subset, DataLoader, random_split


def find_quantile(cal_scores, n, alpha):
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    qhat = np.quantile(cal_scores, q_level, method='higher')
    return qhat


def CP_loop(q_hat, net, test_loader, device):
    test_size = len(test_loader.dataset)
    pbar = tqdm(test_loader, total=len(test_loader))
    net.eval()
    with torch.no_grad():
        set_size = 0
        empirical_coverage = 0
        sum_correct = 0
        count = 0
        for i, sample in enumerate(test_loader):
            inputs, labels = sample
            inputs, labels = inputs.to(device), labels.to(device)
            test_outputs = net(inputs)
            test_outputs = F.softmax(test_outputs, dim=1)
            _, test_pred = torch.max(test_outputs, 1)
            prediction_sets = test_outputs >= (1-q_hat)

            num_correct = (test_pred == labels).sum()
            sum_correct += num_correct
            count += len(labels)
            acc = num_correct.item() / len(labels)

            set_size_batch = prediction_sets.sum()
            set_size += set_size_batch

            empirical_coverage_batch = prediction_sets[np.arange(prediction_sets.shape[0]), labels].float().sum()
            empirical_coverage += empirical_coverage_batch

            pbar.set_description(f"acc: {acc:>f}, [{count:>d}/{test_size:>d}]")
    torch.cuda.empty_cache()
    return empirical_coverage/test_size, set_size/test_size, sum_correct/test_size


def get_calibration(net, alpha, device, cal_dl):
    '''calibration set'''
    cal_size = len(cal_dl.dataset)

    output_batch_list = []
    labels_list = []
    net.eval()
    with torch.no_grad():
        for (inputs, labels) in cal_dl:
            inputs, labels = inputs.to(device), labels.to(device)
            cal_outputs = net(inputs)
            cal_outputs = F.softmax(cal_outputs, dim=1)
            output_batch_list.append(cal_outputs)
            labels_list.append(labels)
    all_cal_outputs = torch.cat(output_batch_list)
    all_labels = torch.cat(labels_list)

    cal_scores = 1 - all_cal_outputs[np.arange(cal_size), all_labels]
    q_hat = find_quantile(cal_scores.cpu().detach().numpy(), cal_size, alpha)
    return q_hat


def get_CP_data_wise(net, alpha, train_retain_dl, train_forget_dl, test_retain_dl, test_forget_dl, device, cal_retain_dl, cal_forget_dl, batch_size=64):

    q_hat = get_calibration(net, alpha, device, cal_retain_dl)
    cov_list = []
    size_list = []
    acc_list = []
    cr_list = []

    coverage_probability, set_size, acc = CP_loop(q_hat, net, train_retain_dl, device)
    cov_list.append(coverage_probability.item())
    size_list.append(set_size.item())
    cr_list.append(coverage_probability/set_size)
    acc_list.append(acc.item())

    coverage_probability, set_size, acc = CP_loop(q_hat, net, train_forget_dl, device)
    cov_list.append(coverage_probability.item())
    size_list.append(set_size.item())
    cr_list.append(coverage_probability/set_size)
    acc_list.append(acc.item())

    coverage_probability, set_size, acc = CP_loop(q_hat, net, test_retain_dl, device)
    cov_list.append(coverage_probability.item())
    size_list.append(set_size.item())
    cr_list.append(coverage_probability/set_size)
    acc_list.append(acc.item())

    q_hat_list = [q_hat]

    return cov_list, size_list, cr_list, acc_list, q_hat_list


def get_CP_class_wise(net, alpha, train_retain_dl, train_forget_dl, test_retain_dl, test_forget_dl, device, cal_retain_dl, cal_forget_dl, batch_size=64):  ##evaluation function'

    q_hat1 = get_calibration(net, alpha, device, cal_retain_dl)
    q_hat2 = get_calibration(net, alpha, device, cal_forget_dl)

    cov_list = []
    size_list = []
    acc_list = []
    cr_list = []

    coverage_probability, set_size, acc = CP_loop(q_hat1, net, train_retain_dl, device)
    # cov_list.append(coverage_probability)
    # size_list.append(set_size)
    acc_list.append(acc)

    coverage_probability, set_size, acc = CP_loop(q_hat2, net, train_forget_dl, device)
    cov_list.append(coverage_probability)
    size_list.append(set_size)
    cr_list.append(coverage_probability/set_size)
    acc_list.append(acc)

    coverage_probability, set_size, acc = CP_loop(q_hat2, net, test_forget_dl, device)
    cov_list.append(coverage_probability)
    size_list.append(set_size)
    cr_list.append(coverage_probability/set_size)
    acc_list.append(acc)

    coverage_probability, set_size, acc = CP_loop(q_hat1, net, test_retain_dl, device)
    cov_list.append(coverage_probability)
    size_list.append(set_size)
    cr_list.append(coverage_probability/set_size)
    acc_list.append(acc)

    q_hat_list = [q_hat1, q_hat2]

    return cov_list, size_list, cr_list, acc_list, q_hat_list

