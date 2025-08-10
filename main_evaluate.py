from torch.utils.data import Subset
import utils
import torch
import argparse
from models import resnet, vit
import numpy as np
import xlwt
from metrics import CR, MIACR
import os
import random


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--unlearn_name',
        type=str)
    parser.add_argument(
        '--unlearn_type',
        type=str,
        default='random')
    parser.add_argument('--model_name', type=str, default='resnet18')
    parser.add_argument('--data_name', type=str, default='cifar10')
    parser.add_argument('--data_dir', type=str, default='../data')
    parser.add_argument('--model_dir', type=str, default='./retraining_saved/retraining_final_model.pth')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--retain_ratio', type=float, default=0.9)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--cal_sizes', type=str, default='3000')
    parser.add_argument('--alphas', type=str, default='0.05')

    parser.add_argument('--worst_case', type=bool, default=None, help="worst_case")

    args = parser.parse_args()
    utils.setup_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_retain_dl, train_forget_dl, test_retain_dl, test_forget_dl, retain_cal_loader, forget_cal_loader, forget_class = utils.generate_dataset(args.unlearn_type,
                                                                                              args.num_classes,
                                                                                              args.batch_size,
                                                                                              args.data_dir,
                                                                                              args.data_name,
                                                                                              args.model_name,
                                                                                              args.retain_ratio,
                                                                                              worst_case=args.worst_case)

    if args.model_name == 'resnet18':
        net = resnet.ResNet18(num_classes=args.num_classes)
    elif args.model_name == 'vit':
        net = vit.ViT(num_classes=args.num_classes, pretrained=False)


    def load_model(net, model_path):
        state_dict = torch.load(model_path, weights_only=True)
        if list(state_dict.keys())[0].startswith("module.") and not isinstance(net, torch.nn.DataParallel):
            state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
        elif not list(state_dict.keys())[0].startswith("module.") and isinstance(net, torch.nn.DataParallel):
            state_dict = {"module." + k: v for k, v in state_dict.items()}
        net.load_state_dict(state_dict)


    load_model(net, args.model_dir)

    if torch.cuda.is_available() and torch.cuda.device_count()>1:
        print("Using {} GPUs.".format(torch.cuda.device_count()))
        net = torch.nn.DataParallel(net)

    net = net.to(device)

    cal_size_list = np.array(args.cal_sizes.split(',')).astype(int)
    alpha_list = np.array(args.alphas.split(',')).astype(float)

    def generate_excel_name(model_dir, corruption_type, corruption_level, unlearn_type, unlearn_name):
        unlearn_type = 'None' if unlearn_type is None else unlearn_type
        sheet_name = f'{unlearn_name}_{unlearn_type}_{corruption_type}_{corruption_level}'
        xlsx_path = '/'.join(model_dir.split('/')[0:-1]) + '/' + sheet_name + '_cpu.xls'
        return sheet_name, xlsx_path

    sheet_name, xlsx_path = generate_excel_name(args.model_dir, args.corruption_type, args.corruption_level, args.unlearn_type, args.unlearn_name)
    file_name = '.'+args.model_dir.split('.')[-2]

    workbook = xlwt.Workbook()
    sheet = workbook.add_sheet(sheet_name)

    row = 0
    for alpha in alpha_list:
        if args.unlearn_type in ['random']:
            cov_list, size_list, cr_list, acc_list, q_hat_list = CR.get_CP_data_wise(net, alpha, train_retain_dl,
                                                                            train_forget_dl, test_retain_dl,
                                                                            test_forget_dl,
                                                                            device, retain_cal_loader, forget_cal_loader,
                                                                            batch_size=args.batch_size)
        elif args.unlearn_type in ['class', 'subclass']:
            cov_list, size_list, cr_list, acc_list, q_hat_list = CR.get_CP_class_wise(net, alpha, train_retain_dl,
                                                                            train_forget_dl, test_retain_dl,
                                                                            test_forget_dl,
                                                                            device, retain_cal_loader, forget_cal_loader,
                                                                            batch_size=args.batch_size)


        row += 1
        value = [alpha]

        value.extend(acc_list)
        value.extend(cov_list)
        value.extend(cr_list)
        value.extend(size_list)

        for i, v in enumerate(value):
            sheet.write(row, i, float(v))


    test_len = len(test_retain_dl.dataset)
    train_len = len(train_retain_dl.dataset)

    all_indices = list(range(train_len))
    shadow_indices = random.sample(all_indices, test_len)

    shadow_train_ds = torch.utils.data.Subset(train_retain_dl.dataset, shadow_indices)
    shadow_train_loader = torch.utils.data.DataLoader(shadow_train_ds, batch_size=args.batch_size, shuffle=False)

    m = MIACR.SVC_MIA(
        shadow_train=shadow_train_loader,
        shadow_test=test_retain_dl,
        target_train=train_forget_dl,
        target_test=test_forget_dl,
        cal_dl=retain_cal_loader,
        model=net,
        device=device
    )
    MIACR_res = np.array(m)
    if not MIACR_res[-1]:
        res_ori = MIACR_res[:-3]
    row += 1
    for i, value in enumerate(MIACR_res):
        sheet.write(row, i, float(value))
    workbook.save(xlsx_path)