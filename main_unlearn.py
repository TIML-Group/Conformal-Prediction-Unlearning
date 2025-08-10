import utils
import torch
import time
import os
import argparse
import unlearn
import numpy as np
import xlwt
from metrics import CR, MIACR
import wandb
from torch.utils.data import DataLoader, ConcatDataset, dataset
from models import resnet, vit
import random


def get_save_dir():
    if args.unlearn_type == 'random':
        save_dir = os.path.join(args.model_name +'_' + args.data_name,
                            args.unlearn_name +'_nips_forget' + str(int(100-args.retain_ratio*100)) +'_epoch' + str(args.num_epochs))
    elif args.unlearn_type == 'class':
        save_dir = os.path.join(args.model_name + '_' + args.data_name,
                                args.unlearn_name + '_nips_forget_class_epoch' + str(args.num_epochs))
    return save_dir


if __name__ == '__main__':
    flag = time.time()
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
    parser.add_argument('--model_dir', type=str, default='../resnet18-cifar10/fine_model_baseline/final_model.pth')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--retain_ratio', type=float, default=0.9)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--milestones', type=str, default=None)
    parser.add_argument('--seed', type=int, default=10)

    parser.add_argument("--msteps", type=int, default=5, help="scrub")
    parser.add_argument("--kd_T", type=float, default=2, help="scrub")
    parser.add_argument("--beta", type=float, default=1, help="scrub")
    parser.add_argument("--gamma", type=float, default=1.0, help="scrub")

    parser.add_argument('--mask_path', type=str, default=None, help="salun")

    parser.add_argument('--worst_case', type=bool, default=False, help="worst_case")

    # evaluation
    parser.add_argument('--cal_sizes', type=str, default='1000')
    parser.add_argument('--alphas', type=str, default='0.05,0.1,0.15,0.2')
    args = parser.parse_args()

    utils.setup_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = get_save_dir()


    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    train_retain_dl, train_forget_dl, test_retain_dl, test_forget_dl, retain_cal_loader, forget_cal_loader, forget_class = utils.generate_dataset(args.unlearn_type,
                                                                                              args.num_classes,
                                                                                              args.batch_size,
                                                                                              args.data_dir,
                                                                                              args.data_name,
                                                                                              args.model_name,
                                                                                              args.retain_ratio,
                                                                                              worst_case=args.worst_case)
    full_train_dl = DataLoader(
        ConcatDataset((train_retain_dl.dataset, train_forget_dl.dataset)),
        batch_size=args.batch_size,
    )

    if args.model_name == 'resnet18':
        net = resnet.ResNet18(num_classes=args.num_classes)
        unlearning_teacher = resnet.ResNet18(num_classes=args.num_classes).to(device) if args.unlearn_name == 'teacher' else None
    elif args.model_name == 'vit':
        if args.unlearn_name == 'retrain':
            net = vit.ViT(num_classes=args.num_classes, pretrained=True)
        else:
            net = vit.ViT(num_classes=args.num_classes, pretrained=False)
        unlearning_teacher = vit.ViT(num_classes=args.num_classes, pretrained=True).to(device) if args.unlearn_name == 'teacher' else None

    if torch.cuda.is_available() and torch.cuda.device_count()>1:
        print("Using {} GPUs.".format(torch.cuda.device_count()))
        net = torch.nn.DataParallel(net)

    if args.unlearn_name != 'retrain':
        net.load_state_dict(torch.load(args.model_dir))

    net = net.to(device)

    # wandb
    proj_name = '{}_{}_{}_epoch{}_seed'.format(args.model_name, args.data_name, args.unlearn_name, args.num_epochs)
    watermark = "seed{}_lr{}".format(args.seed, args.learning_rate)
    wandb.init(project=proj_name, name=watermark)
    wandb.config.update(args)
    wandb.watch(net)

    mask = None
    if args.mask_path is not None and args.unlearn_name=='salun':
        mask = torch.load(args.mask_path)

    kwargs = {
        "model": net,
        "train_retain_dl": train_retain_dl,
        "train_forget_dl": train_forget_dl,
        "test_retain_dl": test_retain_dl,
        "test_forget_dl": test_forget_dl,
        "dampening_constant": 1,   # Lambda for ssd
        "selection_weighting": 10,   # Alpha for ssd
        "num_classes": args.num_classes,
        "dataset_name": 'cifar10',
        "device": device,
        "model_name": args.model_name,
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "milestones": np.array(args.milestones.split(',')).astype(int) if args.milestones is not None else None,
        "batch_size": args.batch_size,
        "full_train_dl": full_train_dl,
        "unlearning_teacher": unlearning_teacher,
        "unlearn_type": args.unlearn_type,
        "forget_class": forget_class,
        "args": args,
        "mask": mask,
        "save_dir": save_dir
    }
    start = time.time()
    if args.unlearn_name == 'sfron':
        train_acc, forget_acc, test_acc, net = getattr(unlearn, args.unlearn_name)(
            **kwargs
        )
    else:
        train_acc, forget_acc, test_acc = getattr(unlearn, args.unlearn_name)(
            **kwargs
        )

    wandb.save("wandb_{}_{}_{}_{}_{}.h5".format(args.model_name, args.data_name, args.unlearn_name, args.num_epochs, args.learning_rate))
    torch.save(net.state_dict(), os.path.join(save_dir, 'seed_'+str(args.seed)+'.pth'))
