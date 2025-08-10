import copy
import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import unlearn
import utils
from models import resnet, vit
import argparse


def save_gradient_ratio(data_loaders, model, criterion, args, save_dir):
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    gradients = {}

    forget_loader = data_loaders
    model.eval()

    for name, param in model.named_parameters():
        gradients[name] = torch.zeros_like(param)

    for i, (image, target) in enumerate(forget_loader):
        image = image.to(device)
        target = target.to(device)

        # compute output
        output_clean = model(image)
        loss = - criterion(output_clean, target)

        optimizer.zero_grad()
        loss.backward()

        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None:
                    gradients[name] += param.grad.data

    with torch.no_grad():
        for name in gradients:
            gradients[name] = torch.abs_(gradients[name])

    threshold_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    for i in threshold_list:
        sorted_dict_positions = {}
        hard_dict = {}

        # Concatenate all tensors into a single tensor
        all_elements = - torch.cat([tensor.flatten() for tensor in gradients.values()])

        # Calculate the threshold index for the top 10% elements
        threshold_index = int(len(all_elements) * i)

        # Calculate positions of all elements
        positions = torch.argsort(all_elements)
        ranks = torch.argsort(positions)

        start_index = 0
        for key, tensor in gradients.items():
            num_elements = tensor.numel()  # len
            # tensor_positions = positions[start_index: start_index + num_elements]
            tensor_ranks = ranks[start_index : start_index + num_elements]

            sorted_positions = tensor_ranks.reshape(tensor.shape)
            sorted_dict_positions[key] = sorted_positions

            # Set the corresponding elements to 1
            threshold_tensor = torch.zeros_like(tensor_ranks)
            threshold_tensor[tensor_ranks < threshold_index] = 1
            threshold_tensor = threshold_tensor.reshape(tensor.shape)
            hard_dict[key] = threshold_tensor
            start_index += num_elements

        torch.save(hard_dict, os.path.join(save_dir, "mask_salun{}.pt".format(i)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--unlearn_name',
        type=str,
        default='teacher')
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
    parser.add_argument('--milestones', type=str, default=None)  # [82,122,163] for retrain 200 epochs None
    parser.add_argument('--seed', type=int, default=10)

    parser.add_argument("--msteps", type=int, default=5, help="scrub")
    parser.add_argument("--kd_T", type=float, default=2, help="scrub")
    parser.add_argument("--beta", type=float, default=0.5, help="scrub")
    parser.add_argument("--gamma", type=float, default=1.0, help="scrub")

    parser.add_argument('--mask_path', type=str, default=None, help="salun")

    parser.add_argument('--worst_case', type=bool, default=False, help="worst_case")
    # evaluation
    parser.add_argument('--cal_sizes', type=str, default='3000')
    parser.add_argument('--alphas', type=str, default='0.05,0.1,0.15,0.2')
    args = parser.parse_args()


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # prepare dataset
    train_retain_dl, train_forget_dl, test_retain_dl, test_forget_dl, _, _, forget_class = utils.generate_dataset(args.unlearn_type,
                                                                                              args.num_classes,
                                                                                              args.batch_size,
                                                                                              args.data_dir,
                                                                                              args.data_name,
                                                                                              args.model_name,
                                                                                              args.retain_ratio,
                                                                                              worst_case=args.worst_case)

    if args.model_name == 'resnet18':
        model = resnet.ResNet18(num_classes=args.num_classes)
    elif args.model_name == 'vit':
        if args.unlearn_name == 'retrain':
            model = vit.ViT(num_classes=args.num_classes, pretrained=True)
        else:
            model = vit.ViT(num_classes=args.num_classes, pretrained=False)

    model.to(device)

    print(f"number of retain dataset {len(train_retain_dl.dataset)}")
    print(f"number of forget dataset {len(train_forget_dl.dataset)}")

    criterion = nn.CrossEntropyLoss()

    if args.worst_case==True:
        save_dir = args.model_name + '_' + args.data_name + '/mask_salun_worst_case/'
    elif args.unlearn_type == 'subclass':
        save_dir = args.model_name +'_' + args.data_name+'/mask_salun_subclass/'
    else:
        save_dir = args.model_name +'_' + args.data_name+'/mask_salun_'+str(int((1-args.retain_ratio)*100))+'/'

    os.makedirs(save_dir, exist_ok=True)
    print('\nsave_dir', save_dir, '\n')

    model.load_state_dict(torch.load(args.model_dir))


    save_gradient_ratio(train_forget_dl, model, criterion, args, save_dir)

if __name__ == "__main__":
    main()
