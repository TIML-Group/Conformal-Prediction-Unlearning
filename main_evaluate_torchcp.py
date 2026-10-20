from torch.utils.data import Subset
import utils
import torch
import argparse
from models import resnet, vit
import numpy as np
import xlwt
from metrics import CP, SVC_MIA, ARC
import os
import random
from torchcp.classification.score import LAC, APS, SAPS, RAPS, EntmaxScore
from torchcp.classification.predictor import SplitPredictor

from torch.utils.data import Subset, DataLoader, random_split

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
    parser.add_argument('--cal_sizes', type=str, default='2000')
    parser.add_argument('--alphas', type=str, default='0.05')

    parser.add_argument('--worst_case', type=bool, default=None, help="worst_case")

    args = parser.parse_args()

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

    utils.setup_seed(args.seed)

    load_model(net, args.model_dir)
    # net.load_state_dict(torch.load(args.model_dir, weights_only=True))

    if torch.cuda.is_available() and torch.cuda.device_count()>1:
        print("Using {} GPUs.".format(torch.cuda.device_count()))
        net = torch.nn.DataParallel(net)

    net = net.to(device)
    net.eval()

    cal_size_list = np.array(args.cal_sizes.split(',')).astype(int)
    alpha_list = np.array(args.alphas.split(',')).astype(float)

    def generate_excel_name(model_dir, corruption_type, corruption_level, unlearn_type, unlearn_name):
        if unlearn_type is None:
            unlearn_type = 'None'
        # sheet_name = model_dir.split('/')[-2]
        sheet_name = f'{unlearn_name}_{unlearn_type}_{corruption_type}_{corruption_level}'
        xlsx_path = '/'.join(model_dir.split('/')[0:-1]) + '/' + sheet_name + '_LAC.xls'
        return sheet_name, xlsx_path

    # sheet_name, xlsx_path = generate_excel_name(args.model_dir, args.corruption_type, args.corruption_level, args.unlearn_type, args.unlearn_name)
    sheet_name = f'{args.unlearn_name}_{args.unlearn_type}'
    file_name = '.'.join(args.model_dir.split('.')[:-1])
    print('file_name', file_name)
    # xlsx_path = os.path.join(save_dir, 'seed_'+str(args.seed)+'_mia.xls')
    xlsx_path = file_name+'_LAC.xls'
    print('xlsx_path', xlsx_path)

    workbook = xlwt.Workbook()
    sheet = workbook.add_sheet(sheet_name)
    print('\nxlsx_path\n', xlsx_path)

    predictor = SplitPredictor(score_function=EntmaxScore(), model=net, alpha=0.1, device=device)

    row = 0

    for alpha in alpha_list:
        if args.unlearn_type in ['random']:
            predictor.calibrate(retain_cal_loader, alpha)
            result_dict_test = predictor.evaluate(test_retain_dl)
            result_dict_forget = predictor.evaluate(train_forget_dl)
            print(f"result_dict_test: {result_dict_test['coverage_rate']:.4f}")
            print(f"result_dict_test: {result_dict_test['average_size']:.4f}")

            print(f"result_dict_forget: {result_dict_forget['coverage_rate']:.4f}")
            print(f"result_dict_forget: {result_dict_forget['average_size']:.4f}")

            CR_forget = result_dict_forget['average_size']/result_dict_forget['average_size']
            CR_test = result_dict_test['average_size']/result_dict_test['average_size']

        row += 1
        value = [alpha]

        value.extend([result_dict_forget['coverage_rate'], result_dict_test['coverage_rate']])
        value.extend([result_dict_forget['average_size'], result_dict_test['average_size']])
        value.extend([CR_forget, CR_test])

        for i, v in enumerate(value):
            sheet.write(row, i, float(v))


    workbook.save(xlsx_path)