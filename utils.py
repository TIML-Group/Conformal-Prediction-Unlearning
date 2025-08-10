import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader, random_split
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import skimage as sk
from skimage.filters import gaussian
# import torchvision.transforms.functional
from PIL import ImageFilter
# import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random
import pickle
import torchvision.datasets
from torchvision.datasets import CIFAR100

class GaussianNoise(object):
    def __init__(self, severity=3):
        self.std_levels = [0.04, 0.06, .08, .09, .10]
        self.std = self.std_levels[severity - 1]

    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std
        noisy_tensor = tensor + noise
        return torch.clamp(noisy_tensor, 0., 1.)


class GaussianBlur(object):
    def __init__(self, severity=3):
        self.radius_levels = [.4, .6, 0.7, .8, 1]
        self.radius = self.radius_levels[severity - 1]

    def __call__(self, tensor):
        img = torchvision.transforms.functional.to_pil_image(tensor)
        img = img.filter(ImageFilter.GaussianBlur(self.radius))
        return torchvision.transforms.functional.to_tensor(img)

def generate_dataset(unlearn_type, num_classes, batch_size, root, data_name, model_name, retain_ratio=0.9, shuffle=True, worst_case=False):

    data_loader_train, data_loader_test, train_ds, test_ds, cal_ds = get_dataset(batch_size, root, data_name, model_name, shuffle, worst_case)

    if worst_case==True:
        with open('./data/worst_case/cifar10_4500_forget.pkl', 'rb') as f:
            forget_idx = pickle.load(f)
        forget_ds = Subset(train_ds, forget_idx)

        with open('./data/worst_case/cifar10_4500_remain.pkl', 'rb') as f:
            retain_idx = pickle.load(f)
        retain_ds = Subset(train_ds, retain_idx)

        train_ds = Subset(train_ds, forget_idx+retain_idx)

        retain_loader = DataLoader(retain_ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)
        forget_loader = DataLoader(forget_ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        retain_cal_loader = DataLoader(cal_ds, batch_size=batch_size, shuffle=False, num_workers=0)

        return retain_loader, forget_loader, test_loader, None, retain_cal_loader, None, -1


    if unlearn_type == 'random':
        dataset_len = len(train_ds)
        retain_size = int(dataset_len * retain_ratio)
        forget_size = dataset_len - retain_size
        retain_ds, forget_ds = random_split(train_ds, [retain_size, forget_size])
        retain_loader = DataLoader(retain_ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)
        forget_loader = DataLoader(forget_ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        retain_cal_loader = DataLoader(cal_ds, batch_size=batch_size, shuffle=False, num_workers=0)

        return (retain_loader, forget_loader,
                test_loader, None,
                retain_cal_loader, None, -1)

    elif unlearn_type == 'class':
        random_class = random.randint(0, num_classes-1)
        forget_indices_train = [i for i in range(len(train_ds)) if train_ds.targets[i] == random_class]
        retain_indices_train = [i for i in range(len(train_ds)) if train_ds.targets[i] != random_class]
        forget_indices_test = [i for i in range(len(test_ds)) if test_ds.targets[i] == random_class]
        retain_indices_test = [i for i in range(len(test_ds)) if test_ds.targets[i] != random_class]
        forget_indices_cal = [i for i in range(len(cal_ds)) if cal_ds.targets[i] == random_class]
        retain_indices_cal = [i for i in range(len(cal_ds)) if cal_ds.targets[i] != random_class]

        forget_train_ds = Subset(train_ds, forget_indices_train)
        retain_train_ds = Subset(train_ds, retain_indices_train)
        forget_test_ds = Subset(test_ds, forget_indices_test)
        retain_test_ds = Subset(test_ds, retain_indices_test)
        forget_cal_ds = Subset(cal_ds, forget_indices_cal)
        retain_cal_ds = Subset(cal_ds, retain_indices_cal)

        train_forget_dataloader = DataLoader(forget_train_ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)
        train_retain_dataloader = DataLoader(retain_train_ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)
        test_forget_dataloader = DataLoader(forget_test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        test_retain_dataloader = DataLoader(retain_test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        cal_forget_dataloader = DataLoader(forget_cal_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        cal_retain_dataloader = DataLoader(retain_cal_ds, batch_size=batch_size, shuffle=False, num_workers=0)

        return (train_retain_dataloader, train_forget_dataloader,
                test_retain_dataloader, test_forget_dataloader,
                cal_retain_dataloader, cal_forget_dataloader, random_class)

    elif unlearn_type == 'subclass':
        random_class = 69
        classwise_train, classwise_test = get_classwise_ds(
            train_ds, 100
        ), get_classwise_ds(test_ds, 100)

        (
            retain_train_ds,
            retain_test_ds,
            forget_train_ds,
            forget_test_ds,
        ) = build_retain_forget_sets(
            classwise_train, classwise_test, 100, random_class
        )

        test_forget_dataloader = DataLoader(forget_test_ds, batch_size)
        test_retain_dataloader = DataLoader(retain_test_ds, batch_size)

        train_forget_dataloader = DataLoader(forget_train_ds, batch_size)
        train_retain_dataloader = DataLoader(retain_train_ds, batch_size, shuffle=True)


        forget_indices_cal = [i for i in range(len(cal_ds)) if cal_ds.targets[i] == random_class]
        retain_indices_cal = [i for i in range(len(cal_ds)) if cal_ds.targets[i] != random_class]
        forget_cal_ds = Subset(cal_ds, forget_indices_cal)
        retain_cal_ds = Subset(cal_ds, retain_indices_cal)
        cal_forget_dataloader = DataLoader(forget_cal_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        cal_retain_dataloader = DataLoader(retain_cal_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        return (train_retain_dataloader, train_forget_dataloader, test_retain_dataloader, test_forget_dataloader, cal_retain_dataloader, cal_forget_dataloader, -2)


def get_dataset(batch_size, root, data_name, model_name, shuffle=True, worst_case=False):

    if data_name == 'cifar10':
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        image_size = 32
    elif data_name == 'cifar20':
        normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
        image_size = 32
    elif data_name == 'cifar100':
        normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
        image_size = 32
    elif data_name == 'tiny_imagenet':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image_size = 64
    else:
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
        image_size = 32

    if model_name == 'resnet':
        transform_train_list = [
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
        transform_test_list = [
            transforms.ToTensor(),
            transforms.Resize((image_size, image_size)),
        ]
    elif model_name == 'vit':
        transform_train_list = [
            transforms.Resize(224),  # Resize images to 224x224 for ViT
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(224, padding=4),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
        ]
        transform_test_list = [
            transforms.Resize(224),  # Resize images to 224x224 for ViT
            transforms.ToTensor(),
        ]
    else:
        transform_train_list = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
        ]
        transform_test_list = [
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ]

    transform_train_list.append(normalize)
    transform_test_list.append(normalize)

    transform_train = transforms.Compose(transform_train_list)
    transform_test = transforms.Compose(transform_test_list)

    # Load dataset
    if data_name == 'cifar10':
        data_train = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)
        test = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)
        data_test, data_cal = random_split(test, [len(test)-2000, 2000])
    elif data_name == 'cifar20':
        data_train = Cifar20(root=root, train=True, download=True, transform=transform_train)
        test = Cifar20(root=root, train=False, download=True, transform=transform_test)
        data_test, data_cal = random_split(test, [len(test)-2000, 2000])
    elif data_name == 'cifar100':
        data_train = torchvision.datasets.CIFAR100(root=root, train=True, download=True, transform=transform_train)
        test = torchvision.datasets.CIFAR100(root=root, train=False, download=True, transform=transform_test)
        data_test, data_cal = random_split(test, [len(test)-2000, 2000])

    elif data_name == 'tiny_imagenet':
        data_train = torchvision.datasets.ImageFolder(root='./data/tiny-imagenet-200/train', transform=transform_train)
        data_test = torchvision.datasets.ImageFolder(root='./data/tiny-imagenet-200/val', transform=transform_test)
        data_cal = torchvision.datasets.ImageFolder(root='./data/tiny-imagenet-200/cal', transform=transform_test)


    data_loader_train = DataLoader(dataset=data_train,
                                   batch_size=batch_size,
                                   shuffle=shuffle,
                                   pin_memory=True)
    data_loader_test = DataLoader(dataset=data_test,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  pin_memory=True)

    return data_loader_train, data_loader_test, data_train, data_test, data_cal


'''
ACC metric
'''
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.sum(preds == labels).item(), len(preds), torch.tensor(torch.sum(preds == labels).item() / len(preds)) * 100

def evaluate_acc_batch(model, batch, device):
    images, clabels = batch
    images, clabels = images.to(device), clabels.to(device)
    out = model(images)
    return accuracy(out, clabels)

def evaluate_acc(model, val_loader, device):
    model.eval()
    corr, total = 0, 0
    with torch.no_grad():
        for batch in val_loader:
            corr_, total_, _ = evaluate_acc_batch(model, batch, device)
            corr += corr_
            total += total_
    torch.cuda.empty_cache()
    return corr/total

'''perturbation'''
def gaussian_noise(x, severity=1):
    c = [0.04, 0.06, .08, .09, .10][severity - 1]
    return torch.tensor(x + np.random.normal(size=x.shape, scale=c), dtype=torch.float32)

def gaussian_blur(x, severity=1):
    c = [.4, .6, 0.7, .8, 1][severity - 1]

    x = gaussian(np.array(x), sigma=c)
    return torch.tensor(x)

def contrast(x, severity=1):
    c = [.95, .5, .4, .3, 0.15][severity - 1]
    x = np.array(x)
    means = np.mean(x, axis=(0, 1), keepdims=True)
    return torch.tensor((x - means)*c + means)

def impulse_noise(x, severity=1):
    c = [.01, .02, .03, .05, .07][severity - 1]

    x = sk.util.random_noise(np.array(x), mode='s&p', amount=c)
    return torch.tensor(x)

def speckle_noise(x, severity=1):
    c = [.06, .1, .12, .16, .2][severity - 1]

    x = np.array(x)
    return torch.tensor(x + x * np.random.normal(size=x.shape, scale=c))

def perturbation(test_ds, batch_size, type='gaussian_noise', severity=1):
    labels = []
    img_p = []
    for (input, label) in test_ds:
        labels.append(label)
        if type == 'gaussian_noise':
            input_p = gaussian_noise(input, severity)
        elif type == 'gaussian_blur':
            input_p = gaussian_blur(input, severity)
        elif type == 'contrast':
            input_p = contrast(input, severity)
        elif type == 'impulse_noise':
            input_p = impulse_noise(input, severity)
        elif type == 'speckle_noise':
            input_p = speckle_noise(input, severity)
        img_p.append(input_p)
    img_p_tensor = torch.stack(img_p)
    labels_tensor = torch.tensor(labels)
    test_ds_p = torch.utils.data.TensorDataset(img_p_tensor, labels_tensor)
    test_dl_p = torch.utils.data.DataLoader(test_ds_p, batch_size=batch_size, shuffle=False, num_workers=0)
    return test_ds_p, test_dl_p
    return

'''unlearn'''
def setup_seed(seed):
    print("setup random seed = {}".format(seed))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def generate_excel_name(model_dir, corruption_type, corruption_level, unlearn_type, unlearn_name):
    if unlearn_type is None:
        unlearn_type = 'None'
    sheet_name = f'{unlearn_name}_{unlearn_type}_{corruption_type}_{corruption_level}'
    xlsx_path = '/'.join(model_dir.split('/')[0:-1])+'/'+sheet_name+'final.xls'
    return sheet_name, xlsx_path

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def show_images(dataset_original, index):
    original_img, original_label = dataset_original[index]
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))

    axs[0].imshow(original_img)
    axs[0].set_title(f'Original Image\nLabel: {original_label}')

    plt.show()

'''Subclass Forgetting'''
def get_classwise_ds(ds, num_classes):
    classwise_ds = {}
    for i in range(num_classes):
        classwise_ds[i] = []

    for img, label, clabel in ds:
        classwise_ds[label].append((img, label, clabel))
    return classwise_ds

def build_retain_forget_sets(
    classwise_train, classwise_test, num_classes, forget_class
):
    # Getting the forget and retain validation data
    forget_valid = []
    for cls in range(num_classes):
        if cls == forget_class:
            for img, label, clabel in classwise_test[cls]:
                forget_valid.append((img, clabel))

    retain_valid = []
    for cls in range(num_classes):
        if cls != forget_class:
            for img, label, clabel in classwise_test[cls]:
                retain_valid.append((img, clabel))

    forget_train = []
    for cls in range(num_classes):
        if cls == forget_class:
            for img, label, clabel in classwise_train[cls]:
                forget_train.append((img, clabel))

    retain_train = []
    for cls in range(num_classes):
        if cls != forget_class:
            for img, label, clabel in classwise_train[cls]:
                retain_train.append((img, clabel))

    return (retain_train, retain_valid, forget_train, forget_valid)


class Cifar100(CIFAR100):
    def __init__(self, root, train, download, transform):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        return x, torch.Tensor([]), y

'''Worst-Case Forgetting'''
class Cifar20(CIFAR100):
    def __init__(self, root, train, download, transform):
        super().__init__(root=root, train=train, download=download, transform=transform)

        # This map is for the matching of subclases to the superclasses. E.g., rocket (69) to Vehicle2 (19:)
        # Taken from https://github.com/vikram2000b/bad-teaching-unlearning
        self.coarse_map = {
            0: [4, 30, 55, 72, 95],
            1: [1, 32, 67, 73, 91],
            2: [54, 62, 70, 82, 92],
            3: [9, 10, 16, 28, 61],
            4: [0, 51, 53, 57, 83],
            5: [22, 39, 40, 86, 87],
            6: [5, 20, 25, 84, 94],
            7: [6, 7, 14, 18, 24],
            8: [3, 42, 43, 88, 97],
            9: [12, 17, 37, 68, 76],
            10: [23, 33, 49, 60, 71],
            11: [15, 19, 21, 31, 38],
            12: [34, 63, 64, 66, 75],
            13: [26, 45, 77, 79, 99],
            14: [2, 11, 35, 46, 98],
            15: [27, 29, 44, 78, 93],
            16: [36, 50, 65, 74, 80],
            17: [47, 52, 56, 59, 96],
            18: [8, 13, 48, 58, 90],
            19: [41, 69, 81, 85, 89],
        }

    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        coarse_y = None
        for i in range(20):
            for j in self.coarse_map[i]:
                if y == j:
                    coarse_y = i
                    break
            if coarse_y != None:
                break
        if coarse_y == None:
            assert coarse_y != None
        return x, y, coarse_y
