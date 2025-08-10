import os
import torch
import numpy as np
from tqdm import tqdm
from torch import nn
import argparse
import utils
from models import resnet, vit
import wandb
import time


def train_loop(dataloader, model, loss_fn, optimizer, device):
    model.train()
    size = len(dataloader.dataset)
    pbar = tqdm(dataloader, total=int(len(dataloader)))
    count = 0
    train_loss = 0.0
    train_acc = 0.0
    for batch, sample in enumerate(pbar):
        x,labels = sample
        x,labels = x.to(device), labels.to(device)

        outputs = model(x)

        loss = loss_fn(outputs, labels)
        _,pred = torch.max(outputs,1)
        num_correct = (pred == labels).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss = loss.item()
        acc = num_correct.item()/len(labels)
        count += len(labels)
        train_loss += loss*len(labels)
        train_acc += num_correct.item()
        pbar.set_description(f"loss: {loss:>f}, acc: {acc:>f}, [{count:>d}/{size:>d}]")
    return train_loss/count, train_acc/count

def test_loop(dataloader, model, loss_fn, device):
    model.eval()
    size = len(dataloader.dataset)

    pbar = tqdm(dataloader, total=int(len(dataloader)))
    count = 0
    test_loss = 0.0
    test_acc = 0.0
    with torch.no_grad():
        for batch, sample in enumerate(pbar):
            x,labels = sample
            x,labels = x.to(device), labels.to(device)

            outputs = model(x)
            loss = loss_fn(outputs, labels)
            _,pred = torch.max(outputs,1)
            num_correct = (pred == labels).sum()
            
            loss = loss.item()
            acc = num_correct.item()/len(labels)
            count += len(labels)
            test_loss += loss*len(labels)
            test_acc += num_correct.item()
            pbar.set_description(f"loss: {loss:>f}, acc: {acc:>f}, [{count:>d}/{size:>d}]")
    return test_loss/count, test_acc/count
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='resnet18')
    parser.add_argument('--data_name', type=str, default='tiny_imagenet')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--num_classes', type=int, default=200)
    parser.add_argument('--milestones', type=str, default=None)
    parser.add_argument('--seed', type=int, default=10)
    args = parser.parse_args()

    utils.setup_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = './{}_{}/{}_{}/'.format(args.model_name, args.data_name, args.num_epochs, args.learning_rate)
    model_name = args.model_name
    batch_size = args.batch_size
    n_epochs = args.num_epochs
    lr = args.learning_rate

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    train_loader, test_loader, train_ds, test_ds, _ = utils.get_dataset(
        args.batch_size,
        root=args.data_dir,
        data_name=args.data_name,
        model_name=args.model_name
    )

    if args.model_name == 'resnet18':
        model = resnet.ResNet18(num_classes=args.num_classes).to(device)
    elif args.model_name == 'vit':
        model = vit.ViT(num_classes=args.num_classes, pretrained=True).to(device)

    if torch.cuda.is_available() and torch.cuda.device_count()>1:
        model = torch.nn.DataParallel(model)
    
    criterion = nn.CrossEntropyLoss().to(device)

    weight_p, bias_p = [],[]
    for name, p in model.named_parameters():
        if 'bias' in name:
            bias_p +=[p]
        else:
            weight_p +=[p]
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    if args.milestones:
        milestones = np.array(args.milestones.split(',')).astype(int)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1, last_epoch=-1)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs)

    idx_best_loss = 0
    idx_best_acc = 0
    
    log_train_loss = []
    log_train_acc = []
    log_test_loss = []
    log_test_acc = []

    proj_name = '{}_{}_epoch{}_lr{}'.format(args.model_name, args.data_name, args.num_epochs, args.learning_rate)
    watermark = "{}_lr{}".format(args.model_name, args.learning_rate)
    wandb.init(project=proj_name, name=watermark)
    wandb.config.update(args)
    wandb.watch(model)

    for epoch in range(1, n_epochs+1):
        start = time.time()
        print("===> Epoch {}/{}, learning rate: {}".format(epoch, n_epochs, scheduler.get_last_lr()))
        train_loss, train_acc = train_loop(train_loader, model, criterion, optimizer, device)
        test_loss, test_acc = test_loop(test_loader, model, criterion, device)
        print("Training loss: {:f}, acc: {:f}".format(train_loss, train_acc))
        print("Test loss: {:f}, acc: {:f}".format(test_loss, test_acc))
        scheduler.step()
        torch.cuda.empty_cache()

        log_train_loss.append(train_loss)
        log_train_acc.append(train_acc)
        log_test_loss.append(test_loss)
        log_test_acc.append(test_acc)

        wandb.log(
            {'epoch': epoch, 'train_loss': train_loss, 'train_acc': train_acc, 'val_loss': test_loss, "val_acc": test_acc,
             "lr": optimizer.param_groups[0]["lr"],
             "epoch_time": time.time() - start})

        if test_loss <= log_test_loss[idx_best_loss]:
            print("Save loss-best model.")
            torch.save(model.state_dict(), os.path.join(save_dir, 'loss_model_{}_lr{}_epoch{}.pth'.format(args.model_name, args.learning_rate, args.num_epochs)))
            idx_best_loss = epoch - 1
        
        if test_acc >= log_test_acc[idx_best_acc]:
            print("Save acc-best model.")
            torch.save(model.state_dict(), os.path.join(save_dir, 'acc_model_{}_lr{}_epoch{}.pth'.format(args.model_name, args.learning_rate, args.num_epochs)))
            idx_best_acc = epoch - 1
        print("")

        if epoch%10 == 0:
            torch.save(model.state_dict(),
                       os.path.join(save_dir, 'final_model_{}_lr{}_epoch{}.pth'.format(args.model_name,
                                                                                       args.learning_rate,
                                                                                       epoch)))

    wandb.save("wandb_{}_{}_{}_{}.h5".format(args.model_name, args.data_name, args.num_epochs, args.learning_rate))

    print("=============================================================")

    print("Loss-best model training loss: {:f}, acc: {:f}".format(log_train_loss[idx_best_loss], log_train_acc[idx_best_loss]))   
    print("Loss-best model test loss: {:f}, acc: {:f}".format(log_test_loss[idx_best_loss], log_test_acc[idx_best_loss]))                
    print("Acc-best model training loss: {:4f}, acc: {:f}".format(log_train_loss[idx_best_acc], log_train_acc[idx_best_acc]))  
    print("Acc-best model test loss: {:f}, acc: {:f}".format(log_test_loss[idx_best_acc], log_test_acc[idx_best_acc]))              
    print("Final model training loss: {:f}, acc: {:f}".format(log_train_loss[-1], log_train_acc[-1]))                 
    print("Final model test loss: {:f}, acc: {:f}".format(log_test_loss[-1], log_test_acc[-1]))
    torch.save(model.state_dict(), os.path.join(save_dir, 'final_model_{}_lr{}_epoch{}.pth'.format(args.model_name,
                                                                                                 args.learning_rate,
                                                                                                 args.num_epochs)))