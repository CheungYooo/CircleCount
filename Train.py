import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset
import glob
import shutil
import time
import torch.backends.cudnn as cudnn
import random
from PIL import Image
import h5py
import cv2
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from Model import CSRNet
from Model import Net1
from Model import Net2

parser = argparse.ArgumentParser(description='PyTorch CSRNet')
parser.add_argument('--resume', default='', type=str, metavar='PATH',  # 可選參數
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-n', '--netname', type=str, metavar='', required=True,  # 必選參數
                    help='name of the net to train')
parser.add_argument('-s', '--sigma', type=int, metavar='', required=True,  # 必選參數
                    help='sigma of gaussian_filter')
args = parser.parse_args()


class CustomDataset(Dataset):
    def __init__(self, dir_list, transform=None):
        self.dir_list = dir_list
        self.transform = transform

    def __len__(self):
        return len(self.dir_list)

    def __getitem__(self, index):
        target_path = self.dir_list[index].replace('.jpg', '.h5').replace('image', 'densitymap%s' % args.sigma)
        # print('target_path：', target_path)
        img = Image.open(self.dir_list[index]).convert('RGB')
        target = h5py.File(target_path)
        target = np.asarray(target['density'])
        target = cv2.resize(target, (target.shape[1] // 8, target.shape[0] // 8),
                            interpolation=cv2.INTER_CUBIC) * 64

        target = target[np.newaxis, :]
        if self.transform is not None:
            img = self.transform(img)
        return img, target


class CustomDataset1(Dataset):
    def __init__(self, dir_list, transform=None):
        self.dir_list = dir_list
        self.transform = transform

    def __len__(self):
        return len(self.dir_list)

    def __getitem__(self, index):
        target_path = self.dir_list[index].replace('.jpg', '.h5').replace('image', 'densitymap%s' % args.sigma)
        img = Image.open(self.dir_list[index]).convert('RGB')
        target = h5py.File(target_path)
        target = np.asarray(target['density'])
        target = target[np.newaxis, :]
        if self.transform is not None:
            img = self.transform(img)
        return img, target


# crop and augment
class CustomDataset2(Dataset):
    def __init__(self, dir_list, is_train=False, transform=None):
        self.dir_list = dir_list
        self.transform = transform
        self.is_train = is_train

    def __len__(self):
        return len(self.dir_list)

    def __getitem__(self, index):
        target_path = self.dir_list[index].replace('.jpg', '.h5').replace('image', 'densitymap%s' % args.sigma)
        img = Image.open(self.dir_list[index]).convert('RGB')
        target = h5py.File(target_path)
        target = np.asarray(target['density'])

        crop_size = (int(img.size[0] / 2), int(img.size[1] / 2))
        img_ls = []
        tar_ls = []
        if self.is_train:
            patch1 = img.crop((0, 0, crop_size[0], crop_size[1]))
            img_ls.append(patch1)
            patch2 = img.crop((crop_size[0], 0, img.size[0], crop_size[1]))
            img_ls.append(patch2)
            patch3 = img.crop((0, crop_size[1], crop_size[0], img.size[1]))
            img_ls.append(patch3)
            patch4 = img.crop((crop_size[0], crop_size[1], img.size[0], img.size[1]))
            img_ls.append(patch4)
            target1 = target[0:crop_size[1], 0:crop_size[0]]
            tar_ls.append(target1)
            target2 = target[0:crop_size[1], crop_size[0]:img.size[0]]
            tar_ls.append(target2)
            target3 = target[crop_size[1]:img.size[1], 0:crop_size[0]]
            tar_ls.append(target3)
            target4 = target[crop_size[1]:img.size[1], crop_size[0]:img.size[0]]
            tar_ls.append(target4)
        else:
            img_ls.append(img)
            tar_ls.append(target)
        if self.transform is not None:
            for i, img_patch in enumerate(img_ls, 0):
                img_patch = self.transform(img_patch)
                img_ls[i] = img_patch
        return img_ls, tar_ls


def train2(model, train_loader, criterion, optimizer):
    running_loss = 0.0
    model.train()
    for batch_idx, (img_ls, tar_ls) in enumerate(train_loader):
        for patch_idx, img_patch in enumerate(img_ls):
            inputs = img_ls[patch_idx].cuda()
            targets = tar_ls[patch_idx].float().cuda().unsqueeze(0)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        if batch_idx % 20 == 19:
            TB_writer.add_scalar('train/loss', running_loss / 80, epoch * len(train_loader) + batch_idx)
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss:.4f}'
                  .format(epoch, batch_idx, len(train_loader),
                          loss=running_loss / 80))
            running_loss = 0.0


def validate(model, val_loader):
    print('Begin Test:')
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        mae = 0.0
        mse = 0.0
        for batch_idx, (imgs, targets) in enumerate(val_loader):
            imgs = imgs.cuda()
            outputs = model(imgs)
            ae = abs(outputs.cpu().sum() - targets.sum())  # absolute error
            mae += ae
            se = ae * ae
            mse += se
        mae = mae / val_ds.__len__()
        mse = np.sqrt(mse / val_ds.__len__())
        TB_writer.add_scalar('val/MSE', mse, epoch)
        TB_writer.add_scalar('val/MAE', mae, epoch)
        print('MAE {mae:.3f}  MSE {mse:.3f} '
              .format(mae=mae, mse=mse))
    return mae


def validate2(model, val_loader):
    print('Begin Test:')
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        mae = 0.0
        mse = 0.0
        for batch_idx, (img_ls, tar_ls) in enumerate(val_loader):
            for patch_idx, img_patch in enumerate(img_ls):
                inputs = img_ls[patch_idx].cuda()
                outputs = model(inputs)
                ae = abs(outputs.cpu().sum() - tar_ls[patch_idx].sum())  # absolute error
                mae += ae
                se = ae * ae
                mse += se
        mae = mae / val_ds.__len__()
        mse = np.sqrt(mse / val_ds.__len__())
        TB_writer.add_scalar('val/MSE', mse, epoch)
        TB_writer.add_scalar('val/MAE', mae, epoch)
        print('MAE {mae:.3f}  MSE {mse:.3f} '
              .format(mae=mae, mse=mse))
    return mae


def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr * (0.1 ** (epoch // 50))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(state, is_best, filename='Checkpoint/%s_sigma=%s_checkpoint.pth.tar' % (args.netname, args.sigma)):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename,
                        'Model/%s_sigma=%s.pth.tar' % (args.netname, args.sigma))


# log_dir = 'Logs/%s_sigma=%s/' % (args.netname, args.sigma) + datetime.now().strftime('%Y%m%d-%H%M%S')
log_dir = 'Logs/%s_sigma=%s/' % (args.netname, args.sigma) + ' add_crop'
TB_writer = SummaryWriter(log_dir)
args.start_epoch = 0
args.epochs = 300
args.print_freq = 20
best_precision = 1e6
args.lr = 1e-6
model = CSRNet().cuda()
# 加入随机种子
args.seed = time.time()
random.seed(args.seed)
torch.manual_seed(int(args.seed))
cudnn.deterministic = True

criterion = nn.MSELoss(reduction='sum').cuda()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.95, weight_decay=5 * 1e-4)

# load data
root = 'Dataset'
train_set = os.path.join(root, 'train', 'image')
test_set = os.path.join(root, 'test', 'image')

all_train_img_paths = []
for img_path in glob.glob(os.path.join(train_set, '*.jpg')):
    all_train_img_paths.append(img_path)

all_test_img_paths = []
for img_path in glob.glob(os.path.join(test_set, '*.jpg')):
    all_test_img_paths.append(img_path)

# initialize dataset and dataloader
train_ds = CustomDataset2(dir_list=all_train_img_paths, is_train=True, transform=transforms.Compose([
    transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])]))
train_loader = torch.utils.data.DataLoader(dataset=train_ds, batch_size=4, shuffle=True, num_workers=0)

val_ds = CustomDataset2(dir_list=all_test_img_paths, is_train=False, transform=transforms.Compose([
    transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])]))
val_loader = torch.utils.data.DataLoader(dataset=val_ds, batch_size=1, shuffle=False)

if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_precision = checkpoint['best_precision']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

for epoch in range(args.start_epoch, args.epochs):
    train(model, train_loader, criterion, optimizer, args.print_freq)
    precision = validate(model, val_loader)
    # remember best precision and save checkpoint
    is_best = precision < best_precision
    best_precision = min(precision, best_precision)
    save_checkpoint({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'best_precision': best_precision,
        'optimizer_state_dict': optimizer.state_dict()
    }, is_best)
    adjust_learning_rate(optimizer, epoch, args)
