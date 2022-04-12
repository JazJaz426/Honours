# outputs = net.noisy_posterior(inputs) # 128x10 probability
# loss = criterion(torch.log(outputs), targets) # apply log as NLL only takes logsoftmax output
# criterion = nn.NLLLoss()
'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

from copy import deepcopy
import numpy as np
import os
import argparse

from models import *
from utils import progress_bar
from data.cifar import CIFAR10

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--n_epoch', type=int, default=100)
parser.add_argument('--seed', type = int, default=5)
parser.add_argument('--noise_type', type = str, help='clean_label, aggre_label, worse_label, random_label1, random_label2, random_label3', default='clean_label')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
# parser.add_argument('--model_path', type=str, help='/content/drive/MyDrive/retrain/checkpoint/best_robust_classifier_T1_on_1_run_1.pth')

args = parser.parse_args()

# set seed for all
torch.manual_seed(args.seed)
np.random.seed(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_val_acc = 0  # best validation accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

clean_net_T1_1_path = '/content/drive/MyDrive/final_retrain_21_mar/checkpoint/T1_on_1_seed_5.pth'
clean_net_T1_2_path = '/content/drive/MyDrive/final_retrain_21_mar/checkpoint/T1_on_2_seed_5.pth'
clean_net_T1_3_path = '/content/drive/MyDrive/final_retrain_21_mar/checkpoint/T1_on_3_seed_5.pth'

# Prepare Data
print('==> Preparing data..')

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_set = CIFAR10(
              root='~/data/',
              download=True,  
              train=True, 
              transform = transform_train,
              noise_type = args.noise_type
            )
val_set = CIFAR10(
              root='~/data/',
              download=True,  
              train=True, 
              transform = transform_val,
              noise_type = args.noise_type
            )

num_train = len(train_set)
indices = list(range(num_train))
np.random.seed(args.seed)
np.random.shuffle(indices)
split = int(np.floor(0.1 * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=128, shuffle=False, sampler=train_sampler, num_workers=2)

val_loader = torch.utils.data.DataLoader(
    val_set, batch_size=128, shuffle=False, sampler=valid_sampler, num_workers=2)

test_set = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)

test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


def test(clean_net_T1_1_path, clean_net_T1_2_path, clean_net_T1_3_path, test_loader): 
    # initialise the model and its parameters
    clean_net_T1_1 = ResNet34() 
    clean_net_T1_2 = ResNet34() 
    clean_net_T1_3 = ResNet34() 

    clean_net_T1_1.to(device)
    clean_net_T1_2.to(device)
    clean_net_T1_3.to(device)

    if device == 'cuda':
      clean_net_T1_1 = torch.nn.DataParallel(clean_net_T1_1)
      clean_net_T1_2 = torch.nn.DataParallel(clean_net_T1_2)
      clean_net_T1_3 = torch.nn.DataParallel(clean_net_T1_3)
  
    state_dict_T1_1 = torch.load(clean_net_T1_1_path)['net']
    state_dict_T1_2 = torch.load(clean_net_T1_2_path)['net']
    state_dict_T1_3 = torch.load(clean_net_T1_3_path)['net']

    # load the best trained parameters 
    clean_net_T1_1.load_state_dict(state_dict_T1_1) 
    clean_net_T1_2.load_state_dict(state_dict_T1_2) 
    clean_net_T1_3.load_state_dict(state_dict_T1_3) 
    clean_net_T1_1.eval()
    clean_net_T1_2.eval()
    clean_net_T1_3.eval()

    acc = 0 
    total = 0 
 
    all_different = 0
    all_different_correct = 0

    all_same = 0
    all_same_correct = 0

    T11_T12_same_T13_diff = 0
    T11_T12_same_T13_diff_same_correct = 0
    T11_T12_same_T13_diff_diff_correct = 0

    T11_T13_same_T12_diff = 0
    T11_T13_same_T12_diff_same_correct = 0
    T11_T13_same_T12_diff_diff_correct = 0

    T12_T13_same_T11_diff = 0
    T12_T13_same_T11_diff_same_correct = 0
    T12_T13_same_T11_diff_diff_correct = 0

    with torch.no_grad(): 
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs_T1_1 = clean_net_T1_1(inputs) #clean logit
            _, pred_T1_1 = outputs_T1_1.max(1)

            outputs_T1_2 = clean_net_T1_2(inputs) #clean logit
            _, pred_T1_2 = outputs_T1_2.max(1)

            outputs_T1_3 = clean_net_T1_3(inputs) #clean logit
            _, pred_T1_3 = outputs_T1_3.max(1)

            for i in range(len(targets)):
              # all different
              if (int(pred_T1_1[i]) != int(pred_T1_2[i])) and (int(pred_T1_1[i]) != int(pred_T1_3[i])) and (int(pred_T1_2[i]) != int(pred_T1_3[i])):
                all_different += 1
                # one correct
                if (int(pred_T1_1[i]) == int(targets[i])) or (int(pred_T1_1[i]) == int(targets[i])) or (int(pred_T1_2[i]) == int(targets[i])):
                  all_different_correct += 1

              # all the same
              elif (int(pred_T1_1[i]) == int(pred_T1_2[i]) == int(pred_T1_3[i])):
                all_same += 1
                if int(pred_T1_1[i]) == int(targets[i]):
                  all_same_correct += 1

              # T11_T12_same_T13_diff
              elif (int(pred_T1_1[i]) == int(pred_T1_2[i])) and (int(pred_T1_1[i]) != int(pred_T1_3[i])):
                T11_T12_same_T13_diff += 1
                if (int(pred_T1_1[i]) == int(targets[i])):
                  T11_T12_same_T13_diff_same_correct += 1
                if (int(pred_T1_3[i]) == int(targets[i])):
                  T11_T12_same_T13_diff_diff_correct += 1
              # T11_T13_same_T12_diff
              elif (int(pred_T1_1[i]) == int(pred_T1_3[i])) and (int(pred_T1_1[i]) != int(pred_T1_2[i])):
                T11_T13_same_T12_diff += 1
                if (int(pred_T1_1[i]) == int(targets[i])):
                  T11_T13_same_T12_diff_same_correct += 1
                if (int(pred_T1_2[i]) == int(targets[i])):
                  T11_T13_same_T12_diff_diff_correct += 1

              # T12_T13_same_T11_diff
              else:
                T12_T13_same_T11_diff += 1
                if (int(pred_T1_2[i]) == int(targets[i])):
                  T12_T13_same_T11_diff_same_correct += 1
                if (int(pred_T1_1[i]) == int(targets[i])):
                  T12_T13_same_T11_diff_diff_correct += 1

    print(f'Size of test samples: 10000\n')
    print(f'Size of different prediction: {round(all_different*100/10000, 2)}\n')

    print(f'% of 1 correct prediction: {round(all_different_correct*100 / all_different, 2)}\n')

    print(f'Size of same prediction: {round(all_same*100 / 10000, 2)}\n')
    print(f'% of 3 correct prediction: {round(all_same_correct*100 / all_same, 2)}\n')

    print(f'Size of T11_T12_same_T13_diff: {round(T11_T12_same_T13_diff*100 / 10000, 2)}\n')
    print(f'% of 1 correct prediction (from diff): {round(T11_T12_same_T13_diff_diff_correct*100 / T11_T12_same_T13_diff, 2)}\n')
    print(f'% of 2 correct prediction (from same): {round(T11_T12_same_T13_diff_same_correct*100 / T11_T12_same_T13_diff, 2)}\n')

    print(f'Size of T11_T13_same_T12_diff: {round(T11_T13_same_T12_diff*100 / 10000, 2)}\n')
    print(f'% of 1 correct prediction (from diff): {round(T11_T13_same_T12_diff_diff_correct*100 / T11_T13_same_T12_diff, 2)}\n')
    print(f'% of 2 correct prediction (from same): {round(T11_T13_same_T12_diff_same_correct*100 / T11_T13_same_T12_diff, 2)}\n')

    print(f'Size of T12_T13_same_T11_diff: {round(T12_T13_same_T11_diff*100 / 10000, 2)}\n')
    print(f'% of 1 correct prediction (from diff): {round(T12_T13_same_T11_diff_diff_correct*100 / T12_T13_same_T11_diff, 2)}\n')
    print(f'% of 2 correct prediction (from same): {round(T12_T13_same_T11_diff_same_correct*100 / T12_T13_same_T11_diff, 2)}\n')
    

# Testing
test(clean_net_T1_1_path, clean_net_T1_2_path, clean_net_T1_3_path, test_loader)
