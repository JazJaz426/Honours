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
clean_net_T2_2_path = '/content/drive/MyDrive/final_retrain_21_mar/checkpoint/T2_on_2_seed_5.pth'
clean_net_T3_3_path = '/content/drive/MyDrive/final_retrain_21_mar/checkpoint/T3_on_3_seed_5.pth'
clean_net_3T_path = '/content/drive/MyDrive/final_retrain_21_mar/checkpoint/robust_classifier_3T_seed_5.pth'

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


def test(clean_net_T1_1_path, clean_net_T2_2_path, clean_net_T3_3_path, clean_net_3T_path, test_loader): 
    # initialise the model and its parameters
    clean_net_T1_1 = ResNet34() 
    clean_net_T2_2 = ResNet34() 
    clean_net_T3_3 = ResNet34()
    clean_net_3T = ResNet34()  

    clean_net_T1_1.to(device)
    clean_net_T2_2.to(device)
    clean_net_T3_3.to(device)
    clean_net_3T.to(device)

    if device == 'cuda':
      clean_net_T1_1 = torch.nn.DataParallel(clean_net_T1_1)
      clean_net_T2_2 = torch.nn.DataParallel(clean_net_T2_2)
      clean_net_T3_3 = torch.nn.DataParallel(clean_net_T3_3)
      clean_net_3T = torch.nn.DataParallel(clean_net_3T)
  
    state_dict_T1_1 = torch.load(clean_net_T1_1_path)['net']
    state_dict_T2_2 = torch.load(clean_net_T2_2_path)['net']
    state_dict_T3_3 = torch.load(clean_net_T3_3_path)['net']
    state_dict_3T = torch.load(clean_net_3T_path)['net']

    # load the best trained parameters 
    clean_net_T1_1.load_state_dict(state_dict_T1_1) 
    clean_net_T2_2.load_state_dict(state_dict_T2_2) 
    clean_net_T3_3.load_state_dict(state_dict_T3_3) 
    clean_net_3T.load_state_dict(state_dict_3T) 

    clean_net_T1_1.eval()
    clean_net_T2_2.eval()
    clean_net_T3_3.eval()
    clean_net_3T.eval()

    acc = 0 
    total = 0 
 
    all_different = 0
    all_different_correct = 0
    all_different_correct_c = 0
    all_different_all_wrong_c = 0

    all_same = 0
    all_same_correct = 0
    all_same_correct_c = 0
    all_same_all_wrong_c = 0

    T11_T22_same_T33_diff = 0
    T11_T22_same_T33_diff_same_correct = 0
    T11_T22_same_T33_diff_same_correct_c = 0
    T11_T22_same_T33_diff_diff_correct = 0
    T11_T22_same_T33_diff_diff_correct_c = 0
    T11_T22_same_T33_diff_all_wrong_c = 0


    T11_T33_same_T22_diff = 0
    T11_T33_same_T22_diff_same_correct = 0
    T11_T33_same_T22_diff_same_correct_c = 0
    T11_T33_same_T22_diff_diff_correct = 0
    T11_T33_same_T22_diff_diff_correct_c = 0
    T11_T33_same_T22_diff_all_wrong_c = 0

    T22_T33_same_T11_diff = 0
    T22_T33_same_T11_diff_same_correct = 0
    T22_T33_same_T11_diff_same_correct_c = 0
    T22_T33_same_T11_diff_diff_correct = 0
    T22_T33_same_T11_diff_diff_correct_c = 0
    T22_T33_same_T11_diff_all_wrong_c = 0

    with torch.no_grad(): 
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs_T1_1 = clean_net_T1_1(inputs) #clean logit
            _, pred_T1_1 = outputs_T1_1.max(1)

            outputs_T2_2 = clean_net_T2_2(inputs) #clean logit
            _, pred_T2_2 = outputs_T2_2.max(1)

            outputs_T3_3 = clean_net_T3_3(inputs) #clean logit
            _, pred_T3_3 = outputs_T3_3.max(1)

            outputs_3T = clean_net_3T(inputs) #clean logit
            _, pred_3T = outputs_3T.max(1)

            for i in range(len(targets)):
              # 1. all different
              if (int(pred_T1_1[i]) != int(pred_T2_2[i])) and (int(pred_T1_1[i]) != int(pred_T3_3[i])) and (int(pred_T2_2[i]) != int(pred_T3_3[i])):
                all_different += 1
                # one correct
                if (int(pred_T1_1[i]) == int(targets[i])) or (int(pred_T3_3[i]) == int(targets[i])) or (int(pred_T2_2[i]) == int(targets[i])):
                  all_different_correct += 1
                  # the correct one equals 3t's prediction
                  if (int(targets[i]) == int(pred_3T[i])):
                    all_different_correct_c += 1
                # all wrong
                else:
                  if (int(targets[i]) == int(pred_3T[i])):
                    all_different_all_wrong_c += 1

              # 2. all the same
              elif (int(pred_T1_1[i]) == int(pred_T2_2[i]) == int(pred_T3_3[i])):
                all_same += 1
                # same all correct
                if int(pred_T1_1[i]) == int(targets[i]):
                  all_same_correct += 1
                  if int(targets[i]) == int(pred_3T[i]):
                    all_same_correct_c += 1
                # same all wrong
                else:
                  if int(targets[i]) == int(pred_3T[i]):
                    all_same_all_wrong_c += 1

              # 3. T11_T22_same_T33_diff
              elif (int(pred_T1_1[i]) == int(pred_T2_2[i])) and (int(pred_T1_1[i]) != int(pred_T3_3[i])):
                T11_T22_same_T33_diff += 1
                # same correct
                if (int(pred_T1_1[i]) == int(targets[i])):
                  T11_T22_same_T33_diff_same_correct += 1
                  if (int(pred_T1_1[i]) == int(pred_3T[i])):
                    T11_T22_same_T33_diff_same_correct_c += 1
                # different correct
                elif (int(pred_T3_3[i]) == int(targets[i])):
                  T11_T22_same_T33_diff_diff_correct += 1
                  if (int(pred_T3_3[i]) == int(pred_3T[i])):
                    T11_T22_same_T33_diff_diff_correct_c += 1
                # all wrong
                else:
                    if (int(pred_T3_3[i]) == int(pred_3T[i])):
                        T11_T22_same_T33_diff_all_wrong_c += 1

              # T11_T33_same_T22_diff
              elif (int(pred_T1_1[i]) == int(pred_T3_3[i])) and (int(pred_T1_1[i]) != int(pred_T2_2[i])):
                T11_T33_same_T22_diff += 1
                # same correct
                if (int(pred_T1_1[i]) == int(targets[i])):
                  T11_T33_same_T22_diff_same_correct += 1
                  if (int(pred_T1_1[i]) == int(pred_3T[i])):
                    T11_T33_same_T22_diff_same_correct_c += 1
                # diff correct
                elif (int(pred_T2_2[i]) == int(targets[i])):
                  T11_T33_same_T22_diff_diff_correct += 1
                  if (int(pred_T2_2[i]) == int(pred_3T[i])):
                    T11_T33_same_T22_diff_diff_correct_c += 1
                # all wrong
                else:
                    if (int(targets[i]) == int(pred_3T[i])):
                        T11_T33_same_T22_diff_all_wrong_c += 1

              # T12_T13_same_T11_diff
              else:
                T22_T33_same_T11_diff += 1
                # same correct
                if (int(pred_T2_2[i]) == int(targets[i])):
                  T22_T33_same_T11_diff_same_correct += 1
                  if (int(pred_T2_2[i]) == int(pred_3T[i])):
                    T22_T33_same_T11_diff_same_correct_c += 1
                # diff correct
                elif (int(pred_T1_1[i]) == int(targets[i])):
                  T22_T33_same_T11_diff_diff_correct += 1
                  if (int(pred_T1_1[i]) == int(pred_3T[i])):
                    T22_T33_same_T11_diff_diff_correct_c += 1
                # all wrong
                else:
                    if (int(targets[i]) == int(pred_3T[i])):
                        T22_T33_same_T11_diff_all_wrong_c += 1

    print(f'Size of test samples: 10000%\n')
    print(f'1. all same prediction\n')
    print(f'% of same prediction: {round(all_same*100 / 10000, 2)}% \n')
    print(f'% of 3 correct: {round(all_same_correct*100 / 10000, 2)}%\n')
    print(f'% of 3 correct & same as C: {round(all_same_correct_c*100 / 10000, 2)}%\n')
    print(f'% of all wrong: {round((all_same-all_same_correct)*100 / 10000, 2)}%\n')
    print(f'% of all wrong & C correct: {round(all_same_all_wrong_c*100 / 10000, 2)}%\n\n')

    print(f'2. all different prediction\n')
    print(f'% of different prediction: {round(all_different*100/10000, 2)}%\n')
    print(f'% of 1 correct: {round(all_different_correct*100 / 10000, 2)}%\n')
    print(f'% of 1 correct & same as C: {round(all_different_correct_c*100 / 10000, 2)}%\n')
    print(f'% of all wrong: {round((all_different-all_different_correct)*100 / 10000, 2)}%\n')
    print(f'% of all wrong & C correct: {round(all_different_all_wrong_c*100 / 10000, 2)}%\n\n')

    print(f'3. T11_T22_same_T33_diff \n')
    print(f'% of T11_T22_same_T33_diff: {round(T11_T22_same_T33_diff*100 / 10000, 2)}%\n')
    print(f'% of 2 correct (from same): {round(T11_T22_same_T33_diff_same_correct*100 / 10000, 2)}%\n')
    print(f'% of 2 correct & same as C (from same): {round(T11_T22_same_T33_diff_same_correct_c*100 / 10000, 2)}%\n')
    print(f'% of 1 correct (from diff): {round(T11_T22_same_T33_diff_diff_correct*100 / 10000, 2)}%\n')
    print(f'% of 1 correct & same as C (from diff): {round(T11_T22_same_T33_diff_diff_correct_c*100 / 10000, 2)}%\n')
    print(f'% of all wrong: {round((T11_T22_same_T33_diff-T11_T22_same_T33_diff_same_correct-T11_T22_same_T33_diff_diff_correct)*100 / 10000, 2)}%\n')
    print(f'% of all wrong, but C correct: {round(T11_T22_same_T33_diff_all_wrong_c*100 / 10000, 2)}%\n\n')

    print(f'4. T11_T33_same_T22_diff \n')
    print(f'% of T11_T33_same_T22_diff: {round(T11_T33_same_T22_diff*100 / 10000, 2)}%\n')
    print(f'% of 2 correct (from same): {round(T11_T33_same_T22_diff_same_correct*100 / 10000, 2)}%\n')
    print(f'% of 2 correct & same as C (from same): {round(T11_T33_same_T22_diff_same_correct_c*100 / 10000, 2)}%\n')
    print(f'% of 1 correct (from diff): {round(T11_T33_same_T22_diff_diff_correct*100 / 10000, 2)}%\n')
    print(f'% of 1 correct & same as C (from diff): {round(T11_T33_same_T22_diff_diff_correct_c*100 / 10000, 2)}%\n')
    print(f'% of all wrong : {round((T11_T33_same_T22_diff-T11_T33_same_T22_diff_same_correct-T11_T33_same_T22_diff_diff_correct)*100 / 10000, 2)}%\n')
    print(f'% of all wrong & C correct : {round(T11_T33_same_T22_diff_all_wrong_c*100 / 10000, 2)}%\n\n')

    print(f'5. T22_T33_same_T11_diff \n')
    print(f'% of T22_T33_same_T11_diff: {round(T22_T33_same_T11_diff*100 / 10000, 2)}%\n')
    print(f'% of 2 correct (from same): {round(T22_T33_same_T11_diff_same_correct*100 / 10000, 2)}%\n')
    print(f'% of 2 correct & same as C (from same): {round(T22_T33_same_T11_diff_same_correct_c*100 / 10000, 2)}%\n')
    print(f'% of 1 correct (from diff): {round(T22_T33_same_T11_diff_diff_correct*100 / 10000, 2)}%\n')
    print(f'% of 1 correct & same as C (from diff): {round(T22_T33_same_T11_diff_diff_correct_c*100 / 10000, 2)}%\n')
    print(f'% of all wrong : {round((T22_T33_same_T11_diff-T22_T33_same_T11_diff_same_correct-T22_T33_same_T11_diff_diff_correct)*100 / 10000, 2)}%\n')
    print(f'% of all wrong but C correct : {round(T22_T33_same_T11_diff_all_wrong_c*100 / 10000, 2)}%\n\n')

# Testing
test(clean_net_T1_1_path, clean_net_T2_2_path, clean_net_T3_3_path, clean_net_3T_path, test_loader)
