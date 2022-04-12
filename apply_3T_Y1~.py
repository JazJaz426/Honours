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
from torch.utils.tensorboard import SummaryWriter

from copy import deepcopy
import numpy as np
import os
import argparse

from models import *
from utils import progress_bar
from data.cifar import CIFAR10

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--weight_decay', type = float, help = 'weight', default=5e-4)
parser.add_argument('--tensorboard_name', type = str, help = 'tensorboard', default='apply_T1_on_1_run_1')
parser.add_argument('--n_epoch', type=int, default=200)
parser.add_argument('--noise_type', type = str, help='clean_label, aggre_label, worse_label, random_label1, random_label2, random_label3', default='clean_label')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--seed', type = int, default=5)
args = parser.parse_args()

# set seed for all
torch.manual_seed(args.seed)
np.random.seed(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_val_acc = 0  # best validation accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
model_save_path = './checkpoint/robust_classifier_3T_Y1~_seed_5.pth'
T1_path = './checkpoint/T1_seed_5.pth'
T2_path = './checkpoint/T2_seed_5.pth'
T3_path = './checkpoint/T3_seed_5.pth'

writer = SummaryWriter('runs/' + args.tensorboard_name)

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

# sample proportional to the index length
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

def transition_matrix_for_loader(loader, model, num_sample, num_classes, batch_size):
    model.eval()
    T_all = torch.rand(num_sample, num_classes, num_classes)
    ind = int(num_sample / batch_size)
    with torch.no_grad():
        # batch by batch fill in T_all
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs = inputs.to(device)
            # 128x10x10
            T_batch = model.module.transition_matrix(inputs)
            if batch_idx < ind:
              T_all[batch_idx*batch_size:(batch_idx+1)*batch_size, :, :] = T_batch
            else:
              T_all[ind*batch_size:, :, :] = T_batch
            
    return T_all

# Training
def train(epoch, T1_train_all, T2_train_all, T3_train_all):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    num_sample = len(train_set)
    batch_size = 128
    ind = int(num_sample / batch_size)

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets  = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        clean_logit = net.module.clean_logit(inputs) #clean_logit 128x10
        clean_posterior = torch.nn.Softmax(dim=1)(clean_logit) # clean posterior 128x10
        clean_posterior_reshaped = clean_posterior.view(-1, 10, 1) # 128x10x1
        
        if batch_idx < ind:
            T1_batch = T1_train_all[batch_idx*batch_size:(batch_idx+1)*batch_size, :, :]
            T2_batch = T2_train_all[batch_idx*batch_size:(batch_idx+1)*batch_size, :, :]
            T3_batch = T3_train_all[batch_idx*batch_size:(batch_idx+1)*batch_size, :, :]
        else:
            T1_batch = T1_train_all[ind*batch_size:, :, :]
            T2_batch = T2_train_all[ind*batch_size:, :, :]
            T3_batch = T3_train_all[ind*batch_size:, :, :]

        # convert ndarray to tensor
        # T_batch = torch.from_numpy(T_batch).float().cuda()
        T1_batch = T1_batch.to(device)
        T2_batch = T2_batch.to(device)
        T3_batch = T3_batch.to(device)
    
        noisy_posterior_1 = torch.matmul(T1_batch, clean_posterior_reshaped) # noisy posterior 128x10x1
        noisy_posterior_reshaped_1 = noisy_posterior_1.view(noisy_posterior_1.size(0), -1) # 128x10

        noisy_posterior_2 = torch.matmul(T2_batch, clean_posterior_reshaped) # noisy posterior 128x10x1
        noisy_posterior_reshaped_2 = noisy_posterior_2.view(noisy_posterior_2.size(0), -1) # 128x10

        noisy_posterior_3 = torch.matmul(T3_batch, clean_posterior_reshaped) # noisy posterior 128x10x1
        noisy_posterior_reshaped_3 = noisy_posterior_3.view(noisy_posterior_3.size(0), -1) # 128x10

        loss_1 = criterion(torch.log(noisy_posterior_reshaped_1), targets)
        loss_2 = criterion(torch.log(noisy_posterior_reshaped_2), targets)
        loss_3 = criterion(torch.log(noisy_posterior_reshaped_3), targets)

        loss_1 += loss_2
        loss_1 += loss_3

        train_loss += loss_1.item() 
        train_loss += loss_2.item() 
        train_loss += loss_3.item() 

        loss_1.backward()
        optimizer.step()

        _, predicted_1 = noisy_posterior_reshaped_1.max(1)
        _, predicted_2 = noisy_posterior_reshaped_2.max(1)
        _, predicted_3 = noisy_posterior_reshaped_3.max(1)

        total += 3*targets.size(0)

        correct += predicted_1.eq(targets).sum().item()
        correct += predicted_2.eq(targets).sum().item()
        correct += predicted_3.eq(targets).sum().item()

        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

        writer.add_scalar("Train Loss", train_loss/(batch_idx+1), epoch)
        writer.add_scalar("Train Acc", 100.*correct/total, epoch)

def validation(epoch, T1_val_all, T2_val_all, T3_val_all):
    global best_val_acc
    net.eval()
    val_loss = 0
    correct = 0
    total = 0

    num_sample = len(val_set)
    batch_size = 128
    ind = int(num_sample / batch_size)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            clean_logit = net.module.clean_logit(inputs)
            clean_posterior = torch.nn.Softmax(dim=1)(clean_logit) # clean posterior 128x10
            clean_posterior_reshaped = clean_posterior.view(-1, 10, 1) # 128x10x1

            if batch_idx < ind:
                T1_batch = T1_val_all[batch_idx*batch_size:(batch_idx+1)*batch_size, :, :]
                T2_batch = T2_val_all[batch_idx*batch_size:(batch_idx+1)*batch_size, :, :]
                T3_batch = T3_val_all[batch_idx*batch_size:(batch_idx+1)*batch_size, :, :]
            else:
                T1_batch = T1_val_all[ind*batch_size:, :, :]
                T2_batch = T2_val_all[ind*batch_size:, :, :]
                T3_batch = T3_val_all[ind*batch_size:, :, :]

            # convert ndarray to tensor
            # T_batch = torch.from_numpy(T_batch).float().cuda()
            T1_batch = T1_batch.cuda()
            T2_batch = T2_batch.cuda()
            T3_batch = T3_batch.cuda()
            
            noisy_posterior_1 = torch.matmul(T1_batch, clean_posterior_reshaped) # noisy posterior 128x10x1
            noisy_posterior_reshaped_1 = noisy_posterior_1.view(noisy_posterior_1.size(0), -1) # 128x10

            noisy_posterior_2 = torch.matmul(T2_batch, clean_posterior_reshaped) # noisy posterior 128x10x1
            noisy_posterior_reshaped_2 = noisy_posterior_2.view(noisy_posterior_2.size(0), -1) # 128x10

            noisy_posterior_3 = torch.matmul(T3_batch, clean_posterior_reshaped) # noisy posterior 128x10x1
            noisy_posterior_reshaped_3 = noisy_posterior_3.view(noisy_posterior_3.size(0), -1) # 128x10

            loss_1 = criterion(torch.log(noisy_posterior_reshaped_1), targets)
            loss_2 = criterion(torch.log(noisy_posterior_reshaped_2), targets)
            loss_3 = criterion(torch.log(noisy_posterior_reshaped_3), targets)

            val_loss += loss_1.item()
            val_loss += loss_2.item()
            val_loss += loss_3.item()

            _, predicted_1 = noisy_posterior_reshaped_1.max(1)
            _, predicted_2 = noisy_posterior_reshaped_2.max(1)
            _, predicted_3 = noisy_posterior_reshaped_3.max(1)

            total += 3*targets.size(0)

            correct += predicted_1.eq(targets).sum().item()
            correct += predicted_2.eq(targets).sum().item()
            correct += predicted_3.eq(targets).sum().item()

            progress_bar(batch_idx, len(val_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (val_loss/(batch_idx+1), 100.*correct/total, correct, total))
            
            writer.add_scalar("Validation Loss", val_loss/(batch_idx+1), epoch)
            writer.add_scalar("Validation Acc", 100.*correct/total, epoch)

    # Save checkpoint.
    val_acc = round(100.*correct/total, 3)
    if val_acc > best_val_acc:
        print('Saving..')
        state = {
            'net': deepcopy(net.state_dict()), # deepcopy instead of a reference 
            'acc': val_acc,
            'epoch': epoch,
            'optimizer': deepcopy(optimizer.state_dict()),
            'scheduler': deepcopy(scheduler.state_dict())
        }
        # save the custom defined model state
        torch.save(state, model_save_path) 
        best_val_acc = val_acc

def test_per_epoch(test_loader, model):
    model.eval() 
    acc = 0 
    total = 0 

    with torch.no_grad(): 
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model.module.clean_logit(inputs) #clean logit
            _, predicted = outputs.max(1)
            total += targets.size(0)
            acc += predicted.eq(targets).sum().item()

        print(f'Testing Accuracy : {round(float(acc)*100/total, 2)}%')

def test_final(model_save_path, test_loader): 
    # initialise the model and its parameters
    net = ResNet34() 
    net.to(device)
    # first put into DP then load state dict https://blog.csdn.net/qxqxqzzz/article/details/106999098
    if device == 'cuda':
      net = torch.nn.DataParallel(net)
    
    best_model_state = torch.load(model_save_path)
    best_model_state_dict = best_model_state['net']
    val_acc = best_model_state['acc']

    # load the best trained parameters 
    net.load_state_dict(best_model_state_dict) 
    net.eval()

    acc = 0 
    total = 0 
 
    with torch.no_grad(): 
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net.module.clean_logit(inputs) #clean logit
            _, predicted = outputs.max(1)
            total += targets.size(0)
            acc += predicted.eq(targets).sum().item()

    val_acc = round(val_acc, 2)
    test_acc = round(float(acc)*100/total, 2)
    return val_acc, test_acc

### Actual Main Code ###

# initialize model
print('==> Building model..')

net = ResNet34()

# put the model to GPU if available
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    torch.backends.cudnn.benchmark = True
    
criterion = nn.NLLLoss() 
optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)

# resume training from last breaking point
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    state = torch.load(model_save_path)
    # load the saved states
    net.load_state_dict(state['net']) # state_dict = parameters
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    best_acc = state['acc'] # validation accuracy
    start_epoch = state['epoch'] + 1 # epoch that ended training

# load the noisy net to calculate T for train and val
net_T1 = ResNet34()
net_T2 = ResNet34()
net_T3 = ResNet34()

net_T1 = net_T1.to(device)
net_T2 = net_T2.to(device)
net_T3 = net_T3.to(device)

if device == 'cuda':
    net_T1 = torch.nn.DataParallel(net_T1)
    net_T2 = torch.nn.DataParallel(net_T2)
    net_T3 = torch.nn.DataParallel(net_T3)

T1_state = torch.load(T1_path)
net_T1.load_state_dict(T1_state['net']) 

T2_state = torch.load(T2_path)
net_T2.load_state_dict(T2_state['net']) 

T3_state = torch.load(T3_path)
net_T3.load_state_dict(T3_state['net']) 

# T1
T1_train_all = transition_matrix_for_loader(train_loader, net_T1, len(train_idx), 10, 128)
T1_val_all = transition_matrix_for_loader(val_loader, net_T1, len(valid_idx), 10, 128)

# T2
T2_train_all = transition_matrix_for_loader(train_loader, net_T2, len(train_idx), 10, 128)
T2_val_all = transition_matrix_for_loader(val_loader, net_T2, len(valid_idx), 10, 128)

# T3
T3_train_all = transition_matrix_for_loader(train_loader, net_T3, len(train_idx), 10, 128)
T3_val_all = transition_matrix_for_loader(val_loader, net_T3, len(valid_idx), 10, 128)

# training and validation
for epoch in range(start_epoch, start_epoch + args.n_epoch):
    train(epoch, T1_train_all, T2_train_all, T3_train_all)
    validation(epoch, T1_val_all, T2_val_all, T3_val_all)
    test_per_epoch(test_loader, net)
    scheduler.step()

# Testing
val_acc, test_acc = test_final(model_save_path, test_loader)
print(f'Validation Accuracy : {val_acc}%\n')
print(f'Testing Accuracy : {test_acc}%\n')

result_path = './result/train_3T_Y1~_together_seed_5_parallel.txt' 
with open(result_path, "a") as myfile:
    myfile.write('Validation Accuracy :' + str(val_acc) + "\n")
    myfile.write('Test Accuracy :' + str(test_acc) + "\n")
