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
parser.add_argument('--viz_path', type = str, help = 'tensorboard', default='Train_T1')
parser.add_argument('--clean_path', type = str, help = 'clean classifier', default='clean_path')
parser.add_argument('--model_save_path', type = str, help = 'save T', default='T1')
parser.add_argument('--n_epoch', type=int, default=200)
parser.add_argument('--seed', type = int, default=5)
parser.add_argument('--noise_type', type = str, help='clean_label, aggre_label, worse_label, random_label1, random_label2, random_label3', default='clean_label')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
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
model_save_path = './checkpoint/' + args.model_save_path + '.pth'
viz_path = args.viz_path

# Initialize the SummaryWriter for TensorBoard
writer = SummaryWriter('runs/' + viz_path)

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

### Helper Functions ###

def load_parameters_and_freeze(net, model_path):
    new_model = net
    # load the trained clean classifer's parameters 
    clean_path = model_path
    clean_state = torch.load(clean_path)
    clean_parameters = clean_state['net']
    new_model.load_state_dict(clean_parameters) 

    # freeze all layers including trained clean layers
    for param in new_model.parameters():
        param.requires_grad = False

    # unfreeze linear_t layers (last layer)
    unfrozen_layer = new_model.module.linear_t
    # weight only - param 100x512 (out_feature x in_feature)
    for param in unfrozen_layer.parameters():
          param.requires_grad = True
    
# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net.module.noisy_posterior(inputs) #noisy posterior
        loss = criterion(torch.log(outputs), targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

        writer.add_scalar("Train Loss", train_loss/(batch_idx+1), epoch)
        writer.add_scalar("Train Acc", 100.*correct/total, epoch)

def validation(epoch):
    global best_val_acc
    net.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net.module.noisy_posterior(inputs)
            loss = criterion(torch.log(outputs), targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(val_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (val_loss/(batch_idx+1), 100.*correct/total, correct, total))
            
            writer.add_scalar("Validation Loss", val_loss/(batch_idx+1), epoch)
            writer.add_scalar("Validation Acc", 100.*correct/total, epoch)


    # Save checkpoint.
    val_acc = 100.*correct/total
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
    # load the best trained parameters 
    net.load_state_dict(best_model_state_dict) 
    net.eval()

    acc = 0 
    total = 0 
 
    with torch.no_grad(): 
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net.module.clean_logit(inputs) #clean posterior
            _, predicted = outputs.max(1)
            total += targets.size(0)
            acc += predicted.eq(targets).sum().item()

        print(f'Testing Accuracy : {round(float(acc)*100/total, 2)}%')

### Actual Main Code ###

# initialize model
print('==> Building model..')
net = ResNet34()

# put the model to GPU if available
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)

# load the trained parameters of clean classifier into the new model
# and freeze all layers except linear_t
clean_path = args.clean_path
load_parameters_and_freeze(net, clean_path)

criterion = nn.NLLLoss() #compare logsoftmax probability and target
# only update grad for linear_t
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, net.parameters()), 
    lr=args.lr, 
    weight_decay=5e-4
)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)

# training and validation
for epoch in range(start_epoch, start_epoch + args.n_epoch):
    train(epoch)
    validation(epoch)
    test_per_epoch(test_loader, net)
    scheduler.step()

# Testing
test_final(model_save_path, test_loader)
