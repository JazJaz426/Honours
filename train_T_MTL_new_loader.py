# outputs = net.noisy_posterior(inputs) # 128x10 probability
# loss = criterion(torch.log(outputs), targets) # apply log as NLL only takes logsoftmax output
# criterion = nn.NLLLoss()
'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from itertools import cycle

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
from data.cifar_3labels import CIFAR10_3labels

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--weight_decay', type = float, help = 'weight', default=5e-4)
parser.add_argument('--viz_path', type = str, help = 'tensorboard', default='Train_T1')
# parser.add_argument('--MTL_T1_save_path', type = str, help = 'save T', default='MTL_T1')
# parser.add_argument('--MTL_T2_save_path', type = str, help = 'save T', default='MTL_T2')
# parser.add_argument('--MTL_T3_save_path', type = str, help = 'save T', default='MTL_T3')
parser.add_argument('--n_epoch', type=int, default=200)
parser.add_argument('--seed', type = int, default=5)
# parser.add_argument('--noise_type', type = str, help='clean_label, aggre_label, worse_label, random_label1, random_label2, random_label3', default='clean_label')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

# set seed for all
torch.manual_seed(args.seed)
np.random.seed(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)
torch.cuda.manual_seed(args.seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_val_acc = 0  # best validation accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
MTL_T1_save_path = './checkpoint/MTL_T1.pth'
MTL_T2_save_path = './checkpoint/MTL_T2.pth'
MTL_T3_save_path = './checkpoint/MTL_T3.pth'
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


train_set = CIFAR10_3labels(
              root='~/data/',
              download=True,  
              train=True, 
              transform = transform_train,
              noise_type_1 = 'random_label1',
              noise_type_2 = 'random_label2',
              noise_type_3 = 'random_label3'
            )
val_set = CIFAR10_3labels(
              root='~/data/',
              download=True,  
              train=True, 
              transform = transform_val,
              noise_type_1 = 'random_label1',
              noise_type_2 = 'random_label2',
              noise_type_3 = 'random_label3'
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

def UpdateCov(weight_matrix, tensor1, tensor2):
    size0 = weight_matrix.size(0)
    final_result = torch.mm(weight_matrix.view(size0, -1), torch.t(torch.matmul(tensor1, torch.matmul(weight_matrix, torch.t(tensor2))).view(size0, -1)))
    return final_result + 0.00001 * torch.eye(final_result.size(0)).cuda()

def MultiTaskLoss(weight_tensor, task_cov_tensor, class_cov_tensor, feature_cov_tensor):
    # weight_matrix: three linear_t weight together
    size_dim0 = weight_tensor.size(0)
    size_dim1 = weight_tensor.size(1)
    size_dim2 = weight_tensor.size(2)
    middle_result1 = torch.matmul(weight_tensor, torch.t(feature_cov_tensor)) # feature_cov_var
    middle_result2 = torch.matmul(class_cov_tensor, middle_result1) # class_cov_var
    final_result = torch.matmul(task_cov_tensor, middle_result2.permute(1,0,2)).permute(1,0,2).contiguous() # feature_cov_var
    return torch.mm(weight_tensor.view(1, -1), final_result.view(-1, 1)).view(1)

def select_func(x):
    if x > 0.1:
        return 1. / x
    else:
        return x

def update_cov():
  # end of each epoch, update covariance
    # get updated weights
    weight_shape = net_1.module.linear_t.weight.size() # 100x512  
    weight_list = [net_1.module.linear_t.weight.view(1, weight_shape[0], weight_shape[1]), net_2.module.linear_t.weight.view(1, weight_shape[0], weight_shape[1]), net_3.module.linear_t.weight.view(1, weight_shape[0], weight_shape[1])] # a list containing three tasks' weights
    weight_tensor = torch.cat(weight_list, dim=0).contiguous() # 3x100x512, a 3d weight tensor   
    temp_task_cov_var = UpdateCov(weight_tensor.data, class_cov_var.data, feature_cov_var.data)

    # update task covariance
    u, s, v = torch.svd(temp_task_cov_var)
    s = s.cpu().apply_(select_func).cuda()
    task_cov_var = torch.mm(u, torch.mm(torch.diag(s), torch.t(v)))
    this_trace = torch.trace(task_cov_var)
    if this_trace > 3000.0:        
        task_cov_var = Variable(task_cov_var / this_trace * 3000.0).cuda()
    else:
        task_cov_var = Variable(task_cov_var).cuda()

# Training
def train(epoch, task_cov_var, class_cov_var, feature_cov_var):
    print('\nEpoch: %d' % epoch)
    networks = [net_1, net_2, net_3]
    net_1.train()
    net_2.train()
    net_3.train()
    total_loss = 0
    classification_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        targets_1, targets_2, targets_3 = targets
        inputs, targets_1, targets_2, targets_3  = inputs.to(device), targets_1.to(device), targets_2.to(device), targets_3.to(device)

        optimizer.zero_grad()
        # calculate classification loss
        outputs_1 = net_1.module.noisy_posterior(inputs) #noisy posterior
        outputs_2 = net_2.module.noisy_posterior(inputs)
        outputs_3 = net_3.module.noisy_posterior(inputs)

        loss_1 = criterion(torch.log(outputs_1), targets_1)
        loss_2 = criterion(torch.log(outputs_2), targets_2)
        loss_3 = criterion(torch.log(outputs_3), targets_3)

        classification_loss = loss_1.item() + loss_2.item() + loss_3.item()

        # calculate multitask loss               
        weight_shape = networks[0].module.linear_t.weight.size() # 100x512  
        weight_list = [networks[i].module.linear_t.weight.view(1, weight_shape[0], weight_shape[1]) for i in range(3)] # a list containing three tasks' weights
        weight_tensor = torch.cat(weight_list, dim=0).contiguous() # 3x100x512, a 3d weight tensor   

        multi_task_loss = MultiTaskLoss(weight_tensor, task_cov_var, class_cov_var, feature_cov_var)
        total_loss = classification_loss + trade_off * multi_task_loss
        # update network parameters
        total_loss.backward()
        optimizer.step()

        _, predicted_1 = outputs_1.max(1)
        _, predicted_2 = outputs_2.max(1)
        _, predicted_3 = outputs_3.max(1)
        total += targets_1.size(0)
        total += targets_2.size(0)
        total += targets_3.size(0)
        correct += predicted_1.eq(targets_1).sum().item()
        correct += predicted_2.eq(targets_2).sum().item()
        correct += predicted_3.eq(targets_3).sum().item()

        update_cov()
        
        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (total_loss/(batch_idx+1), 100.*correct/total, correct, total))

        writer.add_scalar("Avg Total Loss", total_loss/(batch_idx+1), epoch)
        writer.add_scalar("Avg Classification Acc", 100.*correct/total, epoch)

def validation(epoch):
    global best_val_acc
    net_1.eval()
    net_2.eval()
    net_3.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):

            targets_1, targets_2, targets_3 = targets
            inputs, targets_1, targets_2, targets_3  = inputs.to(device), targets_1.to(device), targets_2.to(device), targets_3.to(device)

            outputs_1 = net_1.module.noisy_posterior(inputs)
            outputs_2 = net_2.module.noisy_posterior(inputs)
            outputs_3 = net_3.module.noisy_posterior(inputs)

            loss_1 = criterion(torch.log(outputs_1), targets_1)
            loss_2 = criterion(torch.log(outputs_2), targets_2)
            loss_3 = criterion(torch.log(outputs_3), targets_3)
            
            classification_loss = loss_1.item() + loss_2.item() + loss_3.item()

            val_loss += classification_loss

            _, predicted_1 = outputs_1.max(1)
            _, predicted_2 = outputs_2.max(1)
            _, predicted_3 = outputs_3.max(1)
            total += targets_1.size(0)
            total += targets_2.size(0)
            total += targets_3.size(0)
            correct += predicted_1.eq(targets_1).sum().item()
            correct += predicted_2.eq(targets_2).sum().item()
            correct += predicted_3.eq(targets_3).sum().item()

            progress_bar(batch_idx, len(val_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (val_loss/(batch_idx+1), 100.*correct/total, correct, total))
            
            writer.add_scalar("Validation Loss", val_loss/(batch_idx+1), epoch)
            writer.add_scalar("Validation Acc", 100.*correct/total, epoch)

    # Save checkpoint.
    val_acc = 100.*correct/total
    if val_acc > best_val_acc:
        print('Saving..')
        state_1 = {
            'net': deepcopy(net_1.state_dict()), # deepcopy instead of a reference 
            'acc': val_acc,
            'epoch': epoch,
            'optimizer': deepcopy(optimizer.state_dict()),
            'scheduler': deepcopy(scheduler.state_dict())
        }
        state_2 = {
            'net': deepcopy(net_2.state_dict()), # deepcopy instead of a reference 
            'acc': val_acc,
            'epoch': epoch,
            'optimizer': deepcopy(optimizer.state_dict()),
            'scheduler': deepcopy(scheduler.state_dict())
        }
        state_3 = {
            'net': deepcopy(net_3.state_dict()), # deepcopy instead of a reference 
            'acc': val_acc,
            'epoch': epoch,
            'optimizer': deepcopy(optimizer.state_dict()),
            'scheduler': deepcopy(scheduler.state_dict())
        }
        # save the custom defined model state
        torch.save(state_1, MTL_T1_save_path) 
        torch.save(state_2, MTL_T2_save_path) 
        torch.save(state_3, MTL_T3_save_path) 
        best_val_acc = val_acc

def test_per_epoch(test_loader, model_1, model_2, model_3):
    model_1.eval() 
    model_2.eval() 
    model_3.eval() 

    acc = 0 
    total = 0 

    with torch.no_grad(): 
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs_1 = model_1.module.clean_logit(inputs) #clean logit
            _, predicted_1 = outputs_1.max(1)

            outputs_2 = model_2.module.clean_logit(inputs) #clean logit
            _, predicted_2 = outputs_2.max(1)

            outputs_3 = model_3.module.clean_logit(inputs) #clean logit
            _, predicted_3 = outputs_3.max(1)

            total += targets.size(0)
            total += targets.size(0)
            total += targets.size(0)
            acc += predicted_1.eq(targets).sum().item()
            acc += predicted_2.eq(targets).sum().item()
            acc += predicted_3.eq(targets).sum().item()

        print(f'Testing Accuracy : {round(float(acc)*100/total, 2)}%')

def test_final(model_save_path, test_loader): 
    # initialise the model and its parameters
    net = ResNet34() 
    net.to(device)
    # first put into DP then load state dict https://blog.csdn.net/qxqxqzzz/article/details/106999098
    if device == 'cuda':
      net = torch.nn.DataParallel(net)
      cudnn.benchmark = True
    
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
net_1 = ResNet34()
net_2 = ResNet34()
net_3 = ResNet34()

# put the model to GPU if available
net_1 = net_1.to(device)
net_2 = net_2.to(device)
net_3 = net_3.to(device)
if device == 'cuda':
  net_1 = torch.nn.DataParallel(net_1)
  net_2 = torch.nn.DataParallel(net_2)
  net_3 = torch.nn.DataParallel(net_3)
  cudnn.benchmark = True

# load the trained parameters of clean classifier into the new model
# and freeze all layers except linear_t
clean_path = '/content/drive/MyDrive/final_retrain_21_mar/checkpoint/best_clean_classifier_seed_5.pth'
load_parameters_and_freeze(net_1, clean_path)
load_parameters_and_freeze(net_2, clean_path)
load_parameters_and_freeze(net_3, clean_path)

criterion = nn.NLLLoss() #compare logsoftmax probability and target
# only update grad for linear_t
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, list(net_1.parameters()) + list(net_2.parameters()) + list(net_3.parameters())), 
    lr=args.lr, 
    weight_decay=5e-4
)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)

# initialise covariance
trade_off = 1
num_tasks = 3
output_num = 100
bottleneck_size = 512

# initialize covariance matrix
task_cov = torch.eye(num_tasks)
class_cov = torch.eye(output_num)
feature_cov = torch.eye(bottleneck_size)

task_cov_var = Variable(task_cov).cuda()
class_cov_var = Variable(class_cov).cuda()
feature_cov_var = Variable(feature_cov).cuda()

# training and validation
for epoch in range(start_epoch, start_epoch + args.n_epoch):
    train(epoch, task_cov_var, class_cov_var, feature_cov_var)
    validation(epoch)
    test_per_epoch(test_loader, net_1, net_2, net_3)
    scheduler.step()

# # Testing
# test_final(MTL_T1_save_path, test_loader)
# test_final(MTL_T2_save_path, test_loader)
# test_final(MTL_T3_save_path, test_loader)
