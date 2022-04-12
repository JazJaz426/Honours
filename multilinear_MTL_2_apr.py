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

from models.resnet_without_t import *
from models.resnet import *
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
MTL_clean_net_save_path = './checkpoint/MTL_robust_classifier.pth'
clean_path = '/content/drive/MyDrive/final_retrain_21_mar/checkpoint/best_clean_classifier_seed_5.pth'

T1_path = '/content/drive/MyDrive/final_retrain_21_mar/checkpoint/T1_seed_5.pth'
T2_path = '/content/drive/MyDrive/final_retrain_21_mar/checkpoint/T2_seed_5.pth'
T3_path = '/content/drive/MyDrive/final_retrain_21_mar/checkpoint/T3_seed_5.pth'

MTL_T1_save_path = '/content/drive/MyDrive/final_retrain_21_mar/checkpoint/MTL_T1.pth'
MTL_T2_save_path = '/content/drive/MyDrive/final_retrain_21_mar/checkpoint/MTL_T2.pth'
MTL_T3_save_path = '/content/drive/MyDrive/final_retrain_21_mar/checkpoint/MTL_T3.pth'

T_paths = [T1_path, T2_path, T3_path]

viz_path = args.viz_path
num_tasks = 3

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

def load_T_to_layer(layers_specific, T_paths):
    for i in range(3):
      net_T = ResNet34()

      net_T = net_T.to(device)
      if device == 'cuda':
        net_T = torch.nn.DataParallel(net_T)
      T_state = torch.load(T_paths[i])
      net_T_parameters = T_state['net']
      net_T.load_state_dict(net_T_parameters)
      layers_specific[i] = nn.Sequential(net_T.module.linear_t)
      layers_specific[i] = torch.nn.DataParallel(layers_specific[i])

def load_C(net_shared, clean_path):
    for i in range(3):
      net_C = ResNet34()
      net_C = net_C.to(device)
      if device == 'cuda':
        net_C = torch.nn.DataParallel(net_C)

      net_C_state = torch.load(clean_path)
      net_C_parameters = net_C_state['net']
      net_C.load_state_dict(net_C_parameters)

      source_dict = net_C.state_dict()
      target_dict = net_shared.state_dict()

      # 1. filter out unnecessary keys
      source_dict = {k: v for k, v in source_dict.items() if k in target_dict}
      # 2. overwrite entries in the existing state dict
      target_dict.update(source_dict) 
      # 3. load the new state dict
      net_shared.load_state_dict(target_dict)

def calculate_noisy_posterior(net_clean, layer_t, inputs):
    extracted_features = net_clean.module.feature_extractor(inputs)
    clean_logit = net_clean.module.clean_logit(inputs) # clean logit 128x10
    clean_posterior = torch.nn.Softmax(dim=1)(clean_logit) # clean posterior 128x10
    clean_posterior_reshaped = clean_posterior.view(-1, 10, 1) # 128x10x1

    transition_vector = layer_t(extracted_features) # 128x10x10
    transition_matrix_logit = transition_vector.view(-1, 10, 10) # reshape the transition vector from 128x100 to 128x10x10
    transition_matrix = torch.nn.Softmax(dim=1)(transition_matrix_logit) # 128x10x10
    out = torch.matmul(transition_matrix, clean_posterior_reshaped) # noisy posterior 128x10x1
    out = out.view(out.size(0), -1) # 128x10

    return out

def UpdateCov(weight_matrix, tensor1, tensor2):
    size0 = weight_matrix.size(0)
    final_result = torch.mm(weight_matrix.view(size0, -1), torch.t(torch.matmul(tensor1, torch.matmul(weight_matrix, torch.t(tensor2))).view(size0, -1)))
    final_result = final_result + 0.00001 * torch.eye(final_result.size(0)).cuda()
    return final_result

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

def update_cov(task_cov_var, class_cov_var, feature_cov_var):
  # every 100 iterations, update covariance
    # get updated weights
    weight_shape = layers_specific[0].module[-1].weight.size() # 100x512  
    weight_list = [layers_specific[i].module[-1].weight.view(1, weight_shape[0], weight_shape[1]) for i in range(num_tasks)] # a list containing three tasks' weights
    weight_tensor = torch.cat(weight_list, dim=0).contiguous() # 3x100x512, a 3d weight tensor   
    # update covariances
    temp_task_cov_var = UpdateCov(weight_tensor.data, class_cov_var.data, feature_cov_var.data)
    # temp_class_cov_var = UpdateCov(weight_tensor.data.permute(1, 0, 2).contiguous(), task_cov_var.data, feature_cov_var.data)
    # temp_feature_cov_var = UpdateCov(weight_tensor.data.permute(2, 0, 1).contiguous(), task_cov_var.data, class_cov_var.data)

    # task covariance
    u, s, v = torch.svd(temp_task_cov_var)
    s = s.cpu().apply_(select_func).cuda()
    task_cov_var = torch.mm(u, torch.mm(torch.diag(s), torch.t(v)))
    
    this_trace = torch.trace(task_cov_var)
    if this_trace > 3000.0:        
        task_cov_var = Variable(task_cov_var / this_trace * 3000.0).cuda()
    else:
        task_cov_var = Variable(task_cov_var).cuda()

    return task_cov_var
    # # class covariance
    # u, s, v = torch.svd(temp_class_cov_var)
    # s = s.cpu().apply_(select_func).cuda()
    # class_cov_var = torch.mm(u, torch.mm(torch.diag(s), torch.t(v)))
    # this_trace = torch.trace(class_cov_var)
    # if this_trace > 3000.0:        
    #     class_cov_var = Variable(class_cov_var / this_trace * 3000.0).cuda()
    # else:
    #     class_cov_var = Variable(class_cov_var).cuda()

    # feature covariance
    # u, s, v = torch.svd(temp_feature_cov_var)
    # s = s.cpu().apply_(select_func).cuda()
    # temp_feature_cov_var = torch.mm(u, torch.mm(torch.diag(s), torch.t(v)))
    # this_trace = torch.trace(temp_feature_cov_var)

    # if this_trace > 3000.0:        
    #     feature_cov_var += 0.0003 * Variable(temp_feature_cov_var / this_trace * 3000.0).cuda()
    # else:
    #     feature_cov_var += 0.0003 * Variable(temp_feature_cov_var).cuda()

# Training
def train(epoch, task_cov_var, class_cov_var, feature_cov_var):
    print('\nEpoch: %d' % epoch)

    # reinitialize every epoch
    train_loss = 0
    total_multi_class_loss = 0
    total_classifcation_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device)
        for i in range(len(targets)):
          targets[i] = targets[i].to(device)

        optimizer.zero_grad()
        # calculate classification loss
        outputs = [calculate_noisy_posterior(net_shared, layers_specific[i], inputs) for i in range(num_tasks)] #noisy posterior
        losses = [criterion(torch.log(outputs[i]), targets[i]) for i in range(num_tasks)]
        classification_loss = sum(losses)

        # calculate multitask loss
        if trade_off > 0:            
            weight_shape = layers_specific[0].module[-1].weight.size() # 100x512  
            weight_list = [layers_specific[i].module[-1].weight.view(1, weight_shape[0], weight_shape[1]) for i in range(num_tasks)] # a list containing three tasks' weights
            weight_tensor = torch.cat(weight_list, dim=0).contiguous() # 3x100x512, a 3d weight tensor   

            multi_task_loss = MultiTaskLoss(weight_tensor, task_cov_var, class_cov_var, feature_cov_var)
            loss = classification_loss + trade_off * multi_task_loss
        else:
            loss = classification_loss

        # update network parameters
        loss.backward()
        optimizer.step()

        # increment total loss and corrects
        train_loss += loss.item()
        total_multi_class_loss += multi_task_loss.item()
        total_classifcation_loss += classification_loss.item()

        total += num_tasks * targets[0].size(0)
        for i in range(num_tasks):
            _, predicted_i = outputs[i].max(1)
            correct += predicted_i.eq(targets[i]).sum().item()

        progress_bar(batch_idx, len(train_loader), 'Train C Loss : %.3f |  Multi Loss : %.3f | Acc: %.3f%% (%d/%d) '
                     % (total_classifcation_loss/(batch_idx+1), total_multi_class_loss/(batch_idx+1), 100.*correct/total, correct, total))


def validation(epoch):
    global best_val_acc
    net_shared.eval()
    for i in range(num_tasks):
        layers_specific[i].eval()

    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs = inputs.to(device)
            for i in range(len(targets)):
              targets[i] = targets[i].to(device)

            outputs = [calculate_noisy_posterior(net_shared, layers_specific[i], inputs) for i in range(num_tasks)] #noisy posterior
            losses = [criterion(torch.log(outputs[i]), targets[i]) for i in range(num_tasks)]
            classification_loss = sum(losses)
            val_loss += classification_loss

            # calculate accuracy
            total += num_tasks * targets[0].size(0)

            for i in range(num_tasks):
                _, predicted_i = outputs[i].max(1)
                correct += predicted_i.eq(targets[i]).sum().item()

            progress_bar(batch_idx, len(val_loader), 'Val C Loss: %.3f | Val Acc: %.3f%% (%d/%d)'
                         % (val_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    val_acc = 100.*correct/total
    if val_acc > best_val_acc:
        print('Saving..')
        state_clean_net = {
            'net': deepcopy(net_shared.state_dict()), # deepcopy instead of a reference 
            'acc': val_acc,
            'epoch': epoch,
            'optimizer': deepcopy(optimizer.state_dict()),
            'scheduler': deepcopy(scheduler.state_dict())
        }
        state_T1 = {
            'net': deepcopy(layers_specific[0].state_dict()), # deepcopy instead of a reference 
            'acc': val_acc,
            'epoch': epoch,
            'optimizer': deepcopy(optimizer.state_dict()),
            'scheduler': deepcopy(scheduler.state_dict())
        }
        state_T2 = {
            'net': deepcopy(layers_specific[1].state_dict()), # deepcopy instead of a reference 
            'acc': val_acc,
            'epoch': epoch,
            'optimizer': deepcopy(optimizer.state_dict()),
            'scheduler': deepcopy(scheduler.state_dict())
        }
        state_T3 = {
            'net': deepcopy(layers_specific[2].state_dict()), # deepcopy instead of a reference 
            'acc': val_acc,
            'epoch': epoch,
            'optimizer': deepcopy(optimizer.state_dict()),
            'scheduler': deepcopy(scheduler.state_dict())
        }
        # save the custom defined model state
        torch.save(state_clean_net, MTL_clean_net_save_path) 
        torch.save(state_T1, MTL_T1_save_path) 
        torch.save(state_T2, MTL_T2_save_path) 
        torch.save(state_T3, MTL_T3_save_path) 
        best_val_acc = val_acc

def test_per_epoch(test_loader, model_clean):

    acc = 0 
    total = 0 

    with torch.no_grad(): 
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model_clean.module.clean_logit(inputs) #clean logit
            _, predicted = outputs.max(1)

            total += targets.size(0)
            acc += predicted.eq(targets).sum().item()

        print(f'Testing Accuracy : {round(float(acc)*100/total, 2)}%')

def test_final(model_save_path, test_loader): 
    # initialise the model and its parameters
    net = ResNet34_without_t() 
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
net_shared = ResNet34_without_t()
layers_specific = [[nn.Linear(512, 100, bias=False).cuda()] for i in range(num_tasks)]
for i in range(num_tasks):
  for layer in layers_specific[i]:
      layer.weight.data.normal_(0, 0.01)

layers_specific = [nn.Sequential(*val) for val in layers_specific]

# put the model to GPU if available
net_shared = net_shared.to(device)
for i in range(num_tasks):
    layers_specific[i] = layers_specific[i].to(device)

if device == 'cuda':
  net_shared = torch.nn.DataParallel(net_shared)
  for i in range(num_tasks):
    layers_specific[i] = torch.nn.DataParallel(layers_specific[i])
  cudnn.benchmark = True

load_T_to_layer(layers_specific, T_paths)
# load_C(net_shared, clean_path)

criterion = nn.NLLLoss() #compare logsoftmax probability and target

parameter_dict = [{"params":net_shared.module.parameters(), "lr":3e-4}]
parameter_dict += [{"params":layers_specific[i].module.parameters(), "lr":3e-5} for i in range(num_tasks)] 

optimizer = optim.Adam(
    parameter_dict,
    lr=args.lr, 
    weight_decay=5e-4
)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)

# initialise covariance
trade_off = 1
num_tasks = 3
output_num = 100
bottleneck_size = 512

# initialize covariance matrices as identity matrices
task_cov = torch.eye(num_tasks)
class_cov = torch.eye(output_num)
feature_cov = torch.eye(bottleneck_size)

task_cov_var = Variable(task_cov).cuda()
class_cov_var = Variable(class_cov).cuda()
feature_cov_var = Variable(feature_cov).cuda()

# training and validation
for epoch in range(start_epoch, start_epoch + args.n_epoch):
    train(epoch, task_cov_var, class_cov_var, feature_cov_var)
    # end of every epoch update covariance
    task_cov_var = update_cov(task_cov_var, class_cov_var, feature_cov_var)
    validation(epoch)
    test_per_epoch(test_loader, net_shared)
    scheduler.step()

# # Testing
# test_final(MTL_T1_save_path, test_loader)
# test_final(MTL_T2_save_path, test_loader)
# test_final(MTL_T3_save_path, test_loader)
