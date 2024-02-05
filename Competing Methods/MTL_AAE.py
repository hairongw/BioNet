import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
import matplotlib.gridspec as gridspec
from torch.autograd import Variable
import pandas as pd
import math
import sklearn.preprocessing as sk
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
import random
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from functools import reduce
from itertools import chain
from PIL import Image
from matplotlib import cm
import copy 
import os
from os import listdir
import re
from torch.optim.lr_scheduler import StepLR
from scipy.stats import norm
import matplotlib.pyplot as plt
from torch import sigmoid
from copy import deepcopy
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


gpu_info = !nvidia-smi
gpu_info = '\n'.join(gpu_info)
if gpu_info.find('failed') >= 0:
  print('Select the Runtime > "Change runtime type" menu to enable a GPU accelerator, ')
  print('and then re-execute this cell.')
else:
  print(gpu_info)


def train_split_neighbors(file, k, K, seed):
  # generate train-test split for labeled-data 
  labeled_data = pd.read_csv(file, sep=',')
  n = labeled_data.shape[0]
  # random shuffle data 
  np.random.seed(seed)
  shuffle_idx = np.random.permutation(n)
  labeled_data = labeled_data.iloc[shuffle_idx, :]
  labeled_data.reset_index(drop=True)

  os.chdir(r"/content/drive/MyDrive/Project 1: Recurrent GBM/RecurrentGBM_soft classification_all features/") 
  cwd = os.getcwd()
  neighbors = pd.read_csv('biopsies69_neighbors_3marker_all5_threshold0.5.csv', sep=',') 
  
  slices = np.linspace(0, n, K+1, dtype=int)
  lb = slices[k]
  ub = slices[k+1]
  train_idx = list(range(lb)) + list(range(ub, n))

  train_labeled_data = labeled_data
  train_labeled_data = labeled_data.iloc[train_idx, :]   # train by all data if delete this line
  train_labeled_data.to_csv('train_labeled_data.csv', index=False)

  train_labeled_all = labeled_data
  train_labeled_all.to_csv('all_labeled_data.csv', index=False) 

  train_labeled_AVID = train_labeled_data['AVID']
  df = neighbors.merge(train_labeled_AVID, how='inner', on='AVID')
  train_labeled_neighbors = pd.concat([df, train_labeled_data], axis = 0)
  train_labeled_neighbors.to_csv('train_labeled_neighbors.csv', index=False)
    
  all_labeled_AVID = train_labeled_all['AVID']
  df = neighbors.merge(all_labeled_AVID, how='inner', on='AVID')
  all_labeled_neighbors = pd.concat([df, train_labeled_all], axis = 0)
  all_labeled_neighbors.to_csv('all_labeled_neighbors.csv', index=False)


  test_labeled_data = labeled_data.iloc[lb:ub, :]
  test_labeled_data.to_csv('test_labeled_data.csv', index=False)

  test_labeled_AVID = test_labeled_data['AVID']
  df2 = neighbors.merge(test_labeled_AVID, how='inner', on='AVID')
  test_labeled_neighbors = pd.concat([df2, test_labeled_data], axis = 0)
  test_labeled_neighbors.to_csv('test_labeled_neighbors.csv', index=False)

class BrainTumorDataset(Dataset):
    def __init__(self, file_list=['biopsies69_neighbors_3marker_all5_threshold0.5.csv'], include_label=[True], weight_display=False): #biopsies_scaledNeuN --- acc only on biopsies
        self.patient_id = []
        self.continuous = []
        self.label = [] 
        self.location = []
        self.avid = []

        for i, csv_file in enumerate(file_list):
          data = pd.read_csv(csv_file, sep=',')
          size = data.shape[0]
          self.patient_id.append(np.array(data.iloc[:,0]))
          self.location.append(np.array(data.iloc[:,2:5]))
          self.continuous.append(np.array(data.iloc[:, -336:])) 

          if include_label[i]:
            self.label.append(np.array(data.iloc[:,6:8])) 
            self.avid.append(np.array(data.iloc[:,1]))
          else:
            self.label.append(np.ones((size, 2), dtype='int64')*(-1))  #two fake labels for unlabeled data
            self.avid.append(np.zeros((size, 1), dtype='int64')) 
          

        self.patient_id = np.concatenate(self.patient_id)
        self.location = np.concatenate(self.location)
        self.continuous = np.concatenate(self.continuous)
        self.label = np.concatenate(self.label)
        self.avid = np.concatenate(self.avid)

        if include_label[i]:
          num_A0 = sum(self.label[:,0]<0.5)
          num_A1 = sum(self.label[:,0]>=0.5)
          total_size = self.label.shape[0]

          self.percentage = [num_A0/total_size, num_A1/total_size]
          print("percentage {:4f}(y1=0), percentage {:4f}(y1=1)".format(num_A0/total_size, num_A1/total_size))

    def __len__(self):
        return len(self.patient_id)

    def __getitem__(self, idx):
        y1 = self.label[idx,0]
        y2 = self.label[idx,1]
        col_idx = [i for i in range(0, 224)] + [i for i in range(280, 336)] #to exclude redundant image: [224, 280]
        x = self.continuous[:,col_idx]
        x_continuous = x[idx,:]
        x_location = self.location[idx, 0:3]
        x_patient_id = self.patient_id[idx]
        x_avid = self.avid[idx]
        sample = {'y1': y1,'y2': y2, 'x_continuous':x_continuous, 'x_location':x_location, 'patient_id':x_patient_id, 'AVID':x_avid}
        return sample




class Q_net(nn.Module):
    '''
    Encoder network.
    '''
    def __init__(self, input_size, hidden_sizes, z_size, output_sizes):
        super(Q_net, self).__init__()
        self.shared = nn.Sequential(nn.Dropout(p=1e-1),nn.Linear(input_size, hidden_sizes[0]), nn.ReLU())
        self.hidden_sizes = hidden_sizes.copy()

        self.latent1 = nn.Sequential(nn.Dropout(p=1e-1),nn.Linear(hidden_sizes[0], hidden_sizes[1]), nn.ReLU(), nn.Linear(hidden_sizes[1], z_size[0]))
        self.latent2 = nn.Sequential(nn.Dropout(p=1e-1),nn.Linear(hidden_sizes[0], hidden_sizes[1]), nn.ReLU(), nn.Linear(hidden_sizes[1], z_size[1]))

        # if no share layers
        self.latent1 = nn.Sequential(nn.Dropout(p=1e-1),nn.Linear(input_size, hidden_sizes[0]), nn.ReLU(), nn.Dropout(p=1e-1),nn.Linear(hidden_sizes[0], hidden_sizes[1]), nn.ReLU(), nn.Linear(hidden_sizes[1], z_size[0]))
        self.latent2 = nn.Sequential(nn.Dropout(p=1e-1),nn.Linear(input_size, hidden_sizes[0]), nn.ReLU(), nn.Dropout(p=1e-1),nn.Linear(hidden_sizes[0], hidden_sizes[1]), nn.ReLU(), nn.Linear(hidden_sizes[1], z_size[1]))


    def forward(self, x):
          z_gauss1 = self.latent1(x)
          z_gauss2 = self.latent2(x)
          return z_gauss1,z_gauss2



class P_net(nn.Module):
    '''
    Decoder network.
    '''
    def __init__(self, input_size, hidden_sizes, z_size, output_sizes): #two tasks --- no dropoout??
        super(P_net, self).__init__()
        self.lin1 = nn.Sequential(nn.Dropout(p=1e-1), nn.Linear(z_size, hidden_sizes[0]), nn.ReLU())
        self.lin3 = nn.Sequential(nn.Dropout(p=1e-1), nn.Linear(hidden_sizes[1], input_size))

    def forward(self, x):
        x = self.lin1(x)
        x = self.lin3(x)
        return x


class D_net_cat(nn.Module):
    '''
    Categorical descriminator network.
    '''
    def __init__(self, output_sizes, hidden_sizes):
        super(D_net_cat, self).__init__()

        self.hidden_sizes = hidden_sizes.copy()
        self.towers = [] 
        self.tasks = len(output_sizes)
        for output_size in output_sizes:
          tower = []
          tower.append(nn.Linear(output_size, self.hidden_sizes[0]))
          tower.append(nn.Dropout(p=1e-1))
          tower.append(nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1]))
          tower.append(nn.ReLU())
          tower.append(nn.Linear(self.hidden_sizes[1],1))
          tower.append(nn.Sigmoid())
          self.towers.append(nn.Sequential(*tower))
        self.towers = nn.ModuleList(self.towers)

    def forward(self, x):
        outputs = []
        for tower in self.towers:
          z = tower(x)
          outputs.append(z)

        return outputs 


class D_net_gauss(nn.Module):
    '''
    Gaussian descriminator network.
    '''
    def __init__(self, z_size, hidden_sizes):
      super(D_net_gauss, self).__init__()
      self.shared = nn.Sequential(nn.Linear(z_size, hidden_sizes[0]), nn.ReLU(), nn.Linear(hidden_sizes[0], 1), nn.Sigmoid())


    def forward(self, x):
        z = self.shared(x)
        output = z
        return output

class Sample_net_gauss(nn.Module):
    '''
    Learned prior for gaussian samples
    '''
    def __init__(self, z_size, hidden_sizes):
      super(Sample_net_gauss, self).__init__()
      self.shared = nn.Sequential(nn.Linear(z_size, z_size))


    def forward(self, x):
        z = self.shared(x)
        output = z
        return output


class Sample_net_cat(nn.Module):
    '''
    Learned prior for categorical samples
    '''
    def __init__(self, hidden_sizes):
      super(Sample_net_cat, self).__init__()
      self.shared = nn.Sequential(nn.Dropout(p=1e-1), nn.Linear(2, hidden_sizes[0]), nn.ReLU(),nn.Dropout(p=1e-1), nn.Linear(hidden_sizes[0], hidden_sizes[1]), nn.ReLU(), nn.Linear(hidden_sizes[1], 2))


    def forward(self, x):
        z = self.shared(x)
        output = z
        return output


class classifier(nn.Module):
    '''
    classification from latent z to label
    '''
    def __init__(self, z_size, hidden_sizes):
      super(classifier, self).__init__()
      self.shared = nn.Sequential(nn.Dropout(p=1e-1), nn.Linear(z_size, hidden_sizes[0]), nn.ReLU(), nn.Linear(hidden_sizes[0], hidden_sizes[1]), nn.ReLU(), nn.Linear(hidden_sizes[1], 2))

    def forward(self, x):
        y = self.shared(x)
        return [y]


def reconstruction_loss1(x_labeled, x_low, x_high, P1, Q, D_cat1, D_gauss1):
    x = torch.cat((x_labeled, x_low, x_high),0)
    latent, _ = Q(x)
    x_rec = P1(latent)
    loss_criterion = nn.MSELoss()
    R_loss = loss_criterion(x_rec, x)
    return R_loss

def discriminator_loss1(x_labeled, x_low, x_high, z_size, batch_size, P1, Q, D_cat1, D_gauss1, Sample_cat1, Sample_gauss1):
    x = torch.cat((x_labeled, x_low, x_high),0)
    l = len(x)
    z_real_gauss = Variable(torch.randn(l, z_size[0]))
    z_real_gauss = z_real_gauss.to(device)
    # learned priors for z
    z_real_gauss = Sample_gauss1(z_real_gauss)
    z_fake_gauss, _ = Q(x)
    D_real_gauss = D_gauss1(z_real_gauss)
    D_fake_gauss = D_gauss1(z_fake_gauss)
    D_loss_gauss = - torch.mean(D_real_gauss - D_fake_gauss)
    D_loss = D_loss_gauss

    return D_loss

def generator_loss1(x_labeled, x_low, x_high, P1, Q, D_cat1, D_gauss1):
    x = torch.cat((x_labeled, x_low, x_high),0)
    z_fake_gauss, _ = Q(x)
    D_fake_gauss = D_gauss1(z_fake_gauss)
    G_loss = - torch.mean(D_fake_gauss)

    return G_loss


def mtl_loss1(x, y1, weight, criterion, P1, Q, D_cat1, D_gauss1, cl1): 
    argmax = nn.Softmax(dim=1)
    latent, _ = Q(x)
    logits = cl1(latent)
    loss = 0
    for (i, logit) in enumerate(logits):
      pred_logit = argmax(logit)
      loss += criterion(pred_logit, y1[i], weight) #for tumor prediction

    return loss


def l_loss1(logit, y1, weight): # regression: no weights 
  loss_criterion = nn.L1Loss() #nn.L1Loss() #nn.MSELoss()
  prob = logit[:,1]
  wloss = loss_criterion(prob.view(-1), y1)
  return wloss 


def clean_loss1(x_labeled, y1, weight, P1, Q, D_cat1, D_gauss1, cl1):
  loss = (mtl_loss1(x_labeled, y1, weight, l_loss1, P1, Q, D_cat1, D_gauss1, cl1)) #+ consistency_loss(x_labeled, model, optimizer, step_size_adv, delta, perturb_steps, alpha_adv))/x_labeled.shape[0]
  return loss

def reconstruction_loss2(x_labeled, x_low, x_high, P2, Q, D_cat2, D_gauss2):
    x = torch.cat((x_labeled, x_low, x_high),0)
    _, latent= Q(x)
    x_rec = P2(latent)
    loss_criterion = nn.MSELoss()
    R_loss = loss_criterion(x_rec, x)
    return R_loss

def discriminator_loss2(x_labeled, x_low, x_high, z_size, batch_size, P2, Q, D_cat2, D_gauss2, Sample_cat2, Sample_gauss2):
    x = torch.cat((x_labeled, x_low, x_high),0)
    l = len(x)
    z_real_gauss = Variable(torch.randn(l, z_size[1]))
    z_real_gauss = z_real_gauss.to(device)
    # learned priors for z
    z_real_gauss = Sample_gauss2(z_real_gauss)
    _, z_fake_gauss= Q(x)
    D_real_gauss = D_gauss2(z_real_gauss)
    D_fake_gauss = D_gauss2(z_fake_gauss)
    D_loss_gauss = - torch.mean(D_real_gauss - D_fake_gauss)
    D_loss = D_loss_gauss

    return D_loss

def generator_loss2(x_labeled, x_low, x_high, P2, Q, D_cat2, D_gauss2):
    x = torch.cat((x_labeled, x_low, x_high),0)
    _, z_fake_gauss = Q(x)
    D_fake_gauss = D_gauss2(z_fake_gauss)
    G_loss = - torch.mean(D_fake_gauss)
    return G_loss


def mtl_loss2(x, y2, weight, criterion, P2, Q, D_cat2, D_gauss2, cl2):
    argmax = nn.Softmax(dim=1)
    _, latent = Q(x)
    logits = cl2(latent)
    loss = 0
    for (i, logit) in enumerate(logits):
      pred_logit = argmax(logit)
      loss += criterion(pred_logit, y2[i], weight) 

    return loss


def l_loss2(logit, y2, weight):
  loss_criterion = nn.L1Loss() 
  prob = logit[:,1]
  wloss = loss_criterion(prob.view(-1), y2)
  return wloss 

def clean_loss2(x_labeled, y2, weight, P2, Q, D_cat2, D_gauss2, cl2):
  loss = (mtl_loss2(x_labeled, y2, weight, l_loss2, P2, Q, D_cat2, D_gauss2, cl2)) #+ consistency_loss(x_labeled, model, optimizer, step_size_adv, delta, perturb_steps, alpha_adv))/x_labeled.shape[0]
  return loss


def low_loss(x_low, P2, Q, D_cat2, D_gauss2, cl1, cl2): # criterion is the loss
    latent1, latent2 = Q(x_low)
    y_pred_logit1 = cl1(latent1)
    y_pred_logit1 = y_pred_logit1[0]
    y_pred_logit2 = cl2(latent2)
    y_pred_logit2 = y_pred_logit2[0]


    argmax = nn.Softmax(dim=1)
    y_scaled1 = argmax(y_pred_logit1)
    y_scaled2 = argmax(y_pred_logit2)

    loss_low = torch.sum(y_scaled1 * y_scaled2)

    return loss_low/y_scaled1.shape[0]


def high_loss(x_high, P2, Q, D_cat2, D_gauss2, cl1, cl2): # criterion is the loss
    latent1, latent2 = Q(x_high)
    y_pred_logit1 = cl1(latent1)
    y_pred_logit1 = y_pred_logit1[0]
    y_pred_logit2 = cl2(latent2)
    y_pred_logit2 = y_pred_logit2[0]


    argmax = nn.Softmax(dim=1)
    y_scaled1 = argmax(y_pred_logit1)
    y_scaled2 = argmax(y_pred_logit2)

    loss_high = torch.sum(torch.pow(y_scaled1[:,1], 2)) + torch.sum(torch.pow(y_scaled2[:,1], 2))

    return loss_high/y_scaled1.shape[0]


def semi_sup_loss(x_labeled, y1, y2, weight, P1, Q, D_cat1, D_gauss1, P2, D_cat2, D_gauss2, cl1, cl2, x_low, x_high, alpha, beta, gamma, log_var_a, log_var_b):
  r1 = (1 - alpha) * reconstruction_loss1(x_labeled, x_low, x_high, P1, Q, D_cat1, D_gauss1)
  r2 = (1 - alpha) * reconstruction_loss2(x_labeled, x_low, x_high, P2, Q, D_cat2, D_gauss2)
  c1 = alpha * clean_loss1(x_labeled, y1, weight, P1, Q, D_cat1, D_gauss1, cl1)
  c2 = alpha * clean_loss2(x_labeled, y2, weight, P2, Q, D_cat2, D_gauss2, cl2)
  semi_loss1 = c1 + r1
  semi_loss2 = c2 + r2
  un_loss = beta * low_loss(x_low, P2, Q, D_cat2, D_gauss2, cl1, cl2) + gamma * high_loss(x_high, P2, Q, D_cat2, D_gauss2, cl1, cl2)
  semi_loss = semi_loss1 + semi_loss2 + un_loss

  return semi_loss


def zero_grad_all(*models):
    [m.zero_grad() for m in models]
def train(Epoch, train_labeled_loader, neun_low, neun_high, test_loader, Q, P1, D_cat1, D_gauss1, Sample_cat1, Sample_gauss1, P2, D_cat2, D_gauss2, Sample_cat2, Sample_gauss2, cl1, cl2, eval, loss, ae_optimizer1, semi_optimizer, coptimizers,  D_optimizer1,G_optimizer1, optimizer1,ae_optimizer2, D_optimizer2,G_optimizer2, optimizer2, weight, batch_size, z_size, alpha, beta, gamma, log_var_a, log_var_b): # passing optional arguments as dictionary step_size=0.002, epsilon=0.02, perturb_steps=1, beta=1.0

  Q.train()
  P1.train()
  D_cat1.train()
  D_gauss1.train()
  Sample_cat1.train()
  Sample_gauss1.train()
  P2.train()
  D_cat2.train()
  D_gauss2.train()
  Sample_cat2.train()
  Sample_gauss2.train()
  cl1.train()
  cl2.train()

  train_neun_low_data = BrainTumorDataset([neun_low], include_label=[False])
  train_neun_low_loader = torch.utils.data.DataLoader(train_neun_low_data, batch_size=batch_size, shuffle=True, num_workers=2)
  neun_low_loader_iter = iter(train_neun_low_loader)

  train_neun_high_data = BrainTumorDataset([neun_high], include_label=[False])
  train_neun_high_loader = torch.utils.data.DataLoader(train_neun_high_data, batch_size=batch_size, shuffle=True, num_workers=2)
  neun_high_loader_iter = iter(train_neun_high_loader)

  running_loss = [] 
  acc = []
  acc_inclass = []
  used_files = 1

  labeled_loader_iter = iter(cycle(train_labeled_loader)) #use all unlabeled samples

  c1_optimizer, c2_optimizer = coptimizers

  scheduler1 = StepLR(semi_optimizer, step_size=8, gamma=0.5)
  scheduler2 = StepLR(D_optimizer1, step_size=8, gamma=0.5)
  scheduler3 = StepLR(G_optimizer1, step_size=8, gamma=0.5)
  scheduler4 = StepLR(D_optimizer2, step_size=8, gamma=0.5)
  scheduler5 = StepLR(G_optimizer2, step_size=8, gamma=0.5)


  class_loader = iter(cycle(train_labeled_loader))
  disc_loader_low = iter(cycle(train_neun_low_loader))
  disc_loader_high = iter(cycle(train_neun_high_loader))
  disc_loader = iter(cycle(train_labeled_loader))
  for epoch in range(Epoch):
    for i, sample in enumerate(train_labeled_loader, 0): 
      sample = next(labeled_loader_iter)
      y1, y2, x_continuous = sample['y1'].float().to(device), sample['y2'].float().to(device), sample['x_continuous'].float().to(device)
      x_labeled = x_continuous

      # construct next unlabeled dataloader if current one is exhausted 
      try:
        l_sample = next(neun_low_loader_iter)
        x_low =  l_sample['x_continuous']
        h_sample = next(neun_high_loader_iter)
        x_high =  h_sample['x_continuous']

      except StopIteration:
        used_files += 1 
        neun_low_loader_iter = iter(train_neun_low_loader)
        l_sample = next(neun_low_loader_iter)
        x_low =  l_sample['x_continuous']

        neun_high_loader_iter = iter(train_neun_high_loader)
        h_sample = next(neun_high_loader_iter)
        x_high =  h_sample['x_continuous']
      
      x_low = x_low.float().to(device)
      x_high = x_high.float().to(device)

      zero_grad_all(Q, P1, D_cat1, D_gauss1, Sample_cat1, Sample_gauss1, P2, D_cat2, D_gauss2, Sample_cat2, Sample_gauss2, cl1, cl2)
      semi_loss = semi_sup_loss(x_labeled, [y1], [y2], weight, P1, Q, D_cat1, D_gauss1, P2, D_cat2, D_gauss2, cl1, cl2, x_low, x_high, alpha, beta, gamma, log_var_a, log_var_b)
      semi_loss.backward()
      semi_optimizer.step()

      for _ in range(5): #5
        dsample = next(disc_loader)
        # print('epoch: {}, iteration: {}'.format(epoch, i)) 
        dy1, dy2, dx_continuous = dsample['y1'].float().to(device), dsample['y2'].float().to(device), dsample['x_continuous'].float().to(device)
        dx_labeled = dx_continuous

        l_sample = next(disc_loader_low)
        dx_low =  l_sample['x_continuous']
        h_sample = next(disc_loader_high)
        dx_high =  h_sample['x_continuous']

        dx_low = dx_low.float().to(device)
        dx_high = dx_high.float().to(device)

        zero_grad_all(Q, P1, D_cat1, D_gauss1, Sample_cat1, Sample_gauss1, P2, D_cat2, D_gauss2, Sample_cat2, Sample_gauss2, cl1, cl2)
        D_loss1 = discriminator_loss1(dx_labeled, dx_low, dx_high, z_size, batch_size, P1, Q, D_cat1, D_gauss1, Sample_cat1, Sample_gauss1)
        D_loss1.backward()
        D_optimizer1.step()

        zero_grad_all(Q, P1, D_cat1, D_gauss1, Sample_cat1, Sample_gauss1, P2, D_cat2, D_gauss2, Sample_cat2, Sample_gauss2, cl1, cl2)
        D_loss2 = discriminator_loss2(dx_labeled, dx_low, dx_high, z_size, batch_size, P2, Q, D_cat2, D_gauss2, Sample_cat2, Sample_gauss2)
        D_loss2.backward()
        D_optimizer2.step()

        for p in D_gauss1.parameters():
            p.data.clamp_(-0.01, 0.01)
            
        for p in D_gauss2.parameters():
            p.data.clamp_(-0.01, 0.01)

        # print("discriminator loss: {}/{}".format(D_loss1.item(), D_loss2.item()))

      zero_grad_all(Q, P1, D_cat1, D_gauss1, Sample_cat1, Sample_gauss1, P2, D_cat2, D_gauss2, Sample_cat2, Sample_gauss2, cl1, cl2)
      G_loss1 = generator_loss1(x_labeled, x_low, x_high, P1, Q, D_cat1, D_gauss1)
      G_loss1.backward()
      G_optimizer1.step()

      G_loss2 = generator_loss2(x_labeled, x_low, x_high, P2, Q, D_cat2, D_gauss2)
      G_loss2.backward()
      G_optimizer2.step()


      for _ in range(8* (epoch+1)): #4 * (epoch+1)
        csample = next(class_loader)
        # print('epoch: {}, iteration: {}'.format(epoch, i)) 
        cy1, cy2, cx_continuous = csample['y1'].float().to(device), csample['y2'].float().to(device), csample['x_continuous'].float().to(device)
        cx_labeled = cx_continuous

        zero_grad_all(Q, P1, D_cat1, D_gauss1, Sample_cat1, Sample_gauss1, P2, D_cat2, D_gauss2, Sample_cat2, Sample_gauss2, cl1, cl2)
        semi_loss = semi_sup_loss(cx_labeled, [cy1], [cy2], weight, P1, Q, D_cat1, D_gauss1, P2, D_cat2, D_gauss2, cl1, cl2, x_low, x_high, alpha, beta, gamma, log_var_a, log_var_b)
        semi_loss.backward()
        c1_optimizer.step()
        c2_optimizer.step()

      running_loss.append([semi_loss.item()])


    # evaluating on test set after epoch
    ac, ac_inclass, mae, results = eval(test_loader, Q, P1, D_cat1, D_gauss1, P2, D_cat2, D_gauss2, cl1, cl2, epoch+1, running_loss)

    acc.append(ac)
    acc_inclass.append(ac_inclass)
    # adjust learning rate 
    scheduler1.step()
    scheduler2.step()
    scheduler3.step()
    scheduler4.step()
    scheduler5.step()

  return acc, acc_inclass, running_loss, mae, results
  

def eval(loader, Q, P1, D_cat1, D_gauss1, P2, D_cat2, D_gauss2, cl1, cl2, epoch, running_loss):
  Q.eval()
  P1.eval()
  D_cat1.eval()
  D_gauss1.eval()
  P2.eval()
  D_cat2.eval()
  D_gauss2.eval()
  cl1.eval()
  cl2.eval()
  with torch.no_grad():
    correct_t1 = torch.zeros(2) 
    correct_t2 = torch.zeros(2) 
    total_t1 = torch.zeros(2) 
    total_t2 = torch.zeros(2) 
    all_AE1 = torch.zeros(1).to(device)
    all_AE2 = torch.zeros(1).to(device)
    results = []


    for i, sample in enumerate(loader, 0):
        y_logit1, y_logit2, x_continuous = sample['y1'].float().to(device), sample['y2'].float().to(device), sample['x_continuous'].float().to(device)
        y1 = torch.zeros(len(y_logit1))
        y2 = torch.zeros(len(y_logit2))

        y1 = (y_logit1 >= 0.5).long()
        y2 = (y_logit2 >= 0.5).long()

        x = x_continuous
        y1 = y1.to(device)
        y2 = y2.to(device)


        latent1, latent2 = Q(x)
        y_pred_logit1 = cl1(latent1)
        y_pred_logit1 = y_pred_logit1[0]
        y_pred_logit2 = cl2(latent2)
        y_pred_logit2 = y_pred_logit2[0]


        argmax = nn.Softmax(dim=1)
        y_scaled1 = argmax(y_pred_logit1)
        y_scaled2 = argmax(y_pred_logit2)

        _, yhat1 = torch.max(y_scaled1,1) 
        _, yhat2 = torch.max(y_scaled2,1)

        y_pred1 = y_scaled1[:,1] 
        y_pred2 = y_scaled2[:,1]

        ct1, tt1 = confusion(yhat1, y1)
        correct_t1 += ct1
        total_t1 += tt1

        ct2, tt2 = confusion(yhat2, y2)
        correct_t2 += ct2
        total_t2 += tt2

        pred_results = torch.stack((y_logit1, y_pred1, y_logit2, y_pred2))
        results.append(pred_results)

        abs_batch1 = torch.abs(y_logit1 - y_pred1).to(device)
        AE_batch1 = torch.sum(abs_batch1).to(device)
        all_AE1 += AE_batch1

        abs_batch2 = torch.abs(y_logit2 - y_pred2).to(device)
        AE_batch2 = torch.sum(abs_batch2).to(device)
        all_AE2 += AE_batch2
        
    results = torch.cat(results, dim=-1)

    #total acc
    acc1 = correct_t1.sum()/total_t1.sum()
    acc2 = correct_t2.sum()/total_t2.sum()
    MAE1 = all_AE1/ total_t1.sum()
    MAE2 = all_AE2/ total_t2.sum()

    #inclass acc
    print('SOX2',correct_t1, total_t1)
    acc1_inclass = (correct_t1/total_t1).numpy()
    print('CD68',correct_t2, total_t2)
    acc2_inclass = (correct_t2/total_t2).numpy()
    print('test -- Epoch: {}, Accs for SOX2: {:4f}/{}, Accs for CD68: {:4f}/{}, total/clean/reg/grad: {}'.format(epoch, acc1, acc1_inclass, acc2, acc2_inclass, running_loss[-1] if running_loss else 'initial'))
    print('SOX2 MAE = ', MAE1, 'CD68 MAE = ', MAE2)
    return [acc1, acc2], [acc1_inclass, acc2_inclass],[MAE1, MAE2], results

def confusion(yhat, y):
  correct_zero = ((yhat == y) * (y == 0)).sum().item()
  correct_one = ((yhat == y) * (y == 1)).sum().item()
  total_zero = (y == 0).sum().item()
  total_one = (y==1).sum().item()
  return torch.Tensor([correct_zero, correct_one]), torch.Tensor([total_zero, total_one])


def run(patient_id, LR, input_size, output_sizes, hidden_sizes, Epoch, k, K, seed, batch_size, z_size, neun_low_list, neun_high_list, alpha, beta, gamma):

  os.chdir(r"/content/drive/MyDrive/Project 1: Recurrent GBM/5 contrasts AAE/personalized model SOX2 CD68/"+ patient_id + "/") 
  cwd = os.getcwd()
  test_data = BrainTumorDataset(['biopsies69_3marker_5_5_threshold0.5_' + patient_id +'.csv']) # this line must be previous than train_split_neighbors
  train_split_neighbors('biopsies69_3marker_5_5_threshold0.5_no'+ patient_id +'.csv', k, K, seed) 
  train_labeled_data = BrainTumorDataset(['all_labeled_neighbors.csv'], weight_display=True) 
  train_labeled_loader = torch.utils.data.DataLoader(train_labeled_data, batch_size=batch_size, shuffle=True, num_workers=2)
  test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=2)
  os.chdir(r"/content/drive/MyDrive/Project 1: Recurrent GBM/5 contrasts AAE/personalized model SOX2 CD68/"+ patient_id + "/") 
  cwd = os.getcwd()

  np.random.seed(seed)
  percentage = train_labeled_data.percentage
  percentage = torch.tensor([math.exp(-0*percentage[0]), math.exp(-0*percentage[1])])
  weight = percentage
  neun_low = neun_low_list
  neun_high = neun_high_list
  Q = Q_net(input_size, hidden_sizes, z_size, output_sizes).to(device)
  P1 = P_net(input_size, hidden_sizes, z_size[0], output_sizes).to(device)
  D_cat1 = D_net_cat(output_sizes, hidden_sizes).to(device)
  D_gauss1 = D_net_gauss(z_size[0], hidden_sizes).to(device)
  Sample_cat1 = Sample_net_cat(hidden_sizes).to(device)
  Sample_gauss1 = Sample_net_gauss(z_size[0], hidden_sizes).to(device)

  P2 = P_net(input_size, hidden_sizes, z_size[1], output_sizes).to(device)
  D_cat2 = D_net_cat(output_sizes, hidden_sizes).to(device)
  D_gauss2 = D_net_gauss(z_size[1], hidden_sizes).to(device)
  Sample_cat2 = Sample_net_cat(hidden_sizes).to(device)
  Sample_gauss2 = Sample_net_gauss(z_size[1], hidden_sizes).to(device)

  cl1 = classifier(z_size[0], hidden_sizes).to(device)
  cl2 = classifier(z_size[1], hidden_sizes).to(device)

  log_var_a = torch.zeros((1,), requires_grad=True, device="cuda")
  log_var_b = torch.zeros((1,), requires_grad=True, device="cuda")

  std_1 = torch.exp(log_var_a)**0.5
  std_2 = torch.exp(log_var_b)**0.5
  print([std_1.item(), std_2.item()])

  params = ([p for p in Q.parameters()] + [p for p in cl1.parameters()] + [p for p in cl2.parameters()] + [p for p in P1.parameters()]+ [p for p in P2.parameters()] + [log_var_a] + [log_var_b])
  pytorch_Q_params = sum(p.numel() for p in Q.parameters())
  print('Q num params', pytorch_Q_params)
  pytorch_cl_params = sum(p.numel() for p in cl1.parameters())
  print('classifier num params', pytorch_cl_params)

  semi_optimizer = torch.optim.Adam(params, lr=LR['decoder'], weight_decay=1e-5)
  c1_optimizer = torch.optim.Adam([p for p in cl1.parameters()], lr=LR['class'], weight_decay=1e-5)
  c2_optimizer = torch.optim.Adam([p for p in cl2.parameters()], lr=LR['class'], weight_decay=1e-5)

  ae_optimizer1 = torch.optim.Adam(chain(Q.parameters(), P1.parameters()), lr=LR['base'], weight_decay=1e-5)
  D_optimizer1 = torch.optim.Adam(chain(Sample_cat1.parameters(), Sample_gauss1.parameters(), D_cat1.parameters(), D_gauss1.parameters()), lr=LR['disc'], weight_decay=1e-5)
  G_optimizer1 = torch.optim.Adam(Q.parameters(), lr=LR['gen'], weight_decay=1e-5)
  optimizer1 = torch.optim.Adam(Q.parameters(), lr=LR['base'], weight_decay=1e-5)

  ae_optimizer2 = torch.optim.Adam(chain(Q.parameters(), P2.parameters()), lr=LR['base'], weight_decay=1e-5)
  D_optimizer2 = torch.optim.Adam(chain(Sample_cat2.parameters(), Sample_gauss2.parameters(), D_cat2.parameters(), D_gauss2.parameters()), lr=LR['disc'], weight_decay=1e-5)
  G_optimizer2 = torch.optim.Adam(Q.parameters(), lr=LR['gen'], weight_decay=1e-5)
  optimizer2 = torch.optim.Adam(Q.parameters(), lr=LR['base'], weight_decay=1e-5)

  print('evaluating random initialization')
  acc = [] 
  acc_inclass = [] 
  ac, ac_inclass, a, b = eval(test_loader, Q, P1, D_cat1, D_gauss1, P2, D_cat2, D_gauss2, cl1, cl2, 0, [])
  acc.append(ac)
  acc_inclass.append(ac_inclass)
  # training 
  print('begin train')
  accf, accf_inclass, running_loss, mae, results = train(Epoch, train_labeled_loader, neun_low, neun_high, test_loader, Q, P1, D_cat1, D_gauss1, Sample_cat1, Sample_gauss1, P2, D_cat2, D_gauss2, Sample_cat2, Sample_gauss2, cl1, cl2, eval, semi_sup_loss, ae_optimizer1, semi_optimizer, [c1_optimizer, c2_optimizer], D_optimizer1, G_optimizer1, optimizer1, ae_optimizer2, D_optimizer2,G_optimizer2,optimizer2, weight, batch_size, z_size, alpha, beta, gamma, log_var_a, log_var_b)
  acc = acc + accf
  acc_inclass = acc_inclass + accf_inclass
  return acc, acc_inclass, running_loss, Q, P1, D_cat1, D_gauss1, P2, D_cat2, D_gauss2, cl1, cl2, mae, results


def reult_process(df_all):
    pred = pd.concat(df_all, ignore_index=True)
    for i in range(10):
      print(i)
      SOX2_true = pred.iloc[:,4*i+0]
      SOX2_pred = pred.iloc[:,4*i+1]
      CD68_true = pred.iloc[:,4*i+2]
      CD68_pred = pred.iloc[:,4*i+3]

      SOX2_pred_label = (SOX2_pred >= 0.5)
      CD68_pred_label = (CD68_pred >= 0.5)
      SOX2_true_label = (SOX2_true >= 0.5)
      CD68_true_label = (CD68_true >= 0.5)

      # ACC
      acc_SOX2 = accuracy_score(SOX2_true_label, SOX2_pred_label)
      acc_CD68 = accuracy_score(CD68_true_label, CD68_pred_label)

      # AUC
      fpr = dict()
      tpr = dict()
      fpr, tpr, _ = roc_curve(SOX2_true_label.to_numpy(), SOX2_pred.to_numpy())
      roc_auc_SOX2 = auc(fpr, tpr)
      fpr, tpr, _ = roc_curve(CD68_true_label.to_numpy(), CD68_pred.to_numpy())
      roc_auc_CD68 = auc(fpr, tpr)

      # MAE
      MAE_SOX2 = (SOX2_true - SOX2_pred).abs().mean()
      MAE_CD68 = (CD68_true - CD68_pred).abs().mean()

      roc_auc_SOX2 = round(roc_auc_SOX2, 3)
      roc_auc_CD68 = round(roc_auc_CD68, 3)
      acc_SOX2 = round(acc_SOX2, 3)
      acc_CD68 = round(acc_CD68, 3)
      MAE_SOX2 = round(MAE_SOX2, 3)
      MAE_CD68 = round(MAE_CD68, 3)

      print("AUC", roc_auc_SOX2, roc_auc_CD68,
            "ACC", acc_SOX2, acc_CD68, 
            "MAE", MAE_SOX2, MAE_CD68)

    return pred, roc_auc_SOX2, roc_auc_CD68, acc_SOX2, acc_CD68, MAE_SOX2, MAE_CD68

os.chdir(r"/content/drive/MyDrive/Project 1: Recurrent GBM/RecurrentGBM_soft classification_all features/") 
cwd = os.getcwd()
df = pd.read_csv('biopsies69_3marker_5*5_threshold0.5.csv',sep=',')
patients = df["Patient"]
p = patients.drop_duplicates().to_list()
path = r"/content/drive/MyDrive/Project 1: Recurrent GBM/5 contrasts AAE/BioNet Grid search 2/"
if not os.path.exists(path):
  os.makedirs(path)

p_done = [] #["CU1275", "CU1269", "CU1324", "CU1265","CU1253", "CU1101", "CU1154"] #, "CU1324", "CU1275", "CU1265","CU1253", "CU1101", "CU1154"
p_used = [e for e in p if e not in p_done]
print(len(p_used))

chunks = [p_used[x:x+3] for x in range(0, len(p_used), 3)]
print(len(chunks), len(chunks[0]))

starting = 0
print(len(chunks[starting:]))

for i in range(len(chunks[starting:])):
  print(i+starting)
  pred_all = []
  for patient_id in chunks[i+starting]:

    print(patient_id)

    os.chdir(r"/content/drive/MyDrive/Project 1: Recurrent GBM/5 contrasts AAE/personalized model SOX2 CD68/"+ patient_id + "/") 
    cwd = os.getcwd()
    print('working directory {}'.format(cwd))

    neun_low_list = "certain_low_neun_7000_no" + patient_id +".csv"
    print(neun_low_list)
    neun_high_list = "certain_high_neun_7000_no" + patient_id +".csv"
    print(neun_high_list)

    pt_neun_low_list = "certain_low_neun_" + patient_id + "_train.csv"
    print(pt_neun_low_list)
    pt_neun_high_list = "certain_high_neun_" + patient_id + "_train.csv"
    print(pt_neun_high_list)



    LR = {"gen": 1e-3, "disc":1e-3, "class": 1e-3, "decoder": 4e-3, 'base':1e-3}


    input_size = 280 
    output_sizes = [2]
    z_size = [512, 64] 
    batch_size = 128 
    hidden_sizes = [256, 256] 
    Epoch = 9 
    K = 3 
    accs= [] 
    accs_inclass=[]
    Permute = 1 
    alpha = 0.5 
    beta = 0.1 
    gamma = 0.1
    prediction = []
    for k in range(K):  
      perm = 1
      print('Beginning run {}, Total runs {}...'.format(perm * K + k+1, K * Permute))
      acc, acc_inclass, running_loss, Q, P1, D_cat1, D_gauss1, P2, D_cat2, D_gauss2, cl1, cl2, mae, results = run(patient_id, LR, input_size, output_sizes, hidden_sizes, Epoch, k, K, perm, batch_size, z_size, neun_low_list, neun_high_list, alpha, beta, gamma)
      accs.append(acc)
      accs_inclass.append(acc_inclass)
      prediction.append(results)
      print('final mae', mae)
      print(results.size())

      plt.figure(1)
      running_loss = np.array(running_loss)[:,0]
      plt.plot(running_loss, label='loss')

      plt.figure(2)
      acc = np.array(acc)
      acc_inclass = np.array(acc_inclass)
      plt.plot(acc[:,0], label='SOX2') #, label='SOX2'
      plt.plot(acc[:,1], label='CD68')
      plt.legend()

      plt.show()
      
      if k == 0:
        break 
      
    prediction = torch.cat(prediction, dim=-1)

    a1 = torch.transpose(prediction,0,1).cpu().numpy()
    print(a1)

    DF1 = pd.DataFrame(a1)
    DF1.columns = ['SOX2 score', 'SOX2 pred', 'CD68 score', 'CD68 pred','SOX2 score', 'SOX2 pred', 'CD68 score', 'CD68 pred','SOX2 score', 'SOX2 pred', 'CD68 score', 'CD68 pred','SOX2 score', 'SOX2 pred', 'CD68 score', 'CD68 pred','SOX2 score', 'SOX2 pred', 'CD68 score', 'CD68 pred','SOX2 score', 'SOX2 pred', 'CD68 score', 'CD68 pred','SOX2 score', 'SOX2 pred', 'CD68 score', 'CD68 pred','SOX2 score', 'SOX2 pred', 'CD68 score', 'CD68 pred','SOX2 score', 'SOX2 pred', 'CD68 score', 'CD68 pred','SOX2 score', 'SOX2 pred', 'CD68 score', 'CD68 pred']
    pred_all.append(DF1)

  pred, roc_auc_SOX2, roc_auc_CD68, acc_SOX2, acc_CD68, MAE_SOX2, MAE_CD68 = reult_process(pred_all)
  mean_AUC = (roc_auc_SOX2 + roc_auc_CD68)/2
  file_name = str(mean_AUC) + "_" + str(LR) + "_" + str(batch_size) + "_" + str(hidden_sizes) + "_" + str(Epoch) + "_part" + str(i+starting) + ".csv"
  pred.to_csv(path + file_name, index=False)
  print("----- finish one cluster ----- ", i)

