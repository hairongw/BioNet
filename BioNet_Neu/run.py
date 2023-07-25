import os 
import numpy as np
import torch 
from learn import *
from networks import *
from loss import *
from data import *
import math
import copy 


def run(LR, input_size, output_sizes, hidden_sizes, Epoch, k, K, seed, batch_size_label):
  train_split('Copy of virtual_biop(5000+5000).csv', k, K, seed) 
  train_labeled_data = BrainTumorDataset(['train_labeled_data.csv'], weight_display=True) 
  test_data = BrainTumorDataset(['biopsies69_5*5_threshold0.5.csv']) 
  train_labeled_loader = torch.utils.data.DataLoader(train_labeled_data, batch_size=batch_size_label, shuffle=True, num_workers=2)
  test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size_label, shuffle=True, num_workers=2)


  np.random.seed(seed)

  percentage = train_labeled_data.percentage
  percentage = torch.tensor([math.exp(-0*percentage[0]), math.exp(-0*percentage[1])])
  weight = percentage

  model = SimpleMTL(input_size, output_sizes, hidden_sizes).to(device)
  pytorch_total_params = sum(p.numel() for p in model.parameters())
  print('num params', pytorch_total_params)
  optimizer = torch.optim.Adam(model.parameters(), lr=LR)
  # evaluate initial model
  print('evaluating random initialization')
  acc = [] 
  acc_inclass = [] 
  ac, ac_inclass, a, b = eval(test_loader, model, 0, [])
  acc.append(ac)
  acc_inclass.append(ac_inclass)
  # training 
  print('begin train')
  accf, accf_inclass, running_loss, mae, results = train(Epoch, train_labeled_loader, test_loader, model, eval, clean_loss, optimizer, weight)
  acc = acc + accf
  acc_inclass = acc_inclass + accf_inclass
  return acc, acc_inclass, running_loss, model, mae, results



def run_finetune(LR, input_size, output_sizes, hidden_sizes, Epoch, k, K, seed, batch_size_label, l2_lambda):
  os.chdir(r"/content/drive/MyDrive/RecurrentGBM_soft_classification_all_features/") 
  cwd = os.getcwd()
  train_split_neighbors('biopsies69_5*5_threshold0.5.csv', k, K, seed) 

  train_labeled_data = BrainTumorDataset(['all_labeled_neighbors.csv'], weight_display=True)
  test_data = BrainTumorDataset(['all_labeled_data.csv']) 
  train_labeled_loader = torch.utils.data.DataLoader(train_labeled_data, batch_size=batch_size_label, shuffle=True, num_workers=2)
  test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size_label, shuffle=True, num_workers=2)


  np.random.seed(seed)

  percentage = train_labeled_data.percentage
  percentage = torch.tensor([math.exp(-0*percentage[0]), math.exp(-0*percentage[1])])
  weight = percentage

  # initialized ft_model 
  ft_model = SimpleMTL(input_size, output_sizes, hidden_sizes).to(device)
  ft_model.load_state_dict(torch.load('./pre_trained_model.pth'))

  init_model = SimpleMTL(input_size, output_sizes, hidden_sizes).to(device)
  init_model.load_state_dict(torch.load('./pre_trained_model.pth'))
  


  pytorch_total_params = sum(p.numel() for p in ft_model.parameters())
  print('num params', pytorch_total_params)

  # freeze all layers except for the last one
  # for name, param in ft_model.named_parameters():
  #   if not (str(name) == 'towers.0.4.weight' or str(name) == 'towers.0.4.bias'):
  #     param.requires_grad = False
  #   else:
  #     param.requires_grad = True

  # for name, param in ft_model.named_parameters():
  #   print(name, param.requires_grad)  

  optimizer = torch.optim.Adam(ft_model.parameters(), lr=LR)

  # evaluate initial ft_model
  print('evaluating random initialization')
  acc = [] 
  acc_inclass = [] 
  ac, ac_inclass, a, b = ft_eval(test_loader, ft_model, 0, [])
  acc.append(ac)
  acc_inclass.append(ac_inclass)
  # training 
  print('begin train')
  accf, accf_inclass, running_loss, MAE, results = ft_train(Epoch, train_labeled_loader, test_loader, ft_model, init_model, eval, ft_loss, optimizer, weight, l2_lambda)
  acc = acc + accf
  acc_inclass = acc_inclass + accf_inclass
  return acc, acc_inclass, running_loss, ft_model, MAE, results



