from itertools import cycle
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(Epoch, train_labeled_loader, test_loader, model, eval, loss, optimizer, weight): # passing optional arguments as dictionary step_size=0.002, epsilon=0.02, perturb_steps=1, beta=1.0
  running_loss = [] 
  acc = []
  acc_inclass = []

  scheduler = StepLR(optimizer, step_size=8, gamma=0.1)
  for epoch in range(Epoch):
    for i, sample in enumerate(train_labeled_loader, 0): 
      # print('epoch: {}, iteration: {}'.format(epoch, i)) 
      y, x_continuous = sample['y'].float().to(device), sample['x_continuous'].float().to(device)
      x_labeled = x_continuous

      # zero grad
      optimizer.zero_grad()

      loss_all = loss(x_labeled, [y], weight, model)

      loss_all.backward()

      optimizer.step()

      # # check gradient norm
      # total_norm = 0 
      # # for p in model.parameters():
      # #   param_norm = p.grad.data.norm(2)
      # #   total_norm += param_norm.item() ** 2
      # # total_norm = total_norm ** (1. / 2)

      running_loss.append([loss_all.item()])


    # evaluating on test set after after epoch
    ac, ac_inclass, mae, results = eval(test_loader, model, epoch+1, running_loss)
    acc.append(ac)
    acc_inclass.append(ac_inclass)
    # adjust learning rate 
    scheduler.step()

  # print('unlabeled files used: {}'.format(used_files))
  return acc, acc_inclass, running_loss, mae, results
  

def eval(loader, model, epoch, running_loss):
  with torch.no_grad():
    correct_t = torch.zeros(2) 
    total_t = torch.zeros(2) 
    all_AE = torch.zeros(1).to(device)
    results = []


    for i, sample in enumerate(loader, 0):
        y_logit, x_continuous = sample['y'].float().to(device), sample['x_continuous'].float().to(device)
        y = torch.zeros(len(y_logit))

        for j in range(len(y_logit)):
          if y_logit[j] >= 0.5: #0.5: # threshold = (0 - neun.min())/(neun.max() - neun.min()) 0.4078
            y[j] = 1
          else:
            y[j] = 0
        x = x_continuous
        yhat = model.predict(x)[0]
        y_pred_logit = model(x)[0]
        argmax = nn.Softmax(dim=1)
        y_scaled = argmax(y_pred_logit)
        # print(y_pred_logit, y_scaled)
        y_pred = y_scaled[:,1]
        # print(y_pred_logit, y_scaled, y_pred)

        ct, tt = confusion(yhat, y)
        correct_t += ct
        total_t += tt

        pred_results = torch.stack((y_logit, y_pred))
        results.append(pred_results)

        abs_batch = torch.abs(y_logit - y_pred).to(device)
        AE_batch = torch.sum(abs_batch).to(device)
        all_AE += AE_batch
        
    results = torch.cat(results, dim=-1)

    #total acc

    acc = correct_t.sum()/total_t.sum()
    MAE = all_AE/ total_t.sum()

    #inclass acc
    print(correct_t, total_t)
    acc_inclass = (correct_t/total_t).numpy()

    print('test -- Epoch: {}, Accs for task: {:4f}/{}, total/clean/reg/grad: {}'.format(epoch, acc, acc_inclass, running_loss[-1] if running_loss else 'initial'))
    print('MAE = ', MAE)
    return [acc], [acc_inclass], MAE, results

def confusion(yhat, y):
  correct_zero = ((yhat == y) * (y == 0)).sum().item()
  correct_one = ((yhat == y) * (y == 1)).sum().item()
  total_zero = (y == 0).sum().item()
  total_one = (y==1).sum().item()
  return torch.Tensor([correct_zero, correct_one]), torch.Tensor([total_zero, total_one])



def ft_train(Epoch, train_labeled_loader, test_loader, model, init_model, eval, loss, optimizer, weight, l2_lambda): 
  running_loss = [] 
  acc = []
  acc_inclass = []

  scheduler = StepLR(optimizer, step_size=8, gamma=0.1)
  for epoch in range(Epoch):
    for i, sample in enumerate(train_labeled_loader, 0): 
      # print('epoch {}, iteration {}'.format(epoch, i))
      y, x_continuous = sample['y'].float().to(device), sample['x_continuous'].float().to(device)
      x_labeled = x_continuous

      # zero grad
      optimizer.zero_grad()

      loss_all, l2_loss = loss(x_labeled, [y], weight, model, init_model, l2_lambda)

      loss_all.backward()

      optimizer.step()

      running_loss.append([loss_all.item(), l2_loss.item()])


    # evaluating on test set after after epoch
    ac, ac_inclass, MAE, results = eval(test_loader, model, epoch+1, running_loss)
    acc.append(ac)
    acc_inclass.append(ac_inclass)
    # adjust learning rate 
    scheduler.step()

  return acc, acc_inclass, running_loss, MAE, results
  

def ft_eval(loader, model, epoch, running_loss):
  with torch.no_grad():
    correct_t = torch.zeros(2) 
    total_t = torch.zeros(2) 
    all_AE = torch.zeros(1).to(device)
    results = []


    for i, sample in enumerate(loader, 0):
        y_logit, x_continuous = sample['y'].float().to(device), sample['x_continuous'].float().to(device)
        y = torch.zeros(len(y_logit))

        for j in range(len(y_logit)):
          if y_logit[j] >= 0.5: 
            y[j] = 1
          else:
            y[j] = 0

        x = x_continuous

        T = 5
        y_pred_simu = []
        for i in range(T): 
          _y_pred_logit = model(x)[0]
          argmax = nn.Softmax(dim=1)
          _y_scaled = argmax(_y_pred_logit)
          _y_pred = _y_scaled[:,1]
          y_pred_simu.append(_y_pred)

        
        y_pred_all = torch.stack(y_pred_simu)
        y_pred = torch.mean(y_pred_all, 0)
        # print('y pred avg', y_pred)
        y_pred_std = torch.std(y_pred_all, 0, unbiased = False)
        # print('y pred std', y_pred_std)
        y_pred_entropy = - y_pred * torch.log(y_pred) - (1 - y_pred)* torch.log(1 - y_pred)
        # print('entropy', y_pred_entropy)

        yhat = model.predict(x)[0]
        print('yhat', yhat)
        for i in range(len(yhat)):
          if y_pred[i] > 0.5:
            yhat[i] = 1
          else:
            yhat[i] = 0

        print('yhat new', yhat)

        ct, tt = confusion(yhat, y)
        correct_t += ct
        total_t += tt

        pred_results = torch.stack((y_logit, y_pred, y_pred_std, y_pred_entropy))
        results.append(pred_results)

        abs_batch = torch.abs(y_logit - y_pred).to(device)
        AE_batch = torch.sum(abs_batch).to(device)
        all_AE += AE_batch
        
    results = torch.cat(results, dim=-1)

    acc = correct_t.sum()/total_t.sum()
    MAE = all_AE/ total_t.sum()


    print(correct_t, total_t)
    acc_inclass = (correct_t/total_t).numpy()

    print('test -- Epoch: {}, Accs for task: {:4f}/{}, total/clean/reg/grad: {}'.format(epoch, acc, acc_inclass, running_loss[-1] if running_loss else 'initial'))

    return [acc], [acc_inclass], MAE, results

