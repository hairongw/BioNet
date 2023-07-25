import torch
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define loss   
def mtl_loss(x, y, weight, criterion, model): 
    argmax = nn.Softmax(dim=1)
    logits = model(x)
    loss = 0
    for (i, logit) in enumerate(logits):
      pred_logit = argmax(logit)
      loss += criterion(pred_logit, y[i], weight) 
    return loss

def l_loss(logit, y, weight=None): 
  y_c = torch.stack((1-y, y)).T
  prod = y_c * (-torch.log(logit))
  weightbar = torch.zeros(len(y))
  for i in range(len(y)):
    if y[i]>= 0.5: 
      weightbar[i] = weight[1]
    else:
      weightbar[i] = weight[0]
  prod_sum = torch.sum(prod, axis=1)
  wloss = torch.sum(prod_sum.dot(weightbar.cuda())/torch.sum(weight))
  return wloss 

def clean_loss(x_labeled, y, weight, model):
  loss = (mtl_loss(x_labeled, y, weight, l_loss, model))/x_labeled.shape[0] 
  return loss




def ft_loss(x_labeled, y, weight, model, init_model, l2_lambda):
  clean_loss = (mtl_loss(x_labeled, y, weight, l_loss, model))/x_labeled.shape[0] 
  l2_reg = torch.tensor(0.).to(device)
  for param, init_param in zip(model.parameters(), init_model.parameters()):
    l2_reg += (torch.norm(param - init_param)**2).to(device)

  l2_loss = l2_lambda * l2_reg
  loss = clean_loss + l2_loss
  return loss, l2_loss