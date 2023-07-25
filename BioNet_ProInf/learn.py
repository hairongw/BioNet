from itertools import cycle
import torch 
import torch.nn as nn
from tqdm import tqdm
import torch.multiprocessing
from tqdm import tqdm
torch.multiprocessing.set_sharing_strategy('file_system')



def train(dataloader, model, optimizers, device, custom_loss_function, loss_param, main_task, version):
    sum_of_loss = torch.Tensor([0,0,0,0,0,0])
    # Iterate through the data loader once (one epoch)
    if version == "new":
        for i, batch in enumerate(tqdm(dataloader)):
            # Zero the parameter gradients
            for optimizer in optimizers:
                optimizer.zero_grad()

            # Calculate loss using the custom loss function
            loss, autoencoder_loss, reconstruct_loss, label_loss, kd_loss, neun_pred_loss, barrier_loss = custom_loss_function(batch, model, device, loss_param, main_task, version)
            # Backward pass
            loss.backward()

            # Update model parameters
            for optimizer in optimizers:
                optimizer.step()

            # Tracking loss 
            sum_of_loss += torch.Tensor([autoencoder_loss, reconstruct_loss, label_loss, kd_loss, neun_pred_loss, barrier_loss])
    
    else:
        dataloader_high, dataloader_low = dataloader
        for i, (batch_high, batch_low) in enumerate(zip(dataloader_high, dataloader_low)):
        # for i, batch_high in enumerate(tqdm(dataloader_high)):
            # for j, batch_low in enumerate(tqdm(dataloader_low)):
                batch = [batch_high, batch_low]
                # Zero the parameter gradients
                for optimizer in optimizers:
                    optimizer.zero_grad()

                # Calculate loss using the custom loss function
                loss, autoencoder_loss, reconstruct_loss, label_loss, kd_loss, neun_pred_loss, barrier_loss = custom_loss_function(batch, model, device, loss_param, main_task, version)
                # Backward pass
                loss.backward()

                # Update model parameters
                for optimizer in optimizers:
                    optimizer.step()

                # Tracking loss 
                sum_of_loss += torch.Tensor([autoencoder_loss, reconstruct_loss, label_loss, kd_loss, neun_pred_loss, barrier_loss])
                
    return (sum_of_loss / len(dataloader)).tolist()


def eval(test_loaders, model, device, epoch, main_task):
  """
  report 4 accs, label - (two tasks), acc for neun prediction, two unlabel accs
  model: bionet model
  """
  # label accs 
  label_loader, neun_high_loader, neun_low_loader = test_loaders
  pred_results = []
  label_acc = []
  for task_id in [0,1]:
      acc, results = eval_label(label_loader, model, task_id, device)
      pred_results.append(results)
      label_acc.append(acc)

  # neun high 
  unlabel_high_acc = eval_high(neun_high_loader, model, device)
  # neun low 
  unlabel_low_acc = eval_low(neun_low_loader, model, device)
  # neun pred acc
  neun_pred_acc = eval_neun(test_loaders, model, device)
  print("Epoch: {}, Label acc: {}, Neun high acc: {}, Neun low acc:{}, Pred Neun acc: {}".format(epoch, label_acc, unlabel_high_acc, unlabel_low_acc, neun_pred_acc))


  return label_acc, unlabel_high_acc, unlabel_low_acc, neun_pred_acc, pred_results[main_task]


def eval_label(test_loader, model, task_id, device):
  """
  eval for the label data, fix a task
  """
  correct = 0
  total = 0
  results = []

  with torch.no_grad():  # No gradient calculation is needed during evaluation
      for sample in test_loader:
          x = sample['x_continuous'].to(torch.float32).to(device)
          y = sample['label'][:, task_id].to(device)
          # Get model predictions
          _, _, d_B, d_C = model(x)
          outputs = [d_B, d_C]
          outputs = torch.softmax(outputs[task_id], dim=1)
          
          # Apply the condition to determine if the prediction is correct
          pred_correct = torch.logical_or((outputs[:,1] > 0.5) & (y > 0.5), (outputs[:,1] <= 0.5) & (y <= 0.5))
          
          # Update the total number of samples and the number of correct predictions
          total += y.size(0)
          correct += pred_correct.sum().item()

          # Output predicted scores of one task
          pred_results = torch.stack((y, outputs[:,1]))
          results.append(pred_results)

  # Calculate the accuracy
  accuracy = correct / total
  results = torch.cat(results, dim=-1)
  return accuracy, results




def eval_neun(test_loaders, model, device):
    correct = 0
    total = 0

    for test_loader in test_loaders:
        with torch.no_grad():  # No gradient calculation is needed during evaluation
            for sample in test_loader:
                # Get model predictions
                x = sample['x_continuous'].to(torch.float32).to(device)
                y = sample['y_neun'].to(torch.float32).to(device)
                _, d_A, d_B, d_C = model(x)
                outputs = torch.softmax(d_A, dim=1)
                # Apply the condition to determine if the prediction is correct
                pred_correct = torch.logical_or((outputs[:,1] > 0.5) & (y > 0.5), (outputs[:,1] <= 0.5) & (y <= 0.5))
                
                # Update the total number of samples and the number of correct predictions
                total += outputs.size(0)
                correct += pred_correct.sum().item()

    # Calculate the accuracy
    accuracy = correct / total
    return accuracy



def eval_high(test_loader, model, device):
    correct = 0
    total = 0

    with torch.no_grad():  # No gradient calculation is needed during evaluation
        for sample in test_loader:
            # Get model predictions
            x = sample['x_continuous'].to(torch.float32).to(device)
            _, _, d_B, d_C = model(x)
            p_B = torch.softmax(d_B, dim=1)
            p_C = torch.softmax(d_C, dim=1)
            outputs = [p_B, p_C]
            outputs = torch.cat(outputs, dim=1)
            outputs = outputs[:,(1,3)]
            # print("neun high samples ", outputs)
            
            # Apply the condition to determine if the prediction is correct
            pred_correct = (outputs < 0.5).all(dim=1)
            
            # Update the total number of samples and the number of correct predictions
            total += outputs.size(0)
            correct += pred_correct.sum().item()

    # Calculate the accuracy
    accuracy = correct / total
    return accuracy


def eval_low(test_loader, model, device):
    correct = 0
    total = 0

    with torch.no_grad():  # No gradient calculation is needed during evaluation
        for sample in test_loader:
            # Get model predictions
            x = sample['x_continuous'].to(torch.float32).to(device)
            _, _, d_B, d_C = model(x)
            p_B = torch.softmax(d_B, dim=1)
            p_C = torch.softmax(d_C, dim=1)
            outputs = [p_B, p_C]
            outputs = torch.cat(outputs, dim=1)
            outputs = outputs[:,(1,3)]
            # print("neun low samples ", outputs)
            
            # Apply the condition to determine if the prediction is correct
            pred_correct = ((outputs > 0.5).sum(dim=1) == 1) & ((outputs < 0.5).sum(dim=1) == 1)
            
            # Update the total number of samples and the number of correct predictions
            total += outputs.size(0)
            correct += pred_correct.sum().item()

    # Calculate the accuracy
    accuracy = correct / total
    return accuracy


