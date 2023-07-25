import torch
import torch.nn as nn
import torch.optim as optim

class SimpleMTL(nn.Module):
  def __init__(self, input_size, output_sizes, hidden_sizes):
    super(SimpleMTL, self).__init__()
    self.hidden_sizes = hidden_sizes.copy()
    self.hidden_sizes.insert(0, input_size)
    self.towers = [] 
    self.tasks = len(output_sizes)
    for output_size in output_sizes:
      tower = []
      for j in range(len(self.hidden_sizes)-1):
          tower.append(nn.Dropout(p=1e-1)) 
          tower.append(nn.Linear(self.hidden_sizes[j], self.hidden_sizes[j+1]))
          tower.append(nn.ReLU())
      tower.append(nn.Linear(self.hidden_sizes[-1], output_size))
      self.towers.append(nn.Sequential(*tower))
    self.towers = nn.ModuleList(self.towers)
    
  def predict(self, x):
    with torch.no_grad():
      y = []
      logits = self.forward(x)
      for task in range(self.tasks):
        logit = logits[task]

        ytask = torch.zeros(len(logit))
        for i in range(len(logit)):
          _, ytask[i] = torch.max(logit[i],0)

        y.append(ytask)
      return y


  def forward(self, x):
    outputs = []
    for tower in self.towers:
      outputs.append(tower(x))

    return outputs 