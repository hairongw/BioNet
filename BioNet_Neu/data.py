from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os

# Define custom dataset
class BrainTumorDataset(Dataset):
    def __init__(self, file_list=['Copy of virtual_biop(5000+5000).csv'], weight_display=False): 
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
          self.continuous.append(np.array(data.iloc[:, -336:])) #336

          self.label.append(np.array(data['NeuN_set']))
          self.avid.append(np.array(data.iloc[:,1]))

        self.patient_id = np.concatenate(self.patient_id)
        self.location = np.concatenate(self.location)
        self.continuous = np.concatenate(self.continuous)
        self.label = np.concatenate(self.label)
        self.avid = np.concatenate(self.avid)

        num_A0 = sum(self.label[:]<0.5) 
        num_A1 = sum(self.label[:]>=0.5)
        total_size = self.label.shape[0]
        print(num_A0, total_size)

        self.percentage = [num_A0/total_size, num_A1/total_size]
        print("NeuN, percentage {:4f}(y1=0), percentage {:4f}(y1=1)".format(num_A0/total_size, num_A1/total_size))




    def __len__(self):
        return len(self.patient_id)

    def __getitem__(self, idx):
        y = self.label[idx]
        col_idx = [i for i in range(0, 224)] + [i for i in range(280, 336)]
        col_idx_all = [i for i in range(0, 336)]
        x = self.continuous[:,col_idx]
        x_all = self.continuous[:,col_idx_all]
        x_continuous = x[idx,:]
        x_allfeature = x_all[idx,:]

        x_location = self.location[idx, 0:3]
        x_patient_id = self.patient_id[idx]
        x_avid = self.avid[idx]
        sample = {'y': y, 'x_continuous':x_continuous, 'x_all_features':x_allfeature, 'x_location':x_location, 'patient_id':x_patient_id, 'AVID':x_avid}
        return sample


def train_split(file, k, K, seed):
  labeled_data = pd.read_csv(file, sep=',')
  n = labeled_data.shape[0]
  np.random.seed(seed)
  shuffle_idx = np.random.permutation(n)
  labeled_data = labeled_data.iloc[shuffle_idx, :]
  labeled_data.reset_index(drop=True)

  if K == 1:
    train_labeled_data = labeled_data
    train_labeled_data = labeled_data.iloc[:2, :]   
    train_labeled_data.to_csv('train_labeled_data.csv', index=False)

    test_labeled_data = labeled_data.iloc[2:, :]
    test_labeled_data.to_csv('test_labeled_data.csv', index=False) 

  else:
    slices = np.linspace(0, n, K+1, dtype=int)
    lb = slices[k]
    ub = slices[k+1]
    train_idx = list(range(lb)) + list(range(ub, n))

    train_labeled_data = labeled_data
    train_labeled_data = labeled_data.iloc[train_idx, :] 
    train_labeled_data.to_csv('train_labeled_data.csv', index=False)

    test_labeled_data = labeled_data.iloc[lb:ub, :]
    test_labeled_data.to_csv('test_labeled_data.csv', index=False)


def train_split_neighbors(file, k, K, seed):
  labeled_data = pd.read_csv(file, sep=',')
  n = labeled_data.shape[0]
  np.random.seed(seed)
  shuffle_idx = np.random.permutation(n)
  labeled_data = labeled_data.iloc[shuffle_idx, :]
  labeled_data.reset_index(drop=True)

  os.chdir(r"/content/drive/MyDrive/RecurrentGBM_soft_classification_all_features/") 
  cwd = os.getcwd()
  neighbors = pd.read_csv('biopsies63_neighbors_all5_threshold0.5.csv', sep=',') 


  slices = np.linspace(0, n, K+1, dtype=int)
  lb = slices[k]
  ub = slices[k+1]
  train_idx = list(range(lb)) + list(range(ub, n))

  train_labeled_data = labeled_data
  train_labeled_data = labeled_data.iloc[train_idx, :]  
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


