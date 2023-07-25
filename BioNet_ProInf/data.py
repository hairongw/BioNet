from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
from torch.utils.data import DataLoader, ConcatDataset


class BrainTumorDataset(Dataset):
    def __init__(self, csv_file='biopsies69_neighbors_3marker_all5_threshold0.5.csv', if_label=0, neun_label="high"): #biopsies_scaledNeuN --- acc only on biopsies
      """
      csv_file: data file
      handles label (task B,C) and unlabel (task A) data together
      unlabel data, if_label = 0, 
      neun_label = 1 if neun is high else 0
      Note: we read label data, neun low data, neun high data seperately (each fed in as a csv)
      """
      self.patient_id = []
      self.continuous = []
      self.label = [] 
      self.location = []
      self.avid = []

      data = pd.read_csv(csv_file, sep=',')
      # # if exclude out of distribution patients
      # exclude_patients = ['CU1367', 'CU1306', 'CU1275', 'CU1220-2', 'CU1301', 'CU1215', 'CU1220', 'CU1116-2', 'CU1282']
      # data = data[~data['Patient'].isin(exclude_patients)]

      size = data.shape[0]
      self.patient_id = np.array(data.iloc[:,0])
      self.location = np.array(data.iloc[:,2:5])
      if data.shape[1] > 336:
        self.continuous = np.array(data.iloc[:, -336:]) #336
        continuous_feature_idx = list(range(0, 224)) + list(range(280, 336))
        self.continuous_feature = self.continuous[:, continuous_feature_idx]
      else:
        self.continuous_feature = np.array(data.iloc[:, -280:])
         

      if if_label:
        self.label = np.array(data.iloc[:,6:8]) #scaled scores of SOX2 and CD68 6:8 SOX2 and CD68
        self.avid = np.array(data.iloc[:,1]) 
        self.neun_label = np.array(data.iloc[:,5])
      else:
        self.label = np.ones((size, 2), dtype='int64')*(-1)  #two fake labels for unlabeled data
        self.avid = np.zeros((size, 1), dtype='int64') 
        self.neun_label = np.ones((size), dtype='int64') if neun_label == "high" else np.zeros((size), dtype='int64')
  

    def __len__(self):
        return len(self.patient_id)



    def __getitem__(self, idx):
      """
      y1: label for task B 
      y2: label for task C
      """
      y_neun = self.neun_label[idx]
      y1 = self.label[idx,0]
      y2 = self.label[idx,1]

      x_continuous = self.continuous_feature[idx,:]
      x_location = self.location[idx, 0:3]
      # x_patient_id = self.patient_id[idx]
      # x_avid = self.avid[idx]

      sample = {'x_continuous': x_continuous, 'x_location':x_location, 'y_neun': y_neun}

      if y1 < 0 and y2 < 0:
        sample['label'] = np.array([y1, y2])
        sample['if_label'] = 0.0
      else:
        sample['label'] = np.array([y1, y2])
        sample['if_label'] = 1.0


      return sample







######################### data functions ################################

def setup_data(patient_id, params, cohort, status, version):
  # neun unlabel from other patients
  train_neun_low, train_neun_high = create_patient_exclusive_unlabel_data(patient_id, params["seed"], cohort)
  # neun unlabel data from target patient 
  pt_neun_low_train, pt_neun_low_test = create_patient_unlabel_data(patient_id, params["seed"], "low", cohort, status)
  pt_neun_high_train, pt_neun_high_test = create_patient_unlabel_data(patient_id, params["seed"], "high", cohort, status)
  # label data from other patients: train_label - label data from other patients, test_label - label from the target patient
  train_label, test_label = create_label_data(patient_id, params["seed"], cohort)
  # training data: label from others, neun (low and high) from others, part of neun (low and high) from target
  if version == "new":
    train_data = ConcatDataset([train_label, train_neun_low, train_neun_high, pt_neun_low_train, pt_neun_high_train])
    train_loader = DataLoader(train_data, batch_size=params["training"]["batch_size"], shuffle=True)
  else:
    train_data_high = ConcatDataset([train_label, train_neun_high, pt_neun_high_train])
    train_loader_high = DataLoader(train_data_high, batch_size=params["training"]["batch_size"], shuffle=True)
    train_data_low = ConcatDataset([train_neun_low, pt_neun_low_train])
    train_loader_low = DataLoader(train_data_low, batch_size=params["training"]["batch_size"], shuffle=True)    
    train_loader = [train_loader_high, train_loader_low]

  # test data 
  test_label_loader = DataLoader(test_label, batch_size=params["training"]["batch_size"], shuffle=True)
  test_neun_high_loader = DataLoader(pt_neun_high_test, batch_size=params["training"]["batch_size"], shuffle=True)
  test_neun_low_loader = DataLoader(pt_neun_low_test, batch_size=params["training"]["batch_size"], shuffle=True)

  data = {}
  data["train_loader"] = train_loader
  data["test_loaders"] = (
  test_label_loader,
  test_neun_high_loader,
  test_neun_low_loader,
  )
  return data





def train_split_neighbors(file, wdir, mode, seed, k = 0, K = 3):
    # generate train-test split for labeled-data 
    labeled_data = pd.read_csv(file, sep=',')
    n = labeled_data.shape[0]
    # random shuffle data 
    np.random.seed(seed)
    shuffle_idx = np.random.permutation(n)
    labeled_data = labeled_data.iloc[shuffle_idx, :]
    labeled_data.reset_index(drop=True)

    neighbors = pd.read_csv('./RecurrentGBM/biopsies69_neighbors_3marker_all5_threshold0.5.csv', sep=',') 
    
    slices = np.linspace(0, n, K+1, dtype=int)
    lb = slices[k]
    ub = slices[k+1]
    train_idx = list(range(lb)) + list(range(ub, n))

    train_labeled_data = labeled_data
    train_labeled_data = labeled_data.iloc[train_idx, :]   # train by all data if delete this line
    train_labeled_data.to_csv(wdir + 'train_{}_labeled_data.csv'.format(mode), index=False)

    train_labeled_all = labeled_data
    train_labeled_all.to_csv(wdir + 'all_{}_labeled_data.csv'.format(mode), index=False) 

    train_labeled_AVID = train_labeled_data['AVID']
    df = neighbors.merge(train_labeled_AVID, how='inner', on='AVID')
    train_labeled_neighbors = pd.concat([df, train_labeled_data], axis = 0)
    train_labeled_neighbors.to_csv(wdir + 'train_{}_labeled_neighbors.csv'.format(mode), index=False)
        
    all_labeled_AVID = train_labeled_all['AVID']
    df = neighbors.merge(all_labeled_AVID, how='inner', on='AVID')
    all_labeled_neighbors = pd.concat([df, train_labeled_all], axis = 0)
    all_labeled_neighbors.to_csv(wdir + 'all_{}_labeled_neighbors.csv'.format(mode), index=False)


    test_labeled_data = labeled_data.iloc[lb:ub, :]
    test_labeled_data.to_csv(wdir + 'test_{}_labeled_data.csv'.format(mode), index=False)

    test_labeled_AVID = test_labeled_data['AVID']
    df2 = neighbors.merge(test_labeled_AVID, how='inner', on='AVID')
    test_labeled_neighbors = pd.concat([df2, test_labeled_data], axis = 0)
    test_labeled_neighbors.to_csv(wdir + 'test_{}_labeled_neighbors.csv'.format(mode), index=False)
  


def create_patient_exclusive_unlabel_data(patient_id, seed=1, cohort="CU"):
    """
    create unlabel data from all other patients 
    """
    print("creating unlabeled data set (exclusive) for patient id: {}".format(patient_id))
    if cohort == "CU":
      root_dir = "./RecurrentGBM/personalized model SOX2 CD68/" + patient_id + "/"
      neun_low_list = root_dir + "certain_low_neun_7000_no" + patient_id +".csv"
      neun_high_list = root_dir + "certain_high_neun_7000_no" + patient_id +".csv"
    elif cohort == "CED":
      root_dir = "./RecurrentGBM/personalized model SOX2 CD68/"
      neun_low_list = root_dir + "certain_low_neun_7000.csv"
      neun_high_list = root_dir + "certain_high_neun_7000.csv"
    elif cohort == "unlabel":
      root_dir = "./RecurrentGBM/unlabel/"
      neun_low_list = root_dir + "certain_low_neun_7000_train.csv"
      neun_high_list = root_dir + "certain_high_neun_7000_train.csv"
    print("creating neun low data ... ")
    train_neun_low_data = BrainTumorDataset(neun_low_list, if_label=False, neun_label="low")
    print("creating neun high data ... ")
    train_neun_high_data = BrainTumorDataset(neun_high_list, if_label=False, neun_label="high")

    return train_neun_low_data, train_neun_high_data



def create_patient_unlabel_data(patient_id, seed=1, mode="low", cohort="CU", status="test"):
    """
    create unlabel data of the given patient
    """
    print("creating unlabeled data set for patient id: {}".format(patient_id))
    root_dir = "./RecurrentGBM/personalized model SOX2 CD68/" + patient_id + "/"
    if cohort == "CU":
      pt_neun_train = root_dir + "certain_{}_neun_".format(mode) + patient_id + "_train.csv"
      pt_neun_test = root_dir + "certain_{}_neun_".format(mode) + patient_id + "_test.csv"
    elif cohort == "CED":
      pt_neun_train = root_dir + "certain_{}_neun_".format(mode) + patient_id + "_1_train.csv"
      pt_neun_test = root_dir + "certain_{}_neun_".format(mode) + patient_id + "_1_test.csv"
    elif cohort == "unlabel":
       root_unlabel =  "./RecurrentGBM/unlabel/"  
       pt_neun_train = root_unlabel + "certain_{}_neun_".format(mode) + "7000_train.csv"
       if status == "test":
        pt_neun_test = root_unlabel + "certain_{}_neun_".format(mode) + "7000_test.csv"  
       else:
        pt_neun_test = root_unlabel + "certain_{}_neun_".format(mode) + "7000_train.csv" 
       
    print("create neun {} train data for target patient".format(mode))
    pt_neun_train_data = BrainTumorDataset(pt_neun_train, if_label=False, neun_label=mode) 

    print("create neun {} test data for target patient".format(mode))
    pt_neun_test_data = BrainTumorDataset(pt_neun_test, if_label=False, neun_label=mode) 

    return pt_neun_train_data, pt_neun_test_data


def create_label_data(patient_id, seed=1, cohort="CU"):
    """
    create label data for the target patient
    """
    print("creating label data set (exclusive) for patient id: {}".format(patient_id))
    if cohort == "CU" or cohort == "unlabel":
      root_dir = "./RecurrentGBM/personalized model SOX2 CD68/" + patient_id + "/"
      test_labeled_data = BrainTumorDataset(root_dir + 'biopsies69_3marker_5_5_threshold0.5_' + patient_id +'.csv', if_label=True) # this line must be previous than train_split_neighbors
      if not os.path.isfile(root_dir + 'all_neun_labeled_neighbors.csv'): 
        train_split_neighbors(root_dir + 'biopsies69_3marker_5_5_threshold0.5_no'+ patient_id +'.csv', root_dir, "neun", seed) 

    elif cohort == "CED":
      root_dir = "./RecurrentGBM/"
      root_dir_test = "./RecurrentGBM/personalized model SOX2 CD68/" + patient_id + "/"
      test_labeled_data = BrainTumorDataset(root_dir_test + 'resampled_CED_implant_' + patient_id +'_1.csv', if_label=True) # this line must be previous than train_split_neighbors
      if not os.path.isfile(root_dir + 'all_neun_labeled_neighbors.csv'): 
        train_split_neighbors(root_dir + 'biopsies69_3marker_5_5_threshold0.5.csv', root_dir, "neun", seed) 
    
  
    train_labeled_data = BrainTumorDataset(root_dir + 'all_neun_labeled_neighbors.csv', if_label=True) 

    return train_labeled_data, test_labeled_data
