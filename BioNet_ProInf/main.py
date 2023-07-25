import os 
import torch 
import pandas as pd
from run import *
from data import *
from utils import read_params



def main():
  # # get patient list
  # df = pd.read_csv('./RecurrentGBM/biopsies69_3marker_5_5_threshold0.5.csv',sep=',')
  # patients = df["Patient"]
  # p = patients.drop_duplicates().to_list()

  # directory for saving patient-wise result
  save_dir = "./BioNet_results/personalized"
  os.makedirs(save_dir, exist_ok=True)

  # read params 
  params = read_params()

  # set cuda 
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  cohort = params["training"]["cohort"] # "CU" or "CED"
  status = params["training"]["status"]
  version = params["training"]["version"]
  p = [params["training"]["patient"]] # x1, x2, x4, x5, x6

  for i, patient_id in enumerate(p):
      print("\n {} Processng the {}-th patient: {} {}\n".format("*"*48, i+1, patient_id, "*"*48))
      # create data for the target patient
      data = setup_data(patient_id, params, cohort, status, version) # status only used when cohort = "unlabel"
      # training and eval for the target patient
      accs = []
      results = []
      for i in range(20):
        accs_each_run, pred_results_each_run = run(patient_id, data, device, "{}/{}".format(save_dir, patient_id), params)
        accs.append(accs_each_run)
        results.append(pred_results_each_run)
        # save_dir_results = str(cohort) + "_final_results_task_" + str(params["training"]["main_task"])
        # save_dir_acc = save_dir_results + "/accs_all/"
        # nu = params["loss_param"]["nu"]
        # bar = params["loss_param"]["barrier_weight"]
        # os.makedirs("{}".format(save_dir_results), exist_ok=True)
        # os.makedirs("{}".format(save_dir_acc), exist_ok=True)
        # np.save("./{}/results_{}_task_{}.npy".format(save_dir_results, patient_id, params["training"]["main_task"]), np.array(results))
        # np.save("./{}/accs_{}_nu_{}_bar_{}.npy".format(save_dir_acc, patient_id, nu, bar), np.array(accs))

        save_dir_acc = str(cohort)
        os.makedirs("{}".format(save_dir_acc), exist_ok=True)
        np.save("./{}/accs_{}_{}.npy".format(save_dir_acc, version, status), np.array(accs))




        # acc_save_dir = 'test_nu_0.4_CU1249'
        # os.makedirs("{}".format(acc_save_dir), exist_ok=True)
        # np.save("./{}/accs_{}.npy".format(acc_save_dir, patient_id), np.array(accs))
      return 


if __name__ == '__main__':
    main()