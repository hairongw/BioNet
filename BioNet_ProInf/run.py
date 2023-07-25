import os 
import numpy as np
import torch 
from learn import *
from networks import *
from loss import *
from train_helper import *
import time
import pickle
import json

def run(
        patient_id,
        data, 
        device,
        patient_dir, 
        params,
      ):
  """
  learning a patient
  freeze_epoch (deprecate): the epoch when we start to freeze autoencoder/or decreasing lr significantly
  patient_dir: results dir for the target patient
  """
  # set up parameters
  training = params["training"]
  loss_param = params["loss_param"]
  freeze_epoch = training["freeze_epoch"]

  # setup models
  encoder, optim_encoder, scheduler_encoder = setup_single_model(params, "encoder", device)
  decoder, optim_decoder, scheduler_decoder = setup_single_model(params, "decoder", device)
  shared, optim_shared, scheduler_shared = setup_single_model(params, "shared", device)
  (tower_A, optim_tower_A, scheduler_tower_A), \
  (tower_B, optim_tower_B, scheduler_tower_B), \
  (tower_C, optim_tower_C, scheduler_tower_C) = setup_single_model(params, "towers", device)
  towers = [tower_A, tower_B, tower_C]
  optim_towers = [optim_tower_A, optim_tower_B, optim_tower_C]
  scheduler_towers = [scheduler_tower_A, scheduler_tower_B, scheduler_tower_C]

  # set up bionet 
  model = BioNetModel(encoder, decoder, shared, towers)

  # set up accuracy tracker
  acc_tracker = []
  
  # evaluate initial model
  print('\n Evaluating random initialization ...')
  label_acc, unlabel_high_acc, unlabel_low_acc, neun_pred_acc, _ = eval(data["test_loaders"], model, device, 0, training["main_task"])
  acc_tracker.append([*label_acc, unlabel_high_acc, unlabel_low_acc, neun_pred_acc])

  print('Begin training ...')
  for epoch in range(1, training["epoch"]+1):
    # update loss parameters
    update_params(params, epoch, freeze_epoch)
    # update lr for autoencoder
    zero_lr([optim_encoder, optim_decoder], epoch, freeze_epoch)

    model.train()
      
    autoencoder_loss, reconstruct_loss, label_loss, kd_loss, neun_pred_loss, barrier_loss = train(data["train_loader"], model,\
                                                                                     (optim_encoder, optim_decoder, optim_shared, *optim_towers), device, loss, loss_param, training["main_task"], training["version"])

    print("Epoch: {}, Autoencoder loss: {}, Reconstruct loss: {}, Label loss: {}, Knowledge loss: {}, Neun prediction loss: {}, Barrier loss: {}".format(epoch, \
                                                                                              autoencoder_loss, reconstruct_loss, label_loss, kd_loss, neun_pred_loss, barrier_loss))


    model.eval()

    # evaluation
    label_acc, unlabel_high_acc, unlabel_low_acc, neun_pred_acc, pred_results = eval(data["test_loaders"], model, device, epoch, training["main_task"])
    acc_tracker.append([*label_acc, unlabel_high_acc, unlabel_low_acc, neun_pred_acc])
    
    # # scheduler step 
    # scheduler_encoder.step(autoencoder_loss)
    # scheduler_decoder.step(autoencoder_loss)
    # for scheduler in scheduler_towers:
    #   scheduler.step(label_acc)

    # # save model 
    # save_model(encoder, patient_id, params["model_info"], "encoder")
    # save_model(decoder, patient_id, params["model_info"], "decoder")

  # post processing after training completes
  pred_results = torch.transpose(pred_results, 0, 1).cpu().numpy()

  acc_tracker = np.array(acc_tracker) 
  info = {
          "accuracy": acc_tracker
        }

  # # create save dir for the run
  # run_dir = "{}/{}/".format(patient_dir, params['model_info']) + time.strftime("%Y-%m-%d %H:%M:%S")
  # os.makedirs(run_dir, exist_ok=True)
  # # save results and params
  # with open('{}/info.pkl'.format(run_dir), 'wb') as f:
  #     pickle.dump(info, f)
  # with open('{}/params.json'.format(run_dir), 'w') as f:
  #     json.dump(params, f)

  return acc_tracker, pred_results