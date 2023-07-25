import os 
import torch


def update_params(params, epoch, freeze_epoch):
    """
    update parameters based on epoch
    freeze_epoch (>=1): until which we only learn autoencoder, after which we reset alpha to the one specifed in param files 
    """
    loss_params = params["loss_param"]
    # pretrain_ratio = 0.0
    # # save the default params read from param file, override params temporarily
    # if epoch == 1:
    #   loss_params["default_alpha"] = loss_params["alpha"]
    #   loss_params["default_nu"] = loss_params["nu"]
    #   loss_params["default_gamma"] = loss_params["gamma"]
  
    # if epoch <= int(pretrain_ratio * params["training"]["epoch"]):
    #     loss_params["nu"] = 0.0
    #     loss_params["gamma"] = 1.0
    #     loss_params["alpha"] = 0.0 

    # if epoch == int(pretrain_ratio * params["training"]["epoch"]) + 1:
    #     loss_params["alpha"] = loss_params["default_alpha"]
    #     loss_params["nu"] = loss_params["default_nu"]
    #     loss_params["gamma"] = loss_params["default_gamma"]
    #     print("="*64)
    #     print("{} Start training with label data {}".format("="*24, "="*24))
    #     print("="*64)

    print("Epoch: {}, Loss params: {} \n".format(epoch, loss_params))


def zero_lr(optimizers, epoch, freeze_epoch):
  pass 
  # if epoch > freeze_epoch:
  #   for optimizer in optimizers:
  #     for param_group in optimizer.param_groups:
  #         param_group['lr'] = 0


def save_model(model, patient_id, model_info, model_name):
    """
    save model to file
    """
    os.makedirs("./model_checkpoints/{}/{}".format(patient_id, model_info), exist_ok=True)
    save_dir = './model_checkpoints/{}/{}/{}.pth'.format(patient_id, model_info, model_name)
    torch.save(model.state_dict(), save_dir)
    print("saving {} at {}".format(model_name, save_dir))

def load_model(model, patient_id, model_info, model_name):
  """
  save model to file
  """
  try:
    save_dir = './model_checkpoints/{}/{}/{}.pth'.format(patient_id, model_info, model_name)
    saved_state_dict = torch.load(save_dir)
    model.load_state_dict(saved_state_dict)
    print("loaded {} at {}".format(model_name, save_dir))
  except:
    return 
