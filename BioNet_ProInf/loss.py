import torch
import torch.nn as nn


def simple_autoencoder_loss(input_data, encoder, decoder, **kwargs):
    """
    data: feature
    """
    # Pass the input through the encoder and decoder
    encoded_data = encoder(input_data)
    reconstructed_data = decoder(encoded_data)

    # Calculate the reconstruction loss using the Mean Squared Error (MSE) loss
    loss_function = nn.MSELoss()
    reconstruction_loss = loss_function(input_data, reconstructed_data)
    
    return reconstruction_loss


def layerwise_l1_sparse_autoencoder_loss(data, encoder, decoder, beta=1.0, **kwargs):
    """
    loss for sparse autoencoder
    beta: for l1 reg
    """
    # Forward pass through the encoder
    x = data
    l1_activations = []
    for layer in encoder.layers:
        x = layer(x)
        if isinstance(layer, nn.ReLU):
            l1_activations.append(torch.sum(torch.abs(x)))

    hidden = x
    output = decoder(hidden)

    # Reconstruction loss
    reconstruction_loss = nn.MSELoss()(data, output)

    # Sparsity penalty: L1 loss for activations in the encoder
    sparsity_loss = sum(l1_activations)

    # Total loss
    total_loss = reconstruction_loss + beta * sparsity_loss

    return torch.stack((total_loss, reconstruction_loss))


def denoising_autoencoder_loss(clean_data, encoder, decoder, noise_factor=0.5, **kwargs):
    """
    the noisy autoencoder
    """
    # Corrupt the data by adding noise
    noise = torch.randn_like(clean_data) * noise_factor
    corrupted_data = clean_data + noise

    # Forward pass through the encoder and decoder with corrupted data
    hidden = encoder(corrupted_data)
    output = decoder(hidden)

    # Reconstruction loss: comparing the output to the clean data
    reconstruction_loss = torch.nn.MSELoss()(clean_data, output)

    return reconstruction_loss


def denoising_and_l1_sparse_loss(clean_data, encoder, decoder, noise_factor=0.5, beta=1.0, **kwargs):
    """
    loss for sparse and denoising autoencoder
    beta: for l1 reg
    """
    # Corrupt the data by adding noise
    noise = torch.randn_like(clean_data) * noise_factor
    corrupted_data = clean_data + noise

    # Forward pass through the encoder
    x = corrupted_data
    l1_activations = []
    for layer in encoder.layers:
        x = layer(x)
        if isinstance(layer, nn.ReLU):
            l1_activations.append(torch.sum(torch.abs(x)))

    hidden = x
    output = decoder(hidden)

    # Reconstruction loss
    reconstruction_loss = torch.nn.MSELoss()(clean_data, output)

    # Sparsity penalty: L1 loss for activations in the encoder
    sparsity_loss = sum(l1_activations)

    # Total loss
    total_loss = reconstruction_loss + beta * sparsity_loss

    return torch.stack((total_loss, reconstruction_loss))



def compute_label_loss(data, target_labels, pred, if_label):
    """
    data: feature
    pred: predicted logits from a single tower
    """
    # Apply softmax to the predictions
    softmax_predictions = torch.softmax(pred, dim=1)

    # Use the second entry of the softmax predictions as the final prediction
    final_predictions = softmax_predictions[:, 1]

    # Calculate the loss as the Mean Squared Error between final predictions and target labels
    loss = torch.square(final_predictions - target_labels) * if_label
    loss = 100 * torch.mean(loss)

    return loss

def compute_neun_high_loss(data, preds, if_label, version="new"):
    """
    loss function neun high data
    data: feature
    preds: d_A, d_B, d_C from tower A, B, C
    """
    p_A, p_B, p_C = [torch.softmax(pred, dim=1) for pred in preds]
    outputs = torch.cat([p_B, p_C], dim=1)
    outputs = outputs[:,(1,3)]
    if version == "new":
        loss = 4 * torch.mean(torch.sum(torch.square(outputs), dim=1) * p_A[:,1] * (1.0 - if_label)) 
    else:
        loss = 4 * torch.mean(torch.sum(torch.square(outputs), dim=1) * (1.0 - if_label)) 

    return loss


def compute_neun_low_loss(data, preds, if_label, version="new"):
    """
    loss function neun low data
    data: feature
    preds: d_A, d_B, d_C from tower A, B, C
    """
    p_A, p_B, p_C = [torch.softmax(pred, dim=1) for pred in preds]
    outputs = torch.cat([p_B, p_C], dim=1)
    if version == "new":
        loss = 4 * torch.mean(torch.sum(outputs[:,:1] * outputs[:,2:], dim=1) * p_A[:,0] * (1.0 - if_label)) 
    else:
        loss = 4 * torch.mean(torch.sum(outputs[:,:1] * outputs[:,2:], dim=1) * (1.0 - if_label)) 
    return loss

def compute_knowledge_loss(data, preds, if_label, version):
    """
    knowledge loss 
    preds: d_A, d_B, d_C from tower A, B, C
    """
    return compute_neun_high_loss(data, preds, if_label, version) + compute_neun_low_loss(data, preds, if_label, version)


def compute_neun_pred_loss(data, preds, soft_labels, version="new", epsilon=1e-8):
    """
    prediction loss for neun label
    preds: d_A, d_B, d_C from tower A, B, C
    """
    p_A = torch.softmax(preds[0], dim=1)
    p_A = torch.clamp(p_A, epsilon, 1 - epsilon)
    soft_labels = torch.stack((1.0 - soft_labels, soft_labels), dim=1)
    assert soft_labels.shape[1] == 2
    loss_fn = nn.KLDivLoss(reduction='batchmean', log_target=False)
    if version == "new":
        loss = loss_fn(torch.log(p_A), soft_labels)
    else:
        loss = loss_fn(torch.log(p_A), soft_labels) * 0
    return 6 * loss


autoencoder_loss_funcs = {
  "simple": simple_autoencoder_loss, 
  "sparse": layerwise_l1_sparse_autoencoder_loss, 
  "denoising": denoising_autoencoder_loss,
  "sparse_and_denoising": denoising_and_l1_sparse_loss
}


def log_barrier(data, preds, if_label, threshold=1.0, epsilon=1e-4):
    """
    log barrier to avoid both towers from B and C advocating postive label
    """
    _, p_B, p_C = [torch.softmax(pred, dim=1) for pred in preds]
    barrier_input = torch.clamp(threshold - (p_B[:,1] + p_C[:,1]), epsilon, threshold) 
    barrier_loss = - torch.mean(torch.log(barrier_input))
    return barrier_loss


def loss(data, model, device, loss_param, main_task, version):
    """
    total loss for bionet
    """
    alpha, nu, gamma = loss_param['alpha'], loss_param['nu'], loss_param['gamma']
    barrier_weight, barrier_threshold, barrier_epsilon =  loss_param['barrier_weight'],  loss_param['barrier_threshold'],  loss_param['barrier_epsilon']
    
    if version == "new": 
        # set up sample for computing loss
        x, y, y_neun, if_label = data["x_continuous"], data["label"], data["y_neun"], data["if_label"]
        x = x.to(torch.float32).to(device)
        y = y.to(torch.float32).to(device)
        y_neun = y_neun.to(torch.float32).to(device)
        if_label = if_label.to(torch.float32).to(device)

        # set up loss function for auto encoder 
        autoencoder_loss_func = autoencoder_loss_funcs[loss_param['autoencoder_loss_func']]
        if loss_param['autoencoder_loss_func'] in ['sparse', 'sparse_and_denoising']:
            autoencoder_loss, reconstruct_loss = autoencoder_loss_func(x, model.encoder, model.decoder, **loss_param)
        else:
            autoencoder_loss = autoencoder_loss_func(x, model.encoder, model.decoder, **loss_param)
            reconstruct_loss = autoencoder_loss

        # 1st forward step, compute autoencoder loss
        autoencoder_loss = autoencoder_loss_func(x, model.encoder, model.decoder, **loss_param)
        # 2nd forward step, compute label/knowledge loss
        x_hat, d_A, d_B, d_C = model(x)

        # log barrier loss
        barrier_loss = barrier_weight * log_barrier(data, (d_A, d_B, d_C), if_label, threshold=barrier_threshold, epsilon=barrier_epsilon)

        # label loss
        if main_task == 0:
            label_loss = compute_label_loss(data, y[:,0], d_B, if_label) + 0.1 * compute_label_loss(data, y[:,1], d_C, if_label)
        elif main_task == 1:
            label_loss = 0.1 * compute_label_loss(data, y[:,0], d_B, if_label) + compute_label_loss(data, y[:,1], d_C, if_label)

        # knowledge loss
        kd_loss = compute_knowledge_loss(data, (d_A, d_B, d_C), if_label, version)

        # neun pred loss 
        neun_pred_loss = compute_neun_pred_loss(data, (d_A, d_B, d_C), y_neun, version)
    
    else:
        # set up sample for computing loss
        data_high, data_low = data
        x_h, y_h, y_neun_h, if_label_h = data_high["x_continuous"], data_high["label"], data_high["y_neun"], data_high["if_label"]
        x_h = x_h.to(torch.float32).to(device)
        y_h = y_h.to(torch.float32).to(device)
        y_neun_h = y_neun_h.to(torch.float32).to(device)
        if_label_h = if_label_h.to(torch.float32).to(device)

        x_l, y_l, y_neun_l, if_label_l = data_low["x_continuous"], data_low["label"], data_low["y_neun"], data_low["if_label"]
        x_l = x_l.to(torch.float32).to(device)
        y_l = y_l.to(torch.float32).to(device)
        y_neun_l = y_neun_l.to(torch.float32).to(device)
        if_label_l = if_label_l.to(torch.float32).to(device)

       
        # 2nd forward step, compute label/knowledge loss
        _, d_A_h, d_B_h, d_C_h = model(x_h)
        _, d_A_l, d_B_l, d_C_l = model(x_l)

        kd_loss = compute_neun_low_loss(data_low, (d_A_l, d_B_l, d_C_l), if_label_l, version) + compute_neun_high_loss(data_high, (d_A_h, d_B_h, d_C_h), if_label_h, version)
        
        # label loss
        if main_task == 0:
            label_loss = compute_label_loss(data_high, y_h[:,0], d_B_h, if_label_h) + 0.1 * compute_label_loss(data_high, y_h[:,1], d_C_h, if_label_h)
        elif main_task == 1:
            label_loss = 0.1 * compute_label_loss(data_high, y_h[:,0], d_B_h, if_label_h) + compute_label_loss(data_high, y_h[:,1], d_C_h, if_label_h)

        neun_pred_loss = torch.zeros(1).to(device)
        barrier_loss = torch.zeros(1).to(device)
        autoencoder_loss = torch.zeros(1).to(device)
        reconstruct_loss = torch.zeros(1).to(device)

    # total loss
    loss = (1-alpha-nu-gamma) * label_loss + alpha * autoencoder_loss + nu * kd_loss + gamma * neun_pred_loss + barrier_loss

    return loss, autoencoder_loss.item(), reconstruct_loss.item(), label_loss.item(), kd_loss.item(), neun_pred_loss.item(), barrier_loss.item()



