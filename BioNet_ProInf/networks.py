import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau


class TrivialEncoder(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(TrivialEncoder, self).__init__()
        # place holder layer does nothing 
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(out_features, out_features)

        if in_features != out_features:
            self.shortcut = nn.Sequential(
                nn.Linear(in_features, out_features),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = x
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out += self.shortcut(residual)
        out = self.relu(out)
        return out

class FeedForwardResNet(nn.Module):
    def __init__(self, input_size, block_sizes, output_size, **kwargs):
        super(FeedForwardResNet, self).__init__()

        self.linear1 = nn.Linear(input_size, block_sizes[0])
        self.relu = nn.ReLU()

        self.blocks = nn.ModuleList()
        for i in range(len(block_sizes) - 1):
            self.blocks.append(ResidualBlock(block_sizes[i], block_sizes[i + 1]))

        self.fc = nn.Linear(block_sizes[-1], output_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)

        for block in self.blocks:
            x = block(x)

        x = self.fc(x)
        return x


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.0):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(nn.Dropout(dropout_rate))
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        self.layers.append(nn.ReLU())

        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(nn.Dropout(dropout_rate))
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            self.layers.append(nn.ReLU())

        # Output layer
        self.layers.append(nn.Dropout(dropout_rate))
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.0):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(nn.Dropout(dropout_rate))
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        self.layers.append(nn.ReLU())

        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            self.layers.append(nn.Dropout(dropout_rate))
            self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            self.layers.append(nn.ReLU())

        # Output layer
        self.layers.append(nn.Dropout(dropout_rate))
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))

        # normalize output
        self.tanh = nn.Tanh()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        # normalize output into certain range
        # x = self.tanh(x)
        return x
    


class CustomNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.0, **kwargs):
        super(CustomNN, self).__init__()
        self.layers = nn.ModuleList()

        if hidden_sizes:
            # Input layer
            self.layers.append(nn.Dropout(dropout_rate))
            self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
            self.layers.append(nn.ReLU())

            # Hidden layers
            for i in range(len(hidden_sizes) - 1):
                self.layers.append(nn.Dropout(dropout_rate))
                self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
                self.layers.append(nn.ReLU())

            # Output layer
            self.layers.append(nn.Dropout(dropout_rate))
            self.layers.append(nn.Linear(hidden_sizes[-1], output_size))
        # linear network
        else:
            self.layers.append(nn.Linear(input_size, output_size))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x




class BioNetModel(nn.Module):
    def __init__(self, encoder, decoder, shared, towers):
        super(BioNetModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.shared = shared 
        self.tower_A, self.tower_B, self.tower_C = towers

    def forward(self, x):
        z = self.encoder(x)
        # reconstruction
        x_hat = self.decoder(z)
        # shared layer for classifications
        shared_input = self.shared(z)
        # towers
        d_A = self.tower_A(shared_input)
        d_B = self.tower_B(shared_input)
        d_C = self.tower_C(shared_input)

        return x_hat, d_A, d_B, d_C


########################### setup models ######################



def setup_scheduler(optimizer, factor=0.5, patience=8):
  """
  set up schedulers for optimizers
  """
  scheduler = []
  return ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience, verbose=True)


def setup_single_model(params, name, device):
    """
    set up models and optimizers
    name: encoder/decoder/classifier
    """
    # trivial models
    TRIVIAL_MODELS = [
    "linear_wo_unlabel", 
    "linear_with_unlabel",
    "resnet_wo_unlabel", 
    "resnet_with_unlabel"
    ]

    arch, loss_param, lrs = params["arch"], params["loss_param"], params["training"]["lrs"]
    if name == "encoder":
        encoder_params = arch["encoder"][loss_param["autoencoder_loss_func"]]
        if params["model_info"] in TRIVIAL_MODELS:
            print("Setting up trivial encoder for model info: {}".format(params["model_info"]))
            encoder = TrivialEncoder(**encoder_params).to(device)
        else:
            encoder = Encoder(**encoder_params).to(device)
        # if arch["reuse_autoencoder"]:
        #     load_model(encoder, patient_id, "encoder")
        optim_encoder = optim.Adam(encoder.parameters(), lr=lrs["encoder"])
        scheduler_encoder = setup_scheduler(optim_encoder, factor=0.5, patience=8)
        return encoder, optim_encoder, scheduler_encoder

    if name == "decoder":
        decoder_params = arch["decoder"][loss_param["autoencoder_loss_func"]]
        decoder = Decoder(**decoder_params).to(device)
        optim_decoder = optim.Adam(decoder.parameters(), lr=lrs["decoder"])
        # if arch["reuse_autoencoder"]:
        #     load_model(decoder, patient_id, "decoder")
        optim_decoder = optim.Adam(decoder.parameters(), lr=lrs["decoder"])
        scheduler_decoder = setup_scheduler(optim_decoder, factor=0.5, patience=8)
        return decoder, optim_decoder, scheduler_decoder

    if name == "towers":
        tower_params = arch["towers"]
        towers_specs = []
        for tower_name, tower_param in tower_params.items():
            if tower_param["use_resnet"]:
                tower = FeedForwardResNet(**tower_param).to(device)
            else:
                tower = CustomNN(**tower_param).to(device)
            optim_tower = optim.Adam(tower.parameters(), lr=lrs["towers"][tower_name])
            scheduler_tower = setup_scheduler(optim_tower, factor=0.5, patience=2)
            towers_specs.append((tower, optim_tower, scheduler_tower))
        return towers_specs

    if name == 'shared':
        shared_params = arch["shared"]
        if shared_params["use_resnet"]:
            shared = FeedForwardResNet(**shared_params).to(device)
        else:
            shared = CustomNN(**shared_params).to(device)
        optim_shared = optim.Adam(shared.parameters(), lr=lrs["shared"])
        scheduler_shared = setup_scheduler(optim_shared, factor=0.5, patience=8)
        return shared, optim_shared, scheduler_shared

    assert 0 < -1