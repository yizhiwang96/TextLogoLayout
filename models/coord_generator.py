import torch
import torch.nn as nn
import torch.nn.functional as F
from models.custom_lstm import CustomLSTM

class CoordGenerator(nn.Module):

    def __init__(self, hidden_size, num_hidden_layers, latent_dim):
        super(CoordGenerator, self).__init__()
        self.hidden_size = hidden_size
        self.coord_lstm_encoder = CustomLSTM(hidden_size, hidden_size, True, num_hidden_layers, latent_dim)
        self.coord_lstm_decoder = CustomLSTM(hidden_size, hidden_size, True, num_hidden_layers, hidden_size)
        self.coord_proj = nn.Sequential(*[nn.Linear(hidden_size, 4), nn.Sigmoid()])

    def forward(self, cond_feat, noise, text_len_):
        # batch_size_ = cond_feat.shape[1]
        # run rnn encoder
        mid_feat = self.coord_lstm_encoder(cond_feat, noise, None)

        # the last state in mid_feat
        mid_feat_last = torch.gather(mid_feat, 1, text_len_)
        mid_feat_last = mid_feat_last[:, 0, :]

        # runn rnn decoder to pred coordinates
        output_feat = self.coord_lstm_decoder(mid_feat, mid_feat_last, None)
        coords_pred = self.coord_proj(output_feat)
        return coords_pred