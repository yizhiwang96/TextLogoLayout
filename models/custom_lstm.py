

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pack_sequence,pad_packed_sequence
from models.util_funcs import sequence_mask
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first, num_hidden_layers, init_cond_dim):
        super(CustomLSTM, self).__init__()
        self.num_hidden_layers = num_hidden_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_hidden_layers, batch_first=batch_first)
        self.unbottleneck_dim = hidden_size * 2
        self.unbottlenecks = nn.ModuleList([nn.Linear(init_cond_dim, self.unbottleneck_dim) for _ in range(self.num_hidden_layers)])
    
    def forward(self, inputs, initial_condition, seq_lens):
        # a vanilla rnn encoder that encoding a sequence
        # inputs: [batch_size, max_seqlen, feat_dim]

        # first prepare the init hidden and cell
        init_state_hidden = []
        init_state_cell = []
        for i in range(self.num_hidden_layers):
            unbottleneck = self.unbottlenecks[i](initial_condition)
            (h0, c0) = unbottleneck[:, :self.unbottleneck_dim // 2], unbottleneck[:, self.unbottleneck_dim // 2:]
            init_state_hidden.append(h0.unsqueeze(0))
            init_state_cell.append(c0.unsqueeze(0))
        init_state_hidden = torch.cat(init_state_hidden, dim=0)
        init_state_cell = torch.cat(init_state_cell, dim=0)

        # then run the lstm encoder
        # self.lstm.flatten_parameters()
        ret, (hidden, cell) = self.lstm(inputs, (init_state_hidden, init_state_cell))
        return ret