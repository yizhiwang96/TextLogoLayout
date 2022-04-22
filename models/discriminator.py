import torch
import torch.nn as nn
import torch.nn.functional as F
from models.custom_lstm import CustomLSTM

class SeqDiscriminator(nn.Module):
    def __init__(self, hidden_size, num_hidden_layers):
        super(SeqDiscriminator, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = CustomLSTM(4, hidden_size, True, num_hidden_layers, hidden_size)
        self.fc = nn.Linear(hidden_size, 1, bias=True)
        self.activation = nn.Sigmoid()
    def forward(self, coords, cond, text_len_):
        lstm_output = self.lstm(coords, cond, text_len_)
        # we check the ouput of each seq's last position
        lstm_output_last = torch.gather(lstm_output, 1, text_len_)
        lstm_output_last = lstm_output_last[:, 0, :]
        ret = self.activation(self.fc(lstm_output_last))
        return ret

class ImgDiscriminator(nn.Module):
    def __init__(self, logo_size, logo_channel, hidden_size, ndf=16, norm_layer=nn.BatchNorm2d):
        super(ImgDiscriminator, self).__init__()
        self.logo_size = logo_size
        n_downsampling = 7
        ks_list = [5, 5, 5, 5, 3, 3, 3]
        stride_list = [2, 2, 2, 2, 2, 2, 2]
        encoder_pre = [nn.Conv2d(logo_channel, ndf, kernel_size=7, padding=7 // 2, bias=True),
                    norm_layer(ndf),
                    nn.LeakyReLU(0.2, inplace=True)]
        i = 0
        mult = 1
        encoder_pre += [nn.Conv2d(ndf, ndf * mult * 2, kernel_size=ks_list[i], stride=stride_list[i], padding=ks_list[i] // 2),
                        norm_layer(ndf * mult * 2),
                        nn.LeakyReLU(0.2, inplace=True)]
        self.encode_pre = nn.Sequential(*encoder_pre)

        encoder_last = []
        i = 1
        mult = 2
        encoder_last += [nn.Conv2d(ndf * mult + hidden_size, ndf * mult * 2, kernel_size=ks_list[i], stride=stride_list[i], padding=ks_list[i] // 2),
                    norm_layer(ndf * mult * 2),
                    nn.LeakyReLU(0.2, inplace=True)]
        for i in range(2, n_downsampling):  # adsd downsampling layers
            mult = 2 ** i
            encoder_last += [nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=ks_list[i], stride=stride_list[i], padding=ks_list[i] // 2),
                        norm_layer(ndf * mult * 2),
                        nn.LeakyReLU(0.2, inplace=True)]
        self.encode_last = nn.Sequential(*encoder_last)
        self.fc = nn.Linear(ndf * (2 ** n_downsampling), 1, bias=True)
        self.flatten = nn.Flatten()
        self.activation = nn.Sigmoid()

    def forward(self, imgs, cond):
        # cond : [bact_size, hidden_size]
        ret = self.encode_pre(imgs)
        # insert the condition after the first downsampling conv
        cond = cond.unsqueeze(2)
        cond = cond.unsqueeze(3)
        cond = cond.repeat(1, 1, self.logo_size // 2, self.logo_size // 2)
        ret = torch.cat([ret, cond], 1)
        ret = self.activation(self.fc(self.flatten(self.encode_last(ret))))
        return ret

