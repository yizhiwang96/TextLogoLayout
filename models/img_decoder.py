import torch
import torch.nn as nn
import torch.nn.functional as F
import functools

class ImgDecoder(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=16, norm_layer=nn.LayerNorm):

        super(ImgDecoder, self).__init__()
        n_upsampling = 6
        ks_list = [3, 3, 5, 5, 5, 5]
        stride_list = [2, 2, 2, 2, 2, 2]
        decoder = []
        mult = 2 ** (n_upsampling)
        decoder += [nn.ConvTranspose2d(input_nc, int(ngf * mult / 2),
                       kernel_size=ks_list[0], stride=stride_list[0],
                       padding=ks_list[0] // 2, output_padding=stride_list[0]-1),
                       norm_layer([int(ngf * mult / 2), 2, 2]),
                       nn.ReLU(True)]
        for i in range(1, n_upsampling):  # add upsampling layers
            mult = 2 ** (n_upsampling - i)
            decoder += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=ks_list[i], stride=stride_list[i], padding=ks_list[i] // 2, output_padding=stride_list[i]-1),
                            norm_layer([int(ngf * mult / 2), 2 ** (i+1) , 2 ** (i+1)]),
                            nn.ReLU(True)]
        decoder += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=7 // 2)]
        decoder += [nn.Sigmoid()]
        self.decode = nn.Sequential(*decoder)


    def forward(self, latent_feat):
        """Standard forward"""

        dec_input = latent_feat
        dec_input = dec_input.view(dec_input.size(0),dec_input.size(1), 1, 1)
        dec_out = self.decode(dec_input)
        return dec_out
