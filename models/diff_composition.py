import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Resize
from models.util_funcs import sequence_mask
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DiffComposition(nn.Module):

    def __init__(self, input_size, output_size, max_seqlen, pos_format, align_corners, trunc_logo_pix):
        """
        Note: DiffComposition has not trainable parameters
        """
        super(DiffComposition, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.max_seqlen = max_seqlen
        self.relu = nn.ReLU()
        self.glyph_resizer = Resize([self.output_size, self.output_size])
        self.pos_format = pos_format
        self.align_corners = align_corners
        self.trunc_logo_pix = trunc_logo_pix

    def forward(self, imgs, coords, seq_len):
        """
        Parameters:
        imgs: [self.batch_size, self.input_size, self.max_seqlen * self.input_size]
        coords: [self.batch_size, self.max_seqlen, 4]
        seq_len: [self.batch_size, 1]
        """
        # prepare the coordinates: (0, 1) -> (0, output_size)
        batch_size_ = imgs.shape[0]
        # coords = coords.permute(1, 0, 2) 
        coords = coords * self.output_size
        coords = coords.reshape(batch_size_ * self.max_seqlen, 4)

        if self.pos_format == 'ltrb':
            left = coords[:, 0:1]
            top = coords[:, 1:2]
            right = coords[:, 2:3]
            bottom = coords[:, 3:4]
        else:
            xc = coords[:, 0:1]
            yc = coords[:, 1:2]
            w = coords[:, 2:3]
            h = coords[:, 3:4]            

        # reshape the glyph images
        imgs_rs = imgs.reshape(batch_size_, self.input_size, self.max_seqlen, self.input_size)
        imgs_rs = imgs_rs.permute(0, 2, 1, 3)
        imgs_rs = imgs_rs.reshape(batch_size_ * self.max_seqlen, self.input_size, self.input_size)
        imgs_rs = imgs_rs.unsqueeze(1)
        imgs_rs = imgs_rs.repeat(1, 3, 1, 1) #[batch_size * max_seqlen, 3, input_size, input_size])
        trg_size = (batch_size_ * self.max_seqlen, 3, self.output_size, self.output_size)
        imgs_rs = self.glyph_resizer(imgs_rs)

        # calculate the transformation matrix
        eps = 1e-8 # for stabilizing the values
        # note that the trans_param is different from the equation in our paper because the actual implemention of STN (affine_grid) in pytorch
        if self.pos_format == 'ltrb':
            trans_param = torch.cat([self.output_size / (right - left + eps), left * 0.0,  (self.output_size - left - right) / (right - left + eps), top * 0.0, self.output_size / (bottom - top + eps), (self.output_size - top - bottom) / (bottom - top + eps)], -1)
        else:
            trans_param = torch.cat([self.output_size / (w + eps), xc * 0.0,  2 * (self.input_size - xc) / (w + eps), h * 0.0, self.output_size / (h + eps), 2 * (self.input_size - yc) / (h + eps)], -1)
        
        trans_param = trans_param.view(-1, 2, 3)

        grid = F.affine_grid(trans_param, trg_size, align_corners=self.align_corners)
        imgs_trans = F.grid_sample(imgs_rs, grid, align_corners=self.align_corners)
        imgs_trans = imgs_trans.reshape(batch_size_, self.max_seqlen, 3, self.output_size, self.output_size)
        imgs_trans = imgs_trans.permute(0, 2, 3, 4, 1)

        # prepare the seq mask to mask the useless imgs (positions > seq_len)
        seq_mask = sequence_mask(seq_len, max_len=self.max_seqlen).float()
        seq_mask = seq_mask.unsqueeze(2)
        seq_mask = seq_mask.unsqueeze(3)
        seq_mask = seq_mask.expand(-1, 3, self.output_size, self.output_size, -1)

        # merge the transformed glyphs (by adding)
        imgs_trans = self.relu(imgs_trans)

        # imgs_res_ss: all glyphs have the same pixel value (white)
        imgs_trans_ss = imgs_trans * seq_mask
        imgs_res_ss = torch.sum(imgs_trans_ss, dim=-1)

        # imgs_res_ms: different glyphs have different pixel values
        # define the grey scales for different glyphs
        colormask = (torch.arange(5, 5 + self.max_seqlen) * 10).to(device) / 255. # (50, 60, ..., 240) / 255.
        colormask = colormask.reshape(1, 1, 1, 1, self.max_seqlen)
        colormask = colormask.repeat(batch_size_, 3, self.output_size, self.output_size, 1)

        imgs_res_ms = imgs_trans * colormask * seq_mask
        imgs_res_ms = torch.sum(imgs_res_ms, dim=-1)

        # truncate the values > 1 (pixel value 255)
        imgs_res_ss_tc = 1 - self.relu(1 - imgs_res_ss) 
        imgs_res_ms_tc = 1 - self.relu(1 - imgs_res_ms) 

        return imgs_res_ss, imgs_res_ss_tc, imgs_res_ms, imgs_res_ms_tc, imgs_trans
