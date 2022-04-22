import torch
import torch.nn as nn
import torch.nn.functional as F
from models.img_encoder import ImgEncoder
from models.img_decoder import ImgDecoder
from models.custom_lstm import CustomLSTM
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConditionEncoder(nn.Module):
    def __init__(self, glyph_size, hidden_size, max_seqlen, embed_dim, num_hidden_layers, init_cond_dim, use_embed_word, cond_modality):
        super(ConditionEncoder, self).__init__()
        self.glyph_size = glyph_size
        self.img_encoder = ImgEncoder()
        self.img_decoder = ImgDecoder(512, 3)
        self.cond_modality = cond_modality
        self.use_embed_word = use_embed_word
        self.hidden_size = hidden_size
        if self.use_embed_word:
            self.text_input_size = 2 * embed_dim
        else:
            self.text_input_size = embed_dim
        if self.cond_modality == 'img_text':
            self.lstm_input_size = self.text_input_size + 512
        elif self.cond_modality == 'img':
            self.lstm_input_size = 512
        else:
             self.lstm_input_size = self.text_input_size
        self.cond_lstm_encoder = CustomLSTM(self.lstm_input_size, hidden_size, True, num_hidden_layers, init_cond_dim) # 512 is the feature dim of VGG19 conv5
        self.embed_dim = embed_dim
        self.max_seqlen = max_seqlen
        self.init_cond_dim = init_cond_dim
        self.loss_rec = nn.L1Loss()
    
    def forward(self, imgs_glyph, embeds_char, embeds_word, text_len):
        # reshape the glyph imgs
        batch_size_ = imgs_glyph.shape[0]
        imgs_glyph_rs = imgs_glyph.reshape(batch_size_, self.glyph_size, self.max_seqlen, self.glyph_size)
        imgs_glyph_rs = imgs_glyph_rs.permute(0, 2, 1, 3)
        imgs_glyph_rs = imgs_glyph_rs.reshape(batch_size_ * self.max_seqlen, self.glyph_size, self.glyph_size)
        imgs_glyph_rs = imgs_glyph_rs.unsqueeze(1)
        imgs_glyph_rs = imgs_glyph_rs.repeat(1, 3, 1, 1) #[batch_size * max_seqlen, 3, input_size, input_size])
        
        imgs_feat = self.img_encoder(imgs_glyph_rs) # [batch_size * max_seqlen, 512])
        imgs_glyph_rc = self.img_decoder(imgs_feat)
        rec_res = {}
        rec_res['img'] = imgs_glyph_rc
        rec_res['l1_loss'] = self.loss_rec(imgs_glyph_rs, imgs_glyph_rc)

        imgs_feat = imgs_feat.reshape(batch_size_, self.max_seqlen, 512) # 512 is the feature dim of VGG19 conv5

        if self.use_embed_word:
            embeds_text = torch.cat([embeds_char, embeds_word], -1)
        else:
            embeds_text = embeds_char
        
        if self.cond_modality == 'img_text':
            lstm_input_feat = torch.cat([imgs_feat, embeds_text], -1)
        elif self.cond_modality == 'img':
            lstm_input_feat = imgs_feat
        else:
            lstm_input_feat = embeds_text
        
        init_cond = torch.zeros(batch_size_, self.init_cond_dim).to(device)
        # self.cond_lstm_encoder.get_init_state(init_cond)
        cond_feat = self.cond_lstm_encoder(lstm_input_feat, init_cond, text_len)

        text_len_ = (text_len - 1).unsqueeze(1)
        text_len_ = text_len_.expand(batch_size_, 1, self.hidden_size)
        cond_feat_last = torch.gather(cond_feat, 1, text_len_)
        cond_feat_last = cond_feat_last[:, 0, :]

        return cond_feat, cond_feat_last, text_len_, rec_res