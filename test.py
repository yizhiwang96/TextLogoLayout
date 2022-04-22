import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import Adam
from torchvision.utils import save_image
from tensorboardX import SummaryWriter
from options import get_parser
from dataloader import get_loader
from models.condition_encoder import ConditionEncoder
from models.coord_generator import CoordGenerator
from models.diff_composition import DiffComposition
from models.discriminator import ImgDiscriminator, SeqDiscriminator
from models.util_funcs import cal_overlap_loss
from pytorch_fid.fid_score import calculate_fid_given_paths
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test(opts):
    test_dataloader = get_loader(os.path.join(opts.data_root, opts.data_name), opts.batch_size, mode='test')

    exp_dir = os.path.join("experiments", opts.experiment_name)
    ckpt_dir = os.path.join(exp_dir, "checkpoints")

    # prepare the models
    condition_encoder = ConditionEncoder(opts.input_size, opts.hidden_size, opts.max_seqlen, opts.embed_dim, opts.num_hidden_layers, opts.hidden_size, opts.use_embed_word, opts.cond_modality)
    coord_generator = CoordGenerator(opts.hidden_size, opts.num_hidden_layers, opts.latent_dim)
    diff_composition = DiffComposition(opts.input_size, opts.output_size, opts.max_seqlen, opts.pos_format, opts.align_corners, opts.trunc_logo_pix)
    seq_discriminator = SeqDiscriminator(opts.hidden_size, opts.num_hidden_layers)
    img_discriminator = ImgDiscriminator(opts.output_size, opts.output_channel, opts.hidden_size, ndf=8)

    modules_dict = {'condition_encoder':condition_encoder, 'coord_generator':coord_generator, 'img_discriminator':img_discriminator, 'seq_discriminator':seq_discriminator}
    # load parameters
    for md_name, md in modules_dict.items():
        md_fpath = os.path.join(ckpt_dir, f"epoch_{opts.test_epoch}_" + md_name+ ".pth")
        md.load_state_dict(torch.load(md_fpath))
        md.eval()
        md = md.to(device)

    if torch.cuda.is_available() and opts.multi_gpu:
        condition_encoder = nn.DataParallel(condition_encoder)
        coord_generator = nn.DataParallel(coord_generator)
        seq_discriminator = nn.DataParallel(seq_discriminator)
        img_discriminator = nn.DataParallel(img_discriminator)
    
    modules = [condition_encoder, coord_generator, diff_composition, seq_discriminator, img_discriminator]
    # do testing ...
    print("sampling for testing data")
    res_dir = os.path.join("experiments", opts.experiment_name, "results", "%04d"%opts.test_epoch)
    os.makedirs(res_dir, exist_ok=True)
    if not os.path.exists(os.path.join(res_dir, opts.data_name)):
        os.makedirs(os.path.join(res_dir, opts.data_name))

    for idx_test, data_test in enumerate(test_dataloader):
        for sample_id in range(opts.test_sample_times):
            imgs_list_test, _, _, _, batch_size_test_, _ = network_forward(data_test, modules, opts)
            imgs_fake_list_test, _, _, _ = imgs_list_test
            _, imgs_logo_fake_test, _, _, _ = imgs_fake_list_test

            for idx_img in range(batch_size_test_):
                # sample_fpath = os.path.join(res_dir, "%04d"%(idx_test * opts.batch_size + idx_img) + "_%02d"%sample_id +'.png')
                # save_image(imgs_logo_fake_test[idx_img, :], sample_fpath, 'png')
                sample_fpath = os.path.join(res_dir, opts.data_name, "%04d"%(idx_test * opts.batch_size + idx_img) + "_%02d"%sample_id +'_invert.png')
                save_image(1.0 - imgs_logo_fake_test[idx_img, :], sample_fpath, 'png')
    
    '''
    # testing FID metric if needed
    print("calculating FID")
    path_fake = os.listdir(res_dir, opts.data_root, opts.data_name)
    for idx_ in range(len(path_fake)):
        path_fake[idx_] = os.path.join(res_dir, path_fake[idx_])
    path_gt = os.listdir(os.path.join(opts.data_root, opts.data_name, 'train'))
    for idx_ in range(len(path_gt)):
        path_gt[idx_] = os.path.join(opts.data_root, opts.data_name, 'train', path_gt[idx_], 'logo_resized.png')
    fid_value = calculate_fid_given_paths(paths=[path_fake, path_gt], batch_size=opts.batch_size, device=device, dims=2048,)
    print("FID value: %04f"%fid_value)
    '''

def network_forward(data, modules, opts):
    condition_encoder, coord_generator, diff_composition, seq_discriminator, img_discriminator = modules
    # prepare the data
    imgs_glyph = data['imgs_glyph'].to(device) # [batch_size, input_size, input_size * max_seqlen])
    imgs_logo_native = data['imgs_logo'].to(device).unsqueeze(1)
    if opts.pos_format == 'whxy':
        coords_gt_ = data['coords_gt_centre'].to(device) # default format: left, top, right, bottom, trans to xc, yc, w, h
        coords_xc = (coords_gt_[:, :, 0:1] + coords_gt_[:, :, 2:3]) / 2.0
        coords_yc = (coords_gt_[:, :, 1:2] + coords_gt_[:, :, 3:4]) / 2.0
        coords_w = coords_gt_[:, :, 2:3] - coords_gt_[:, :, 0:1]
        coords_h = coords_gt_[:, :, 3:4] - coords_gt_[:, :, 1:2]
        coords_gt_ = torch.cat([coords_xc, coords_yc, coords_w, coords_h], -1)        
    else:
        coords_gt_ = data['coords_gt_centre'].to(device) # format: left, top, right, bottom

    coords_gt = coords_gt_ / opts.output_size

    # coords_gt = coords_gt.permute(1, 0, 2) # B x L x C -> L x B x C
    text_len = data['text_len'].to(device) # [batch_size, 1]
    embeds_char = data['embeds_char'].to(device)
    embeds_word = data['embeds_word'].to(device)
    batch_size_ = imgs_glyph.shape[0]

    # network forward
    condition_features, condition_feat_last, text_len_, loss_rec = condition_encoder(imgs_glyph, embeds_char, embeds_word, text_len) # [max_seqlen, batch_size, hidden_size]

    noise = torch.randn(batch_size_, opts.latent_dim).to(device)
    coords_fake = coord_generator(condition_features, noise, text_len_)

    # imgs_logo_fake_ss, imgs_logo_fake_ss_tc, imgs_logo_fake_ms, imgs_logo_fake_ms_tc, imgs_trans_fake = diff_composition(imgs_glyph, coords_fake, text_len)
    # imgs_logo_real_ss, imgs_logo_real_ss_tc, imgs_logo_real_ms, imgs_logo_real_ms_tc, imgs_trans_real = diff_composition(imgs_glyph, coords_gt, text_len)
    imgs_fake_list = diff_composition(imgs_glyph, coords_fake, text_len)
    imgs_real_list = diff_composition(imgs_glyph, coords_gt, text_len)
    imgs_list = [imgs_fake_list, imgs_real_list, imgs_logo_native, imgs_glyph]
    coords_list = [coords_gt, coords_fake]
    return imgs_list, coords_list, condition_feat_last, text_len_, batch_size_, loss_rec

def main():
    opts = get_parser().parse_args()
    opts.experiment_name = opts.experiment_name

    print(f"Testing on experiment {opts.experiment_name}...")
    test(opts)

if __name__ == "__main__":
    main()