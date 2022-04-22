from calendar import c
import os
import shutil
import copy
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

def train(opts):
    train_dataloader = get_loader(os.path.join(opts.data_root, opts.data_name), opts.batch_size, mode=opts.mode)
    test_dataloader = get_loader(os.path.join(opts.data_root, opts.data_name), opts.batch_size, mode='test')

    exp_dir = os.path.join("experiments", opts.experiment_name)
    sample_dir = os.path.join(exp_dir, "samples")
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    log_dir = os.path.join(exp_dir, "logs")

    if opts.tboard:
        writer = SummaryWriter(log_dir)

    # prepare the models
    condition_encoder = ConditionEncoder(opts.input_size, opts.hidden_size, opts.max_seqlen, opts.embed_dim, opts.num_hidden_layers, opts.hidden_size, opts.use_embed_word, opts.cond_modality)
    coord_generator = CoordGenerator(opts.hidden_size, opts.num_hidden_layers, opts.latent_dim)
    diff_composition = DiffComposition(opts.input_size, opts.output_size, opts.max_seqlen, opts.pos_format, opts.align_corners, opts.trunc_logo_pix)
    seq_discriminator = SeqDiscriminator(opts.hidden_size, opts.num_hidden_layers)
    img_discriminator = ImgDiscriminator(opts.output_size, opts.output_channel, opts.hidden_size, ndf=8)

    
    if torch.cuda.is_available() and opts.multi_gpu:
        condition_encoder = nn.DataParallel(condition_encoder)
        coord_generator = nn.DataParallel(coord_generator)
        seq_discriminator = nn.DataParallel(seq_discriminator)
        img_discriminator = nn.DataParallel(img_discriminator)

    condition_encoder = condition_encoder.to(device)
    coord_generator = coord_generator.to(device)
    diff_composition = diff_composition.to(device)
    seq_discriminator = seq_discriminator.to(device)
    img_discriminator = img_discriminator.to(device)

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(list(seq_discriminator.parameters()) + list(img_discriminator.parameters()), lr=opts.lr, betas=(opts.beta1, opts.beta2))
    
    '''
    optimizerD = optim.Adam(
        [
            {"params": seq_discriminator.parameters(), "lr": opts.lr},
            {"params": img_discriminator.parameters(), "lr": opts.lr / 10},
        ],
        lr=opts.lr,
        betas=(opts.beta1, opts.beta2)
    )
    '''
    optimizerE = optim.Adam(list(condition_encoder.parameters()), lr=opts.lr, betas=(opts.beta1, opts.beta2))
    optimizerG = optim.Adam(list(coord_generator.parameters()), lr=opts.lr, betas=(opts.beta1, opts.beta2))

    for epoch in range(opts.init_epoch, opts.n_epochs):
        for idx, data in enumerate(train_dataloader):
            batches_done = epoch * len(train_dataloader) + idx + 1 
      
            modules = [condition_encoder, coord_generator, diff_composition, seq_discriminator, img_discriminator]
            imgs_list, coords_list, condition_feat_last, text_len_, batch_size_, rec_res = network_forward(data, modules, opts)

            imgs_fake_list, imgs_real_list, imgs_logo_native, imgs_glyph = imgs_list
            imgs_logo_fake_ss, imgs_logo_fake_ss_tc, imgs_logo_fake_ms, imgs_logo_fake_ms_tc, imgs_trans_fake = imgs_fake_list
            imgs_logo_real_ss, imgs_logo_real_ss_tc, imgs_logo_real_ms, imgs_logo_real_ms_tc, imgs_trans_real = imgs_real_list

            coords_gt, coords_fake = coords_list
            seq_discriminator.zero_grad()
            img_discriminator.zero_grad()

            if opts.imgdis_logo_ms:
                if opts.trunc_logo_pix == False:
                    imgs_logo_real = imgs_logo_real_ms
                    imgs_logo_fake = imgs_logo_fake_ms
                else:
                    imgs_logo_real = imgs_logo_real_ms_tc
                    imgs_logo_fake = imgs_logo_fake_ms_tc                    
            else:
                if opts.trunc_logo_pix == False:
                    imgs_logo_real = imgs_logo_real_ss
                    imgs_logo_fake = imgs_logo_fake_ss
                else:
                    imgs_logo_real = imgs_logo_real_ss_tc
                    imgs_logo_fake = imgs_logo_fake_ss_tc                    

            if opts.DG_read_diff_data:
                logitsD_real_seq = seq_discriminator(coords_gt[:batch_size_ // 2, :, :], condition_feat_last.detach()[:batch_size_ // 2], text_len_[:batch_size_ // 2, :, :])
                logitsD_fake_seq = seq_discriminator(coords_fake.detach()[:batch_size_ // 2, :, :], condition_feat_last.detach()[:batch_size_ // 2], text_len_[:batch_size_ // 2, :, :])
                logitsD_real_img = img_discriminator(imgs_logo_real[:batch_size_ // 2], condition_feat_last.detach()[:batch_size_ // 2])
                logitsD_fake_img = img_discriminator(imgs_logo_fake.detach()[:batch_size_ // 2], condition_feat_last.detach()[:batch_size_ // 2])
            else:
                logitsD_real_seq = seq_discriminator(coords_gt, condition_feat_last.detach(), text_len_)
                logitsD_fake_seq = seq_discriminator(coords_fake.detach(), condition_feat_last.detach(), text_len_)
                logitsD_real_img = img_discriminator(imgs_logo_real, condition_feat_last.detach())
                logitsD_fake_img = img_discriminator(imgs_logo_fake.detach(), condition_feat_last.detach())                
            # define the loss iterms and perform optimization
            criterion = nn.BCELoss()

            label_real = torch.full((batch_size_, 1), 1., dtype=torch.float, device=device)
            label_fake = torch.full((batch_size_, 1), 0., dtype=torch.float, device=device)

            label_real_fh = torch.full((batch_size_ // 2, 1), 1., dtype=torch.float, device=device)
            label_fake_fh = torch.full((batch_size_ // 2, 1), 0., dtype=torch.float, device=device)

            label_real_lh = torch.full((batch_size_ - batch_size_ // 2, 1), 1., dtype=torch.float, device=device)
            label_fake_lh = torch.full((batch_size_ - batch_size_ // 2, 1), 0., dtype=torch.float, device=device)
            
            if opts.DG_read_diff_data:
                lossD_real_seq = criterion(logitsD_real_seq, label_real_fh)
                lossD_fake_seq = criterion(logitsD_fake_seq, label_fake_fh)
                lossD_real_img = criterion(logitsD_real_img, label_real_fh)
                lossD_fake_img = criterion(logitsD_fake_img, label_fake_fh)
            else:
                lossD_real_seq = criterion(logitsD_real_seq, label_real)
                lossD_fake_seq = criterion(logitsD_fake_seq, label_fake)
                lossD_real_img = criterion(logitsD_real_img, label_real)
                lossD_fake_img = criterion(logitsD_fake_img, label_fake)
                        
            lossD_seq = lossD_real_seq + lossD_fake_seq
            lossD_img = lossD_real_img + lossD_fake_img

            if batches_done % opts.n_train_imgdis_interval== 0:
                lossD = opts.loss_seqdis_w * lossD_seq + opts.loss_imgdis_w * lossD_img
            else:
                lossD = opts.loss_seqdis_w * lossD_seq
            lossD.backward()
            optimizerD.step()

            # first clear the gradient
            condition_encoder.zero_grad()
            coord_generator.zero_grad()
            diff_composition.zero_grad()
            if opts.DG_read_diff_data:
                logitsG_fake_seq = seq_discriminator(coords_fake[batch_size_ // 2:, :, :], condition_feat_last[batch_size_ // 2:], text_len_[batch_size_ // 2:, :, :])
                logitsG_fake_img = img_discriminator(imgs_logo_fake[batch_size_ // 2:], condition_feat_last[batch_size_ // 2:])
            else:
                logitsG_fake_seq = seq_discriminator(coords_fake, condition_feat_last, text_len_)
                logitsG_fake_img = img_discriminator(imgs_logo_fake, condition_feat_last)     
            
            if opts.DG_read_diff_data:
                lossG_fake_seq = criterion(logitsG_fake_seq, label_real_lh)
                lossG_fake_img = criterion(logitsG_fake_img, label_real_lh)
            else:
                lossG_fake_seq = criterion(logitsG_fake_seq, label_real)
                lossG_fake_img = criterion(logitsG_fake_img, label_real)
            
            lossG_overlap = cal_overlap_loss(imgs_trans_fake, opts.max_seqlen)
            lossG_overlap_ = cal_overlap_loss(imgs_trans_real, opts.max_seqlen)

            if batches_done % opts.n_train_imgdis_interval == 0:
                if opts.opt_G_jointly == True:
                    lossG = lossG_fake_seq + opts.loss_imgdis_w * lossG_fake_img + opts.loss_ol_w * lossG_overlap 
                    lossG.backward()
                else:
                    lossG_1 = lossG_fake_seq + opts.loss_ol_w * lossG_overlap
                    lossG_1.backward(retain_graph=True)
                    lossG_2 = opts.loss_imgdis_w * lossG_fake_img
                    lossG_2.backward()
            else:
                lossG_1 = lossG_fake_seq + opts.loss_ol_w * lossG_overlap
                lossG_1.backward()
            optimizerE.step()
            optimizerG.step()

            if batches_done % opts.n_log_interval == 0:
                loss_message = (
                    f"Epoch: {epoch}/{opts.n_epochs}, Batch: {idx}/{len(train_dataloader)}, "
                    # f"loss_glyph_rec: {rec_res['l1_loss'].item():.6f}, "
                    f"lossD_fake_seq: {lossD_fake_seq.item():.6f}, "
                    f"lossD_real_seq: {lossD_real_seq.item():.6f}, "
                    f"lossD_real_img: {lossD_real_img.item():.6f}, "
                    f"lossD_fake_img: {lossD_fake_img.item():.6f}, "
                    f"lossG_fake_seq: {lossG_fake_seq.item():.6f}, "
                    f"lossG_fake_img: {lossG_fake_img.item():.6f}, "
                    f"lossG_overlap_fake: {opts.loss_ol_w * lossG_overlap.item():.6f}, "  
                    f"lossG_overlap_real: {opts.loss_ol_w * lossG_overlap_.item():.6f}, "                
                )
                print(loss_message)
            if batches_done % opts.n_summary_interval == 0:
                if opts.tboard:
                    writer.add_scalar('Loss/lossD_fake_seq', lossD_fake_seq.item(), batches_done)
                    writer.add_scalar('Loss/lossD_real_seq', lossD_real_seq.item(), batches_done)
                    writer.add_scalar('Loss/lossD_real_img', lossD_real_img.item(), batches_done)
                    writer.add_scalar('Loss/lossD_fake_img', lossD_fake_img.item(), batches_done)
                    writer.add_scalar('Loss/lossG_fake_seq', lossG_fake_seq.item(), batches_done)
                    writer.add_scalar('Loss/lossG_fake_img', lossG_fake_img.item(), batches_done)
                    writer.add_scalar('Loss/lossG_overlap', lossG_overlap.item(), batches_done)
                    writer.add_image('Images/imgs_glyph_rec', rec_res['img'][0], batches_done)
                    writer.add_image('Images/imgs_logo_fake_ss', imgs_logo_fake_ss[0], batches_done)
                    writer.add_image('Images/imgs_logo_real_ss', imgs_logo_real_ss[0], batches_done)
                    writer.add_image('Images/imgs_logo_fake_ss_tc', imgs_logo_fake_ss_tc[0], batches_done)
                    writer.add_image('Images/imgs_logo_real_ss_tc', imgs_logo_real_ss_tc[0], batches_done)
                    writer.add_image('Images/imgs_logo_fake_ms', imgs_logo_fake_ms[0], batches_done)
                    writer.add_image('Images/imgs_logo_real_ms', imgs_logo_real_ms[0], batches_done)
                    writer.add_image('Images/imgs_logo_fake_ms_tc', imgs_logo_fake_ms_tc[0], batches_done)
                    writer.add_image('Images/imgs_logo_real_ms_tc', imgs_logo_real_ms_tc[0], batches_done)
                    writer.add_image('Images/imgs_logo_native', imgs_logo_native[0], batches_done)
                    writer.add_image('Images/imgs_trans_fake_0', imgs_trans_fake[0, ..., 0], batches_done)
                    writer.add_image('Images/imgs_trans_fake_1', imgs_trans_fake[0, ..., 1], batches_done)
                    writer.add_image('Images/imgs_trans_fake_2', imgs_trans_fake[0, ..., 2], batches_done)
                    writer.add_image('Images/imgs_elements', imgs_glyph.unsqueeze(1)[0], batches_done)

        if epoch % opts.n_ckpt_interval == 0:
        # if epoch % opts.n_ckpt_interval == 0:
            modules_ = [condition_encoder, coord_generator, img_discriminator, seq_discriminator]
            module_names = ['condition_encoder', 'coord_generator', 'img_discriminator', 'seq_discriminator']
            module_fpths = []
            for mod_idx in range(len(module_names)):
                module_fpths.append(os.path.join(ckpt_dir, f"epoch_{epoch}_" + module_names[mod_idx] + ".pth"))
                if torch.cuda.is_available() and opts.multi_gpu:
                    torch.save(modules_[mod_idx].module.state_dict(), module_fpths[mod_idx])
                else:
                    torch.save(modules_[mod_idx].state_dict(), module_fpths[mod_idx])
            # do sampling ...
            print("sampling for testing data")
            sample_dir = os.path.join("experiments", opts.experiment_name, "samples", "%04d"%epoch)
            loss_overlap_test = 0.0
            os.makedirs(sample_dir, exist_ok=True)
            with torch.no_grad():
                for idx_test, data_test in enumerate(test_dataloader):
                    for sample_id in range(opts.train_sample_times):
                        imgs_list_test, _, _, _, batch_size_test_, _ = network_forward(data_test, modules, opts)
                        imgs_fake_list_test, _, _, _ = imgs_list_test
                        imgs_logo_fake_ss_test, imgs_logo_fake_ss_tc_test, _, _, imgs_trans_fake_test = imgs_fake_list_test

                        loss_overlap_test += cal_overlap_loss(imgs_trans_fake_test, opts.max_seqlen)
                        # trunc the pixel values:
                        for idx_img in range(batch_size_test_):
                            sample_fpath = os.path.join(sample_dir, "%04d"%(idx_test * opts.batch_size + idx_img) + "_%02d"%sample_id +'.png')
                            save_image(imgs_logo_fake_ss_tc_test[idx_img, :], sample_fpath, 'png')
            # testing FID metric
            print("calculating FID")
            path_fake = os.listdir(sample_dir)
            for idx_ in range(len(path_fake)):
                path_fake[idx_] = os.path.join(sample_dir, path_fake[idx_])
            path_gt = os.listdir(os.path.join(opts.data_root, opts.data_name, 'train'))
            for idx_ in range(len(path_gt)):
                path_gt[idx_] = os.path.join(opts.data_root, opts.data_name, 'train', path_gt[idx_], 'logo_resized_centre.png')
            fid_value = calculate_fid_given_paths(paths=[path_fake, path_gt], batch_size=opts.batch_size, device=device, dims=2048, )
            print("FID value: %04f"%fid_value)
            if opts.tboard:
                writer.add_scalar('Testing/FID', fid_value, batches_done)
                writer.add_scalar('Testing/overlap', loss_overlap_test / len(test_dataloader), batches_done)

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
    os.makedirs("experiments", exist_ok=True)
    debug = True
    if opts.mode == 'train':
        # Create directories
        experiment_dir = os.path.join("experiments", opts.experiment_name)
        os.makedirs(experiment_dir, exist_ok=False)  # False to prevent multiple train run by mistake
        os.makedirs(os.path.join(experiment_dir, "samples"), exist_ok=True)
        os.makedirs(os.path.join(experiment_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(experiment_dir, "results"), exist_ok=True)
        os.makedirs(os.path.join(experiment_dir, "logs"), exist_ok=True)
        os.makedirs(os.path.join(experiment_dir, "codes_backup"), exist_ok=True)
        print(f"Training on experiment {opts.experiment_name}...")
        # Dump options
        with open(os.path.join(experiment_dir, "opts.txt"), "w") as f:
            for key, value in vars(opts).items():
                f.write(str(key) + ": " + str(value) + "\n")
        for root, dirs, files in os.walk('models'):
            for fname in files:
                if fname.split('.')[-1] == 'py':
                    shutil.copy(os.path.join("models", fname), os.path.join(experiment_dir, "codes_backup"))
        shutil.copy("train.py", os.path.join(experiment_dir, "codes_backup"))
        train(opts)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()