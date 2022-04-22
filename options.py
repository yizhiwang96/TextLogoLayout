import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    # basic parameters: data shapes
    parser.add_argument('--input_size', type=int, default=64, help='the size of glyph image (character or word)')
    parser.add_argument('--output_size', type=int, default=128, help='the size of logo image')
    parser.add_argument('--max_seqlen', type=int, default=20, help='the max length of charcters(words)')
    parser.add_argument('--in_channel', type=int, default=3, help='the input glyph image channel')
    parser.add_argument('--output_channel', type=int, default=3, help='the logo image channel')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--latent_dim', type=int, default=128, help='batch size')
    parser.add_argument('--embed_dim', type=int, default=300, help='the dim of char/word embeddings')
    parser.add_argument('--pos_format', type=str, default='ltrb', choices=['whxy', 'ltrb'], help='the format of element position, width-height-centre_x-centre_y or left-top-right-bottom')
    # conditon encoding related
    parser.add_argument('--cond_modality', type=str, default='img_text', choices=['img_text', 'img', 'text'], help='whether to use word embeddings')
    # parser.add_argument('--embeds_corpus', type=str, default='baidubaike', choices=['baidubaike', 'wiki'], help='which corpus of text embeds to use')
    parser.add_argument('--use_embed_word', type=bool, default=True, help='whether to use word embeddings')
    parser.add_argument('--glyph_rec', type=bool, default=False, help='use glyph reconsturction as additional supervision')
    parser.add_argument('--loss_pt_c_w', type=float, default=0.001, help='the weight of perceptual content loss')
    parser.add_argument('--loss_rec_l1_w', type=float, default=0.1, help='the loss weight of rec l1 loss')
    # Sequence model (LSTM) realted
    parser.add_argument('--hidden_size', type=int, default=128, help='LSTM hidden size')
    parser.add_argument('--num_hidden_layers', type=int, default=2, help='LSTM number of hidden layer')
    # loss weight
    parser.add_argument('--loss_ol_w', type=float, default=100.0, help='the loss weight of overlap loss')
    # discriminator related
    parser.add_argument('--n_train_imgdis_interval', type=int, default=5, help='the interval of training Image Discriminator')
    parser.add_argument('--loss_seqdis_w', type=int, default=1.0, help='the loss weight of seqdis')
    parser.add_argument('--loss_imgdis_w', type=int, default=0.01, help='the loss weight of imgdis')
    parser.add_argument('--DG_read_diff_data', type=bool, default=True, help='Discriminator and Generator read different data, to avoid overfitting')
    parser.add_argument('--imgdis_logo_ms', type=bool, default=False, help='whether to use glyphs with different grey scales as img dis input')
    parser.add_argument('--opt_G_jointly', type=bool, default=False, help='whether to optimize the G_seq and G_img loss jointly')
    # Diff Composition Related
    parser.add_argument('--align_corners', type=bool, default=False, help='align_corners in affine 2D')
    parser.add_argument('--trunc_logo_pix', type=bool, default=False, help='whether to trunc the pixel value of composited logo images into [0, 1] ([0, 255]) for discrimination')
    # experiment related
    parser.add_argument('--data_root', type=str, default='dataset')
    parser.add_argument('--data_name', type=str, default='TextLogo3K')
    parser.add_argument('--experiment_name', type=str, default='textlogolayout')
    parser.add_argument('--init_epoch', type=int, default=0, help='init epoch')
    parser.add_argument('--multi_gpu', type=bool, default=True, help='whether to use multi-gpu')
    parser.add_argument('--n_epochs', type=int, default=800, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--n_ckpt_interval', type=int, default=20, help='save checkpoint frequency of epoch')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 of Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 of Adam optimizer')
    parser.add_argument('--eps', type=float, default=1e-8, help='Adam epsilon')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('--tboard', type=bool, default=True, help='whether use tensorboard to visulize loss')
    parser.add_argument('--n_summary_interval', type=int, default=50, help='the interval of batches when writing summary')
    parser.add_argument('--n_log_interval', type=int, default=10, help='the interval of batches when printing summary')
    # testing and validation related
    parser.add_argument('--test_epoch', type=int, default=560, help='the epoch for testing')
    parser.add_argument('--test_sample_times', type=int, default=10, help='the number of sampling times for each test case when testing')
    parser.add_argument('--train_sample_times', type=int, default=5, help='the number of sampling times for each test case when training')
    return parser