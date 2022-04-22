from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import argparse
import numpy as np
import os
import time
import jieba

def crop_img(img):
    img = np.array(img)
    # cal the minimum bbox
    img0 = np.sum(img, axis=0)
    img1 = np.sum(img, axis=1)
    x_range = np.where(img1>0)[0]
    y_range = np.where(img0>0)[0]
    if len(x_range) > 0 and len(x_range) > 0:
        x1 = x_range[0]
        x2 = x_range[-1] + 1
        y1 = y_range[0]
        y2 = y_range[-1] + 1
        img = img[x1:x2, y1:y2]
    return Image.fromarray(img)

def gen_glyph_imgs(opts, cid):
    # generate elemental images
    print('Step 1: generate elemental images...')
    try:
        font = ImageFont.truetype(opts.ttf_path, opts.glyph_size, encoding="unic")
    except:
        print('cannot open the font')
        return
    img_ele = np.ndarray((opts.glyph_size, opts.glyph_size * opts.max_seqlen), np.uint8)
    img_ele[:, :] = 0
    img_ele = Image.fromarray(img_ele)
    for idx in range(0, len(opts.input_text)):
        char = opts.input_text[idx]
        array = np.ndarray((opts.canvas_size, opts.canvas_size), np.uint8)
        array[:, :] = 0
        img_char = Image.fromarray(array)
        draw = ImageDraw.Draw(img_char)
        draw.text((opts.starting_pos, opts.starting_pos), char, (255), font=font)
        img_char = crop_img(img_char)
        img_char = img_char.resize((opts.glyph_size, opts.glyph_size), Image.ANTIALIAS)
        img_ele.paste(img_char, (idx * opts.glyph_size, 0))
    img_ele.save(os.path.join(opts.output_dir, opts.split, cid, 'elements.png'))

def gen_emebds(opts, cid):
    print('Step 2: generate char/word embeddings ...')

    print('(1) dicting embeddings ...')
    f = open(opts.embed_path,'r')
    lines = f.read().split('\n')
    word_dict = {}
    
    for line in lines[1:]:
        tmp = line.split(' ')
        embeddding_str = tmp[1:-1]
        embeddding = [float(i) for i in embeddding_str]
        embeddding_np = np.array(embeddding)
        word_dict[tmp[0]] = embeddding_np
    f.close()

    print('(2) lookup char/word embeddings ...')
    cur_idx = 0
    embed_char_res = np.zeros((opts.max_seqlen, opts.embed_dim))
    for char in opts.input_text:
        if word_dict.get(char) is not None:
            embed_char_res[cur_idx] = word_dict[char]
            cur_idx += 1
        else:
            embed_char = np.zeros(opts.embed_dim)
            embed_char_res[cur_idx] = embed_char
            cur_idx += 1
    np.save(os.path.join(opts.output_dir, opts.split, cid, 'char_embeds.npy'), embed_char_res)
    np.save(os.path.join(opts.output_dir, opts.split, cid, 'len.npy'), np.array([len(opts.input_text)]))
    
    print('(3) lookup word embeddings ...')
    cur_idx = 0
    seg_list = jieba.cut(opts.input_text)
    word_str = "\t".join(seg_list)
    words = word_str.split('\t')
    embed_word_res = np.zeros((opts.max_seqlen, opts.embed_dim))
    for word in words:
        if word_dict.get(word) is not None:
            for idx_rpt in range(len(word)):
                embed_word_res[cur_idx] = word_dict[word]
                cur_idx += 1
        else:
            for idx_rpt in range(len(word)):
                embed_word = np.zeros(opts.embed_dim)
                embed_word_res[cur_idx] = embed_word
                cur_idx += 1
    np.save(os.path.join(opts.output_dir, opts.split, cid, 'word_embeds.npy'), embed_word_res)
    np.save(os.path.join(opts.output_dir, opts.split, cid, 'len.npy'), np.array([len(opts.input_text)]))

def gen_fake_gts(opts, cid):
    print('generate fake gts')

    img_logo = Image.new('L', (opts.logo_size, opts.logo_size))
    img_logo.save(os.path.join(opts.output_dir, opts.split, cid, 'logo_resized.png'))

    coords_seg_np = np.zeros((opts.max_seqlen, 4))
    np.save(os.path.join(opts.output_dir, opts.split, cid, 'coords_seg.npy'), coords_seg_np)
    np.save(os.path.join(opts.output_dir, opts.split, cid, 'coords_seg_centre.npy'), coords_seg_np)

def gen_data(opts):
    
    if not os.path.exists(opts.output_dir):
        os.mkdir(opts.output_dir)

    if not os.path.exists(os.path.join(opts.output_dir, opts.split)):
        os.mkdir(os.path.join(opts.output_dir, opts.split))
    
    if len(opts.input_text) < 1 or len(opts.input_text) > opts.max_seqlen:
        print('invalid text length')
        return 

    cid = str(time.time()).replace('.', '')

    if not os.path.exists(os.path.join(opts.output_dir, opts.split, cid)):
        os.mkdir(os.path.join(opts.output_dir, opts.split, cid))

    gen_glyph_imgs(opts, cid)
    gen_emebds(opts, cid)
    gen_fake_gts(opts, cid)

def main():
    parser = argparse.ArgumentParser(description="prepare elemental data")
    parser.add_argument("--input_text", type=str, default='你好')
    parser.add_argument("--ttf_path", type=str, default='./dataset/ttfs/FZShangKJW.TTF')
    parser.add_argument('--canvas_size', type=int, default=300)
    parser.add_argument('--glyph_size', type=int, default=64)
    parser.add_argument('--logo_size', type=int, default=128)
    parser.add_argument('--starting_pos', type=int, default=20, help='the starting position')
    parser.add_argument("--output_dir", type=str, default='./dataset/YourDataSet/')
    parser.add_argument("--split", type=str, default='test')
    parser.add_argument("--max_seqlen", type=int, default=20)
    parser.add_argument("--embed_path", type=str, default='./dataset/Embeddings/sgns.baidubaike.bigram-char')
    parser.add_argument("--embed_dim", type=int, default=300)
    opts = parser.parse_args()
    gen_data(opts)

if __name__ == "__main__":
    main()

