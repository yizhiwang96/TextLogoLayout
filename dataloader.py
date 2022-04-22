import os
from torchvision.io import read_image
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as T
import numpy as np

class TextLogoDataset(data.Dataset):
    def __init__(self, data_root, mode='train'):
        data_root_path = os.path.join(data_root, mode)
        self.content_paths = []
        for root, dirs, files in os.walk(data_root_path):
            for dir_name in dirs:
                self.content_paths.append(os.path.join(data_root_path, dir_name))
        self.content_paths.sort()

    def __len__(self):
        return len(self.content_paths)

    def __getitem__(self, index):
        content_path = self.content_paths[index]
        item = {}
        item['imgs_glyph'] = torch.FloatTensor(np.array(Image.open(os.path.join(content_path, 'elements.png')))) / 255.
        item['imgs_logo'] = torch.FloatTensor(np.array(Image.open(os.path.join(content_path, 'logo_resized.png')))) / 255.
        item['coords_gt'] = torch.FloatTensor(np.load(os.path.join(content_path, 'coords_seg.npy')))
        item['coords_gt_centre'] = torch.FloatTensor(np.load(os.path.join(content_path, 'coords_seg_centre.npy')))
        item['text_len'] = torch.LongTensor(np.load(os.path.join(content_path, 'len.npy')))
        item['embeds_char'] = torch.FloatTensor(np.load(os.path.join(content_path, 'char_embeds.npy')))
        item['embeds_word'] = torch.FloatTensor(np.load(os.path.join(content_path, 'word_embeds.npy')))
        return item

def get_loader(data_root, batch_size, mode='train'):
    dataset = TextLogoDataset(data_root, mode)
    dataloader = data.DataLoader(dataset, batch_size, shuffle=(mode == 'train'), num_workers=batch_size, drop_last=(mode == 'train'))
    return dataloader