import torch
import torch.nn as nn
import torchvision
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ImgEncoder(torch.nn.Module):
    def __init__(self):
        super(ImgEncoder, self).__init__()
        vgg = torchvision.models.vgg19_bn(pretrained=True).to(device)

        vgg_features = vgg.features.train()
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        for x in range(5):
            self.slice1.add_module(str(x), vgg_features[x])
        for x in range(5, 12):
            self.slice2.add_module(str(x), vgg_features[x])
        for x in range(12, 19):
            self.slice3.add_module(str(x), vgg_features[x])
        for x in range(19, 32):
            self.slice4.add_module(str(x), vgg_features[x])
        for x in range(32, 45):
            self.slice5.add_module(str(x), vgg_features[x])
        
        self.conv_last1 = nn.Sequential(*[nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=3 // 2), nn.BatchNorm2d(512), nn.ReLU(True)])
        self.conv_last2 = nn.Sequential(*[nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=3 // 2), nn.BatchNorm2d(512), nn.ReLU(True)])

        # self.pool = torch.nn.MaxPool2d((4, 4))
    
    def forward(self, img):
        img = (img - self.mean) / self.std
        conv1_2 = self.slice1(img)
        conv2_2 = self.slice2(conv1_2)
        conv3_2 = self.slice3(conv2_2)
        conv4_2 = self.slice4(conv3_2)
        conv5_2 = self.slice5(conv4_2)
        # ret = self.pool(conv5_2)
        ret = self.conv_last1(conv5_2)
        ret = self.conv_last2(ret)
        return ret
