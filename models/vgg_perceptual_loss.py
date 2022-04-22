import torch
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VGG19Feats(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(VGG19Feats, self).__init__()
        vgg = torchvision.models.vgg19(pretrained=True).to(device) #.cuda()
        # vgg.eval()
        vgg_pretrained_features = vgg.features.eval()
        self.requires_grad = requires_grad
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(3): #(3):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(3, 8): #(3, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(8, 13): #(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(13, 22): #(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(22, 31):#(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not self.requires_grad:
            for param in self.parameters():
                param.requires_grad = False
 
    def forward(self, img):
        conv1_2 = self.slice1(img)
        conv2_2 = self.slice2(conv1_2)
        conv3_2 = self.slice3(conv2_2)
        conv4_2 = self.slice4(conv3_2)
        conv5_2 = self.slice5(conv4_2)
        out = [conv1_2, conv2_2, conv3_2, conv4_2, conv5_2]
        return out


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        self.vgg = VGG19Feats().to(device)
        self.criterion = torch.nn.functional.l1_loss
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.weights = [1.0/2.6, 1.0/4.8, 1.0/3.7, 1.0/5.6, 1.0*10/1.5]
        # self.weights = [0, 0, 1.0, 1.0, 0]
    def forward(self, input_img, target_img):

        if input_img.shape[1] != 3:
            input_img = input_img.repeat(1, 3, 1, 1)
            target_img = target_img.repeat(1, 3, 1, 1)
        input_img = (input_img - self.mean) / self.std
        target_img = (target_img - self.mean) / self.std

        x_vgg, y_vgg = self.vgg(input_img), self.vgg(target_img)
        loss = {}

        loss['pt_c_loss'] = self.weights[0] * self.criterion(x_vgg[0], y_vgg[0])+\
                            self.weights[1] * self.criterion(x_vgg[1], y_vgg[1])+\
                            self.weights[2] * self.criterion(x_vgg[2], y_vgg[2])+\
                            self.weights[3] * self.criterion(x_vgg[3], y_vgg[3])+\
                            self.weights[4] * self.criterion(x_vgg[4], y_vgg[4])
        
        return loss['pt_c_loss']