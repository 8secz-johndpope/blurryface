import torch
import torch.backends.cudnn as cudnn

import torch.nn as nn

from lightcnn import LightCNN_9Layers, LightCNN_29Layers_v2
import torch.nn.functional as F

cudnn.benchmark = True


class LatentGen(nn.Module):
    def __init__(self, model):
        super(LatentGen, self).__init__()
        self.model = model

    def grayscale(self, x):
        R = x[:, 0, :, :]
        G = x[:, 1, :, :]
        B = x[:, 2, :, :]
        return (0.299 * R + 0.587 * G + 0.114 * B).view(-1, 1, 128, 128)

    def forward(self, x):
        x = F.interpolate(x, size=(128, 128))
        x = self.grayscale(x)
        x = self.model(x)[0]
        return x

    def train(self, mode=True):
        if mode:
            self.model.fc2.train()
        else:
            self.eval()

    def eval(self):
        self.model.fc2.eval()

    @classmethod
    def build(cls, model_type="LightCNN_9Layers", checkpoint="LightCNN_9Layers_checkpoint.pth.tar", cuda=True):
        assert model_type in ["LightCNN_9Layers", "LightCNN_29Layers_v2"]

        if not checkpoint:
            checkpoint = model_type + "_checkpoint.pth.tar"

        if model_type == "LightCNN_9Layers":
            model = LightCNN_9Layers()
        else:
            model = LightCNN_29Layers_v2()
        model.eval()
        model = torch.nn.DataParallel(model)
        if cuda:
            model = model.cuda()

        print("=> loading face embedding checkpoint '{}'".format(checkpoint))

        if cuda:
            checkpoint = torch.load(checkpoint)
        else:
            checkpoint = torch.load(checkpoint, map_location='cpu')

        model.load_state_dict(checkpoint['state_dict'])

        # Now replace the linear layer with a new linear layer
        model.module.fc2 = nn.Linear(256, 512)  # maps to the latent space of the stylegan

        model.eval()
        for param in model.module.parameters():
            param.requires_grad = False

        model.module.fc2.train()
        for param in model.module.fc2.parameters():
            param.requires_grad = True

        return cls(model)




