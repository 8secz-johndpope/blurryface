"""
Uses LightCNN to generate features from a face
Code is based off the implementation from https://github.com/AlfredXiangWu/LightCNN by AlfredXiangWu

Download and extract the model from here: https://drive.google.com/uc?id=0ByNaVHFekDPRWk5XUFRvTTRIVmc&export=download
"""

__author__ = "Animesh Koratana"

import torch
import torch.backends.cudnn as cudnn

from lightcnn import LightCNN_9Layers, LightCNN_29Layers_v2
import torch.nn.functional as F

cudnn.benchmark = True


class EmbeddingGen(torch.nn.Module):
    def __init__(self, model_type="LightCNN_9Layers"):
        super(EmbeddingGen, self).__init__()
        assert model_type in ["LightCNN_9Layers", "LightCNN_29Layers_v2"]

        if model_type == "LightCNN_9Layers":
            self.model = LightCNN_9Layers()
        else:
            self.model = LightCNN_29Layers_v2()

    def _diff_grayscale(self, img):
        R = img[:, 0, :, :]
        G = img[:, 1, :, :]
        B = img[:, 2, :, :]
        return (0.299 *R + 0.587 *G + 0.114 *B).view(-1, 1, 128, 128) # im interpolating the images like they do in CV2 which was used to preprocess the images

    def forward(self, images):
        """
        :param images: a tensor of shape (batch, channels, image_size, image_size)
        :return: features: returns a (batch, 256) tensor of features
        """
        transformed = F.interpolate(images, size=(128, 128))
        transformed = self._diff_grayscale(transformed)
        if cuda_flag:
            transformed.cuda()
        return model.module.get_features(transformed)

    @classmethod
    def build(cls, model_type="LightCNN_9Layers", cuda=True, checkpoint="LightCNN_9Layers_checkpoint.pth.tar"):
        global model, img_transforms, cuda_flag

        model = cls(model_type)
        model = torch.nn.DataParallel(model)
        if cuda:
            model = model.cuda()

        print("=> loading face embedding checkpoint '{}'".format(checkpoint))
        checkpoint = torch.load(checkpoint) if cuda else torch.load(checkpoint, map_location='cpu')

        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        return model




