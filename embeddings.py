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

model = None
cuda_flag = None
# img_transforms = None


def init(model_type="LightCNN_9Layers", checkpoint=None, cuda=True):
    assert model_type in ["LightCNN_9Layers", "LightCNN_29Layers_v2"]
    global model, img_transforms, cuda_flag

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
        cuda_flag = cuda

    print("=> loading face embedding checkpoint '{}'".format(checkpoint))
    if cuda:
        checkpoint = torch.load(checkpoint)
    else:
        checkpoint = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])

    # img_transforms = transforms.Compose([
    #     transforms.Grayscale(),
    #     transforms.Resize((128, 128))
    # ])


def _diff_grayscale(img):
    R = img[:, 0, :, :]
    G = img[:, 1, :, :]
    B = img[:, 2, :, :]
    return (0.299 *R + 0.587 *G + 0.114 *B).view(-1, 1, 128, 128) # im interpolating the images like they do in CV2 which was used to preprocess the images


def get_features(images):
    """
    :param images: a tensor of shape (batch, channels, image_size, image_size)
    :return: features: returns a (batch, 256) tensor of features
    """
    global model, cuda_flag
    assert model
    transformed = F.interpolate(images, size=(128,128))
    transformed = _diff_grayscale(transformed)
    if cuda_flag:
        transformed.cuda()
    return model.module.get_features(transformed)


