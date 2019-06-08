from style_model import *
from ffhq_dataset import get_dataloader

import torch.nn as nn

def get_style_gan():
    model = nn.Sequential(OrderedDict([
        ('g_mapping', G_mapping()),
        #('truncation', Truncation(avg_latent)),
        ('g_synthesis', G_synthesis())
    ]))
    # Load weights
    model.load_state_dict(torch.load('./karras2019stylegan-ffhq-1024x1024.for_g_all.pt'))

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # model.eval()
    model.to(device)

    for p in model.parameters():
        p.requires_grad = False

    return model
