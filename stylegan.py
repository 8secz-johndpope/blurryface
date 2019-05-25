from style_model import *
from latentgen import LatentGen
from embeddings import EmbeddingGen
from ffhq_dataset import get_dataloader

import torch.nn as nn
# imgs = torchvision.utils.make_grid(imgs, nrow=nb_cols)
# torchvision.utils.save_image(imgs, "sample.png", nrow=10, range=(-1, 1))


def main():
    discriminator = EmbeddingGen.build()
    latent_generator = LatentGen.build()

    model = get_style_gan()
    data_loader = get_dataloader(batch_size=8)

    optimizer = torch.optim.Adam(latent_generator.model.module.fc2.parameters(), lr=1e-3)

    train(model, latent_generator, discriminator, optimizer, data_loader)



def train(model, latent_generator, discriminator, optimizer, data_loader):
    latent_generator.eval()
    latent_generator.model.module.fc2.train()
    discriminator.eval()
    model.eval()


    for i, (input, _) in enumerate(data_loader):
        input = input.cuda()
        latents = latent_generator(input)
        initial_embeddings = discriminator(input)
        imgs = model(latents)
        imgs = (imgs.clamp(-1, 1) + 1) / 2.0
        # imgs = imgs.cpu()
        final_embeddings = discriminator(imgs)

        loss = - nn.modules.loss.KLDivLoss()(final_embeddings, initial_embeddings.detach())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i % 100 == 0:
            print("iteration:", i, "loss:", loss.item())


# def loss_function(initial_embeddings, final_embeddings):
#     return - loss.KLDivLoss()(initial_embeddings, final_embeddings)

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

if __name__ == "__main__":
    main()
