from style_model import *
from latentgen import LatentGen
from embeddings import EmbeddingGen
from ffhq_dataset import get_dataloader

import torch.nn as nn
# imgs = torchvision.utils.make_grid(imgs, nrow=nb_cols)
# torchvision.utils.save_image(imgs, "sample.png", nrow=10, range=(-1, 1))


def main():
    batch_size = 8
    discriminator = EmbeddingGen.build()
    latent_generator = LatentGen.build()

    model = get_style_gan()
    data_loader = get_dataloader(batch_size=batch_size)

    optimizer = torch.optim.Adam(latent_generator.model.module.fc2.parameters(), lr=1e-3)

    train(model, latent_generator, discriminator, optimizer, data_loader, batch_size)


def train(model, latent_generator, discriminator, optimizer, data_loader, batch_size):
    latent_generator.eval()
    latent_generator.model.module.fc2.train()
    discriminator.eval()
    model.eval()
    iteration_counter = 0
    for epoch in range(10):
        for i, (input, _) in enumerate(data_loader):
            input = input.cuda()
            latents = latent_generator(input)
            initial_embeddings = discriminator(input)
            imgs = model(latents)
            imgs = (imgs.clamp(-1, 1) + 1) / 2.0
            # imgs = imgs.cpu()
            # final_embeddings = discriminator(imgs)
            final_image = torch.nn.functional.interpolate(imgs, size=(128, 128))
            loss = nn.modules.loss.MSELoss()(final_image, input.detach())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if iteration_counter % 100 == 0:
                print("iteration:", iteration_counter * batch_size, "loss:", loss.item())
                if iteration_counter % 10000 == 0:
                    print("Saving checkpoint ...")
                    torch.save(model.state_dict(), "checkpoints/" +
                               "model_at_iteration_" + str(iteration_counter) + ".tar")
                    print("checkpoints/" +
                          "model_at_iteration_" + str(iteration_counter) + ".tar")
                    # Sample input/output
                    input_image = torchvision.utils.make_grid(imgs, nrow=4)
                    output_image = torchvision.utils.make_grid(input, nrow=4)
                    torchvision.utils.save_image(input_image, "input_output/" + str(
                        iteration_counter) + "input" + ".png", nrow=10, range=(-1, 1))
                    torchvision.utils.save_image(final_image, "input_output/" + str(
                        iteration_counter) + "output" + ".png", nrow=10, range=(-1, 1))
                    print("saved", str(
                        iteration_counter) + "input" + ".png", str(
                            iteration_counter) + "output" + ".png")
            iteration_counter += 1
        print("epoch", epoch, "done")


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
