import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
t
import matplotlib.pyplot as plt
import time
import os
import copy

from stylegan import get_style_gan
from ffhq_dataset import get_dataloader

from torch.nn.functional import interpolate

def main():
    batch_size = 8

    resnet = build_resnet_model()
    anonymizer = get_style_gan()
    data_loader, pretransforms = get_dataloader(batch_size=batch_size)

    optimizer = optim.Adam(filter(lambda x: x.requires_grad, resnet.parameters()), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    num_iterations = 0

    for epoch in range(10):
        for i, (og_image, _) in enumerate(data_loader):
            og_image = og_image.cuda()
            latent_features = resnet(pretransforms(og_image))
            generated_image = anonymizer(latent_features)
            generated_image = (generated_image.clamp(-1, 1) + 1) / 2.0
            generated_image = interpolate(generated_image, size=(128, 128))

            loss = loss_fn(generated_image, og_image) # we wanna make the latent features representative
            loss.backward()
            optimizer.step()

            num_iterations += 1

            if num_iterations % 100 == 0:
                print("iteration:", num_iterations * batch_size, "loss:", loss.item())
                if num_iterations % 10000 == 0:
                    print("Saving checkpoint ...")
                    torch.save(resnet.state_dict(), "checkpoints/" +
                               "resnet_at_iteration_" + str(num_iterations) + ".tar")
                    print("checkpoints/" +
                          "resnet_at_iteration_" + str(num_iterations) + ".tar")

                    # Sample input/output
                    input_image = torchvision.utils.make_grid(og_image, nrow=4)
                    output_image = torchvision.utils.make_grid(generated_image, nrow=4)
                    torchvision.utils.save_image(input_image, "input_output/" + str(
                        num_iterations) + "input" + ".png", nrow=10, range=(-1, 1))
                    torchvision.utils.save_image(output_image, "input_output/" + str(
                        num_iterations) + "output" + ".png", nrow=10, range=(-1, 1))
                    print("saved", str(
                        num_iterations) + "input" + ".png", str(
                            num_iterations) + "output" + ".png")

def build_resnet_model(latent_space=512, feature_extracting=True):
    resnet = models.resnet18(pretrained=True)

    if feature_extracting:
        for param in resnet.parameters():
            param.requires_grad = False
    resnet.fc = nn.Linear(512, latent_space, bias=True)
    for param in resnet.fc.parameters():
        param.requires_grad = True

    resnet = resnet.cuda()

    return resnet



# def train(model, latent_generator, discriminator, optimizer, data_loader, batch_size):
#     latent_generator.eval()
#     latent_generator.model.module.fc2.train()
#     discriminator.eval()
#     model.eval()
#     iteration_counter = 0
#     for epoch in range(10):
#         for i, (input, _) in enumerate(data_loader):
#             input = input.cuda()
#             latents = latent_generator(input)
#             initial_embeddings = discriminator(input)
#             imgs = model(latents)
#             imgs = (imgs.clamp(-1, 1) + 1) / 2.0
#             # imgs = imgs.cpu()
#             # final_embeddings = discriminator(imgs)
#             final_image = torch.nn.functional.interpolate(imgs, size=(128, 128))
#             loss = nn.modules.loss.MSELoss()(final_image, input.detach())
#             loss.backward()
#             optimizer.step()
#             optimizer.zero_grad()
#
#             if iteration_counter % 100 == 0:
#                 print("iteration:", iteration_counter * batch_size, "loss:", loss.item())
#                 if iteration_counter % 10000 == 0:
#                     print("Saving checkpoint ...")
#                     torch.save(model.state_dict(), "checkpoints/" +
#                                "model_at_iteration_" + str(iteration_counter) + ".tar")
#                     print("checkpoints/" +
#                           "model_at_iteration_" + str(iteration_counter) + ".tar")
#                     # Sample input/output
#                     input_image = torchvision.utils.make_grid(imgs, nrow=4)
#                     output_image = torchvision.utils.make_grid(input, nrow=4)
#                     torchvision.utils.save_image(input_image, "input_output/" + str(
#                         iteration_counter) + "input" + ".png", nrow=10, range=(-1, 1))
#                     torchvision.utils.save_image(final_image, "input_output/" + str(
#                         iteration_counter) + "output" + ".png", nrow=10, range=(-1, 1))
#                     print("saved", str(
#                         iteration_counter) + "input" + ".png", str(
#                             iteration_counter) + "output" + ".png")
#             iteration_counter += 1
#         print("epoch", epoch, "done")

if __name__ == '__main__':
    main()
