import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import models

from torchvision import transforms

from stylegan import get_style_gan

from torch.nn.functional import interpolate

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

def main():
    batch_size = 32
    lr = 1e-3

    resnet = build_resnet_model()
    anonymizer = get_style_gan()

    optimizer = optim.Adam(filter(lambda x: x.requires_grad, resnet.parameters()), lr=lr)
    loss_fn = torch.nn.MSELoss()

    for i in range(0, 10000000, batch_size):

        with torch.no_grad():
            latents = torch.randn(batch_size, 512).cuda()
            generated_image = anonymizer(latents)
            generated_image = (generated_image.clamp(-1, 1) + 1) / 2.0
            generated_image = interpolate(generated_image, size=(224, 224)).cpu()
            generated_image = torch.stack([normalize(x).cpu() for x in generated_image]).detach().cuda()

        predicted_features = resnet(generated_image)

        loss = loss_fn(predicted_features, latents) # we wanna make the latent features representative
        loss.backward()
        optimizer.step()

        if i % 128 == 0:
            print(f"Iteration: {i} \t Loss {loss.item()}")

        # if i % 128 == 0:
        #     print("iteration:", i * batch_size, "loss:", loss.item())
        #     if i % 128 == 0:
        #         print("Saving checkpoint ...")
        #         torch.save(resnet.state_dict(), "checkpoints/" +
        #                    "resnet_at_iteration_" + str(i) + ".tar")
        #         print("checkpoints/" +
        #               "resnet_at_iteration_" + str(i) + ".tar")
        #
        #         # Sample input/output
        #         input_image = torchvision.utils.make_grid(og_image, nrow=4)
        #         output_image = torchvision.utils.make_grid(generated_image, nrow=4)
        #         torchvision.utils.save_image(input_image, "input_output/" + str(
        #             i) + "input" + ".png", nrow=10, range=(-1, 1))
        #         torchvision.utils.save_image(output_image, "input_output/" + str(
        #             i) + "output" + ".png", nrow=10, range=(-1, 1))
        #         print("saved", str(
        #             i) + "input" + ".png", str(
        #                 i) + "output" + ".png")

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
