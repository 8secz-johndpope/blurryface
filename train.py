import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from stylegan import get_style_gan
from torchvision import transforms
from torch.nn.functional import interpolate

import facenet_pytorch as fp

trans = transforms.Compose([
    torch.Tensor.byte
])

def main():
    num_eval = 100
    batch_size = 32
    fc_lr = 1e-3

    mtcnn, facenet, fc = build_facenet_model()
    anonymizer = get_style_gan()
    loss_fn = torch.nn.MSELoss()

    # Now freeze the full model and then train only the fc layer
    for param in facenet.parameters(): # Freeze the full model
        param.requires_grad = False
    for param in fc.parameters(): # Unfreeze the fc layer
        param.requires_grad = True

    facenet.eval()
    fc.train()
    anonymizer.eval()
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, fc.parameters()), lr=fc_lr)
    run_training(10000, batch_size, anonymizer, mtcnn, facenet, fc, optimizer, loss_fn)


    ######
    # Run Eval
    print(f"Evaluating On {num_eval} Images")
    print("Saving checkpoint ...")

    anonymizer.eval()
    facenet.eval()
    for i in range(0, num_eval, 5):
        with torch.no_grad:
            latents = torch.randn(batch_size, 512).cuda()
            generated_image = anonymizer(latents)
            generated_image = (generated_image.clamp(-1, 1) + 1) / 2.0

            preprocessed_image = interpolate(generated_image, size=(160, 160))

            # aligned = []
            # for img in preprocessed_image:
            #     x_aligned = mtcnn(img.cpu())
            #     aligned.append(x_aligned)
            # preprocessed_image = torch.stack(aligned)
            predicted_features = facenet(preprocessed_image)
            predicted_features = fc(fp.prewhiten(predicted_features))

            resnet_based_images = anonymizer(predicted_features)
            resnet_based_images = (resnet_based_images.clamp(-1, 1) + 1) / 2.0

            generator_image = torchvision.utils.make_grid(generated_image, nrow=4)
            resnet_image = torchvision.utils.make_grid(resnet_based_images, nrow=4)
            torchvision.utils.save_image(generator_image, "input_output/" + str(
                i) + "gen" + ".png", nrow=10, range=(-1, 1))
            torchvision.utils.save_image(resnet_image, "input_output/" + str(
                i) + "res" + ".png", nrow=10, range=(-1, 1))

    print("Saved Images")




def run_training(num_images, batch_size, anonymizer, mtcnn, facenet, fc, optimizer, loss_fn):
    for i in range(0, num_images, batch_size):

        with torch.no_grad():
            latents = torch.randn(batch_size, 512).cuda()
            generated_image = anonymizer(latents)
            generated_image = (generated_image.clamp(-1, 1) + 1) / 2.0
            generated_image = interpolate(generated_image, size=(160, 160))

            # aligned = []
            # for img in generated_image:
            #     x_aligned = mtcnn(img.cpu())
            #     aligned.append(x_aligned)
            # generated_image = torch.stack(aligned)
            predicted_features = facenet(fp.prewhiten(generated_image))

        predicted_features = fc(predicted_features)

        loss = loss_fn(predicted_features, latents) # we wanna make the latent features representative
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i % 128 == 0:
            print(f"Iteration: {i} \t\t Loss {loss.item()}")

def build_facenet_model(latent_space=512):
    mtcnn = fp.MTCNN(device=torch.device("cuda"))
    facenet = fp.InceptionResnetV1(pretrained='vggface2').eval()
    for p in facenet.parameters():
        p.requires_grad = False

    facenet = facenet.cuda()

    fc = nn.Sequential([
        nn.Linear(512, 1024),
        nn.ReLU(),
        nn.Linear(1024,latent_space)
    ])

    fc.train()

    return mtcnn, facenet, fc






    # resnet = models.resnet18(pretrained=True)
    # resnet.fc = nn.Linear(512, latent_space, bias=True)
    # for param in resnet.fc.parameters():
    #     param.requires_grad = True
    # resnet = resnet.cuda()

    return resnet


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
