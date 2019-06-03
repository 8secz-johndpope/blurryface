import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from stylegan import get_style_gan
from torchvision import transforms
from torch.nn.functional import interpolate

import facenet_pytorch as fp
import cv2, dlib
import math

scale = 4
detector = dlib.get_frontal_face_detector()

def main():
    num_eval = 50
    batch_size = 32
    fc_lr = 1e-2

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
    run_training(1000, batch_size, anonymizer, mtcnn, facenet, fc, optimizer, loss_fn)


    ######
    # Run Eval
    print(f"Evaluating On {num_eval} Images")
    print("Saving checkpoint ...")

    anonymizer.eval()
    facenet.eval()
    for i in range(0, num_eval, 5):
        with torch.no_grad():
            latents = torch.randn(batch_size, 512).cuda()
            generated_image = anonymizer(latents)
            generated_image = (generated_image.clamp(-1, 1) + 1) / 2.0

            aligned = []
            for ogimg in generated_image:
                img = ogimg.permute(1, 2, 0).cpu().numpy() * 255
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                dets = detector(img.astype('uint8'), 1)

                if len(dets) > 0:
                    cropped_image = fp.prewhiten(crop_image(ogimg.permute(1, 2, 0), dets[0])).permute(2,0,1)
                    cropped_image = interpolate(cropped_image, size=160)
                    aligned.append(cropped_image)
                else:
                    aligned.append(interpolate(ogimg, size=160))
            for i in range(len(aligned)):
                target = torch.zeros(3,160,160)
                w = aligned[i].size(1)
                h = aligned[i].size(2)
                target[:, math.floor((160 - w)/2):160 - math.ceil((160 - w)/2), :] = aligned[i]
                aligned[i] = target
            preprocessed_image = torch.stack(aligned).cuda()
            predicted_features = facenet(preprocessed_image)
            predicted_features = fc(predicted_features)

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
    for j in range(0, num_images, batch_size):

        with torch.no_grad():
            latents = torch.randn(batch_size, 512).cuda()
            generated_image = anonymizer(latents)
            generated_image = (generated_image.clamp(-1, 1) + 1) / 2.0
            generated_image = interpolate(generated_image, size=(160, 160))

            aligned = []
            for ogimg in generated_image:
                img = ogimg.permute(1, 2, 0).cpu().numpy() * 255
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                dets = detector(img.astype('uint8'), 1)
                if len(dets) > 0:
                    cropped_image = fp.prewhiten(crop_image(ogimg.permute(1, 2, 0), dets[0])).permute(2,0,1)
                    cropped_image = interpolate(cropped_image, size=160)
                    aligned.append(cropped_image)
                else:
                    aligned.append(interpolate(ogimg, size=160))
            for i in range(len(aligned)):
                    target = torch.zeros(3,160,160)
                    w = aligned[i].size(1)
                    h = aligned[i].size(2)
                    target[:, math.floor((160 - w)/2):160 - math.ceil((160 - w)/2), :] = aligned[i]
                    aligned[i] = target
            generated_image = torch.stack(aligned).cuda()
            predicted_features = facenet(generated_image)

        predicted_features = fc(predicted_features)

        loss = loss_fn(predicted_features, latents) # we wanna make the latent features representative
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if j % 128 == 0:
            print(f"Iteration: {j} \t\t Loss {loss.item()}")

def build_facenet_model(latent_space=512):
    mtcnn = fp.MTCNN(device=torch.device("cuda"))
    facenet = fp.InceptionResnetV1(pretrained='vggface2').eval()
    for p in facenet.parameters():
        p.requires_grad = False

    facenet = facenet.cuda()

    fc = nn.Sequential(*[
#        nn.Linear(512, 512),
#        nn.ReLU(),
#        nn.Linear(512, 512),
#        nn.ReLU(),
#        nn.Linear(512, 512),
#        nn.ReLU(),
#        nn.Linear(512, 512),
#        nn.ReLU(),
       nn.Linear(512,latent_space)
    ])

    fc.train()
    fc = fc.cuda()

    return mtcnn, facenet, fc






    # resnet = models.resnet18(pretrained=True)
    # resnet.fc = nn.Linear(512, latent_space, bias=True)
    # for param in resnet.fc.parameters():
    #     param.requires_grad = True
    # resnet = resnet.cuda()

    # return resnet


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

def crop_image(image, det):
    left, top, right, bottom = rect_to_tuple(det)
    return image[top:bottom, left:right]

def rect_to_tuple(rect):
    left = rect.left()
    right = rect.right()
    top = rect.top()
    bottom = rect.bottom()
    return left, top, right, bottom

if __name__ == '__main__':
    main()
