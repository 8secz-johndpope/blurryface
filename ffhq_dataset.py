from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from torch.nn.functional import interpolate, normalize

from torchvision import transforms

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def pretransforms(x):
    x = interpolate(x, size=(224,224), mode="bilinear")
    x = normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return x

def get_dataloader(root="ffhq", batch_size=32, shuffle=True, workers=8):
    data = ImageFolder(root=root, transform=transforms.Compose([
        transforms.ToTensor()
    ]))
    loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=workers, pin_memory=True)

    return loader, pretransforms