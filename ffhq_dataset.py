from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from torch.nn.functional import interpolate

from torchvision import transforms

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

def get_dataloader(root="ffhq", batch_size=32, shuffle=True, workers=8):
    data = ImageFolder(root=root, transform=transforms.Compose([
        transforms.ToTensor()
    ]))
    loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=workers, pin_memory=True)

    pretransforms = transforms.Compose([
        transforms.Lambda(lambda x: interpolate(x, size=(224,224), mode="bilinear")),
        normalize
    ])

    return loader, pretransforms