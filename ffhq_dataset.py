from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms



def get_dataloader(root="ffhq", batch_size=32, shuffle=True, workers=8):
    data = ImageFolder(root=root, transform=transforms.ToTensor())
    loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=workers, pin_memory=True)
    return loader