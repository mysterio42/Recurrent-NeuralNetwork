import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

train_ds = MNIST(root='../data',
                 train=True,
                 transform=transforms.ToTensor(),
                 download=True)
test_ds = MNIST(root='../data',
                train=False,
                transform=transforms.ToTensor(),
                download=True)

data_size = len(train_ds)


def loaders(batch_size):
    train_ds_loader = DataLoader(dataset=train_ds,
                                 batch_size=batch_size,
                                 shuffle=True)
    test_ds_loader = DataLoader(dataset=test_ds,
                                batch_size=batch_size,
                                shuffle=False)
    return train_ds_loader, test_ds_loader
