import pytorch_lightning as pl
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "../data/"):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        self.mnist_train = MNIST(self.data_dir, train=True, transform=self.transform)
        self.mnist_train = [self.mnist_train[i] for i in range(2200)]
        self.mnist_train, self.mnist_val = random_split(self.mnist_train, [2000, 200])
        self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)
        self.mnist_test = [self.mnist_test[i] for i in range(3000,4000)]

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=32)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=32)


if __name__ == '__main__':
    mnistdatamodule = MNISTDataModule()
    dataset = mnistdatamodule.train_dataloader()