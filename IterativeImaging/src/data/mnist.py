import os

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

import torchvision as tv

import lightning as L

from ffcv.fields import IntField, RGBImageField
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import Convert, ToTensor, ToTorchImage
from ffcv.writer import DatasetWriter


from constants import DATASETS_ROOT


class MNISTDataset(Dataset):
    def __init__(self, root: Path, train=True, download=True) -> None:
        self.dataset = tv.datasets.MNIST(root, train=train, download=download)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx):

        # The original MNIST dataset is grayscale, but we want to convert it to RGB
        img, label = self.dataset[idx]
        img = img.convert('RGB')

        return img, label


class MNISTDataModule(L.LightningDataModule):

    DS_ROOT = Path(DATASETS_ROOT) / 'mnist'
    FFCV_TRAIN = DS_ROOT / 'train.beton'
    FFCV_TEST = DS_ROOT / 'test.beton'

    def __init__(self, batch_size: int = 32, num_workers: int = -1) -> None:
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.image_pipeline = [ToTensor(),
                               ToTorchImage(), Convert(torch.float32),
                               tv.transforms.Resize((32, 32)),
                               tv.transforms.Normalize(127.5, 127.5)]
        self.label_pipeline = [IntDecoder(), ToTensor()]
        # `ToDevice` is removed from the pipelines as lightning will handle it

    def ffcv_packing(self, dataset: Dataset, save_path: Path) -> None:
        writer = DatasetWriter(
            save_path,
            fields={
                'image': RGBImageField(),
                'label': IntField(),
            }
        )
        writer.from_indexed_dataset(dataset)

    def prepare_data(self) -> None:

        if not os.path.exists(self.FFCV_TRAIN):
            print('Packing MNIST trainset via ffcv ...')
            trainset = MNISTDataset(self.DS_ROOT, train=True, download=True)
            self.ffcv_packing(trainset, self.FFCV_TRAIN)

        if not os.path.exists(self.FFCV_TEST):
            print('Packing MNIST testset via ffcv ...')
            testset = MNISTDataset(self.DS_ROOT, train=False, download=True)
            self.ffcv_packing(testset, self.FFCV_TEST)

    def train_dataloader(self) -> DataLoader:
        return Loader(
            self.FFCV_TRAIN,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            order=OrderOption.RANDOM,
            drop_last=True,
            pipelines={
                'image': self.image_pipeline,
                'label': self.label_pipeline
            }
        )

    def val_dataloader(self) -> DataLoader:
        return Loader(
            self.FFCV_TEST,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            order=OrderOption.RANDOM,
            # drop_last=True,
            pipelines={
                'image': self.image_pipeline,
                'label': self.label_pipeline
            }
        )

    def test_dataloader(self) -> DataLoader:
        return Loader(
            self.FFCV_TEST,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            order=OrderOption.RANDOM,
            # drop_last=True,
            pipelines={
                'image': self.image_pipeline,
                'label': self.label_pipeline
            }
        )
