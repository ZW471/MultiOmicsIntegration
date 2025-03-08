import os
import pathlib
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Iterable, Callable, Any, List

import pandas as pd
import torch
from loguru import logger
import lightning as L
from beartype import beartype as typechecker
from torch.utils.data import DataLoader
from torch_geometric import transforms as T
from torch_geometric.data import Dataset
from tqdm import tqdm

from datasets.download import download_file


class Collater:
    def __call__(self, batch):
        return batch

class SingleCellDataLoader(DataLoader):
    def __init__(
            self,
            dataset,
            batch_size,
            shuffle=False,
            **kwargs
    ):
        super().__init__(dataset, batch_size, shuffle, collate_fn=Collater(), **kwargs)


class SingleCellDataModule(L.LightningDataModule, ABC):

    @abstractmethod
    def download(self):
        ...

    def setup(self, stage: Optional[str] = None):
        self.download()

        if stage == "fit" or stage is None:
            logger.info("Preprocessing training data")
            self.train_ds = self.train_dataset()
            logger.info("Preprocessing validation data")
            self.val_ds = self.val_dataset()
        elif stage == "test":
            logger.info("Preprocessing test data")
            if hasattr(self, "test_dataset_names"):
                for split in self.test_dataset_names:
                    setattr(self, f"{split}_ds", self.test_dataset(split))
            else:
                self.test_ds = self.test_dataset()
        elif stage == "lazy_init":
            logger.info("Preprocessing validation data")
            self.val_ds = self.val_dataset()


    @typechecker
    def compose_transforms(self, transforms: Iterable[Callable]) -> T.Compose:
        """Compose an iterable of Transforms into a single transform.

        :param transforms: An iterable of transforms.
        :type transforms: Iterable[Callable]
        :raises ValueError: If ``transforms`` is not a list or dict.
        :return: A single transform.
        :rtype: T.Compose
        """
        if isinstance(transforms, list):
            return T.Compose(transforms)
        elif isinstance(transforms, dict):
            return T.Compose(list(transforms.values()))
        else:
            raise ValueError("Transforms must be a list or dict")

    @abstractmethod
    def train_dataset(self) -> Dataset:
        """
        Implement the construction of the training dataset.

        :return: The training dataset.
        :rtype: Dataset
        """
        ...

    @abstractmethod
    def val_dataset(self) -> Dataset:
        """
        Implement the construction of the validation dataset.

        :return: The validation dataset.
        :rtype: Dataset
        """
        ...

    @abstractmethod
    def test_dataset(self) -> Dataset:
        """
        Implement the construction of the test dataset.

        :return: The test dataset.
        :rtype: Dataset
        """
        ...

    @abstractmethod
    def train_dataloader(self) -> SingleCellDataLoader:
        """
        Implement the construction of the training dataloader.

        :return: The training dataloader.
        :rtype: SingleCellDataLoader
        """
        ...

    @abstractmethod
    def val_dataloader(self) -> SingleCellDataLoader:
        """Implement the construction of the validation dataloader.

        :return: The validation dataloader.
        :rtype: SingleCellDataLoader
        """
        ...

    @abstractmethod
    def test_dataloader(self) -> SingleCellDataLoader:
        """Implement the construction of the test dataloader.

        :return: The test dataloader.
        :rtype: SingleCellDataLoader
        """
        ...


class MojitooDataModule(SingleCellDataModule, ABC):
    @staticmethod
    def _download(name, path, overwrite=False):
        file_path = os.path.join(path, name + '.Rds')
        if not overwrite and (path / (name + '.Rds')).exists():
            logger.info(f"{name} already exists, skipping download")
            return
        logger.info(f"Downloading {name} to {path}")
        url = f"https://zenodo.org/records/6348128/files/{name}.Rds?download=1"

        download_file(url, file_path)
        logger.info("Download completed!")

        logger.info("Converting to python")
        subprocess.run(["Rscript", "mojitoo_data_to_py.R", name], check=True)


class SingleCellDataset(Dataset, ABC):
    def __init__(
            self,
            modalities: Iterable[str],
            root: Optional[str] = None,
            transform: Optional[List[Callable]] = None,
            pre_transform: Optional[Callable] = None,
            pre_filter: Optional[Callable] = None,
            log: bool = True,
            overwrite: bool = False,
            in_memory: bool = False,
            **kwargs):
        super().__init__(**kwargs)
        self.modalities = modalities
        self.root = root
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        self.log = log
        self.overwrite = overwrite
        self.in_memory = in_memory

        # Determine whether to download raw structures
        if not self.overwrite and all(
                os.path.exists(Path(self.root) / f"{m}.h5ad")
                for m in self.modalities
        ):
            logger.info(
                "All data already processed and overwrite=False. Skipping download."
            )
            self._skip_download = True
        else:
            self._skip_download = False

        super().__init__(root, transform, pre_transform, pre_filter, log)
        if self.in_memory:
            logger.info("Reading data into memory")
            self.data = [
                torch.load(pathlib.Path(self.root) / f"{m}.h5ad", weights_only=False)
                for m in tqdm(self.modalities)
            ]

    def download(self):
        raise NotImplementedError(
            "SingleCellDataset does not implement download()., please use DataModule to download instead")

    def process(self):
        pass

