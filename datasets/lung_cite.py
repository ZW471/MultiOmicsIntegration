import os
from pathlib import Path
from typing import Iterable, Optional, Callable

from datasets.base import MojitooDataModule, SingleCellDataset


class LungCiteDataModule(MojitooDataModule):

    def __init__(
            self,
            path: str,
            batch_size: int,
            modalities: Iterable[str],
            pin_memory: bool = True,
            in_memory: bool = False,
            num_workers: int = 16,
            dataset_fraction: float = 1.0,
            transforms: Optional[Iterable[Callable]] = None,
            overwrite: bool = False,
    ):
        super().__init__()

        self.modalities = modalities
        for m in self.modalities:
            assert m in ["RNA", "ADT"], f"Invalid modality: {m}"
        self.data_dir = Path(path)
        self.raw_dir = self.data_dir
        self.processed_dir = self.data_dir / "processed" / "lung_cite"
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        if transforms is not None:
            # self.transform = self.compose_transforms(
            #     omegaconf.OmegaConf.to_container(transforms, resolve=True)
            # )
            raise NotImplementedError("transforms not implemented yet")  # TODO
        else:
            self.transform = None

        self.in_memory = in_memory
        self.overwrite = overwrite

        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.format = format

        self.dataset_fraction = dataset_fraction

        self.prepare_data_per_node = False

    def download(self):
        self._download("LUNG-CITE", self.raw_dir, self.overwrite)

    def train_dataset(self):
        return SingleCellDataset(
            modalities=self.modalities,
            root=str(self.processed_dir),
            transform=self.transform,
            log=True,
            overwrite=self.overwrite,
            in_memory=self.in_memory,
        )