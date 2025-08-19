import logging
from pathlib import Path

from torchvision.transforms.v2 import Transform

from anomalib.data.datamodules.base.image import AnomalibDataModule
from anomalib.data.datasets.image.bmad import BMADDataset
from anomalib.data.utils import DownloadInfo, Split, TestSplitMode, ValSplitMode, download_and_extract
from anomalib.utils import deprecate

logger = logging.getLogger(__name__)

DOWNLOAD_INFO = DownloadInfo(
    name="mvtecad",
    url="https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/"
    "download/420938113-1629952094/mvtec_anomaly_detection.tar.xz",
    hashsum="cf4313b13603bec67abb49ca959488f7eedce2a9f7795ec54446c649ac98cd3d",
) # TODO

class BMAD(AnomalibDataModule):
    def __init__(
        self,
        root: Path | str = "./datasets/BMAD",
        category: str = "Brain",
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 8,
        train_augmentations: Transform | None = None,
        val_augmentations: Transform | None = None,
        test_augmentations: Transform | None = None,
        augmentations: Transform | None = None,
        test_split_mode: TestSplitMode | str = TestSplitMode.FROM_DIR,
        test_split_ratio: float = 0.2,
        val_split_mode: ValSplitMode | str = ValSplitMode.SAME_AS_TEST,
        val_split_ratio: float = 0.5,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            train_augmentations=train_augmentations,
            val_augmentations=val_augmentations,
            test_augmentations=test_augmentations,
            augmentations=augmentations,
            test_split_mode=test_split_mode,
            test_split_ratio=test_split_ratio,
            val_split_mode=val_split_mode,
            val_split_ratio=val_split_ratio,
            seed=seed,
        )

        self.root = Path(root)
        self.category = category

    def _setup(self, _stage: str | None = None) -> None:
        self.train_data = BMADDataset(
            split=Split.TRAIN,
            root=self.root,
            category=self.category,
        )
        self.validation_data = BMADDataset(
            split="valid",
            root=self.root,
            category=self.category
        )
        self.test_data = BMADDataset(
            split=Split.TEST,
            root=self.root,
            category=self.category,
        )

    def prepare_data(self):
        if (self.root / self.category).is_dir():
            logger.info("Found the dataset.")
        else:
            download_and_extract(self.root, DOWNLOAD_INFO)
