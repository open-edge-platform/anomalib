from pathlib import Path

import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from torchvision.transforms.v2 import Transform

from anomalib.data.datasets.base.image import AnomalibDataset
from anomalib.data.errors import MisMatchError
from anomalib.data.utils import LabelName, Split, validate_path

CATEGORIES = ("Brain", "Chest", "Histopathology", "Liver", "Retina_OCT2017", "Retina_RESC")

class BMADDataset(AnomalibDataset):
    def __init__(
            self,
            root: str | Path,
            category: str,
            augmentations: Transform | None = None,
            split: str | Split | None = None
    ) -> None:
        super().__init__(augmentations=augmentations)

        self.root_category = Path(root) / category
        self.split = split
        self.samples = make_bmad_dataset(path=self.root_category, split=self.split)


def make_bmad_dataset(path: Path, split: str | Split | None = None) -> DataFrame:
    path = validate_path(path)

    samples_list = [
        (
            (str(path), *filename.parts[-4:]) 
            if filename.parts[-3] != "train"
            else (str(path), *filename.parts[-3:-1], "", filename.parts[-1])
        )
        for filename in path.glob("**/*")
        if filename.suffix in {".png", ".PNG"}
    ]

    samples = pd.DataFrame(samples_list, columns=["path", "split", "label", "temp", "image_path"])

    samples["image_path"] = (
        samples.path + "/" 
        + samples.split + "/" 
        + samples.label + "/" 
        + np.where(samples.temp != "", samples.temp + "/", "") 
        + samples.image_path
    )

    samples.loc[(samples.label == "good"), "label_index"] = LabelName.NORMAL
    samples.loc[(samples.label != "good"), "label_index"] = LabelName.ABNORMAL
    samples.label_index = samples.label_index.astype(int)

    mask_samples = samples.loc[samples.temp == "label"].sort_values(
        by="image_path",
        ignore_index=True,
    )
    samples = samples[samples.temp != "label"].sort_values(
        by="image_path",
        ignore_index=True,
    )

    samples["mask_path"] = None
    if len(mask_samples):
        samples.loc[
            (samples.split == "test") | (samples.split == "valid"),
            "mask_path",
        ] = mask_samples.image_path.to_numpy()

    if len(mask_samples):
        abnormal_samples = samples.loc[samples.label_index == LabelName.ABNORMAL]
        if (
            len(abnormal_samples)
            and not abnormal_samples.apply(
                lambda x: Path(x.image_path).stem in Path(x.mask_path).stem,
                axis=1,
            ).all()
        ):
            msg = (
                "Mismatch between anomalous images and ground truth masks. Make sure "
                "mask files in 'ground_truth' folder follow the same naming "
                "convention as the anomalous images (e.g. image: '000.png', "
                "mask: '000.png' or '000_mask.png')."
            )
            raise MisMatchError(msg)

    samples.attrs["task"] = "classification" if (samples["mask_path"] == "").all() else "segmentation"
    split = "train"
    if split:
        samples = samples[samples.split == split].reset_index(drop=True)

    return samples
