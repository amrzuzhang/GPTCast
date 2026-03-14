from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

from lightning import LightningDataModule
from pandas import DataFrame
from torch.utils.data import DataLoader, Dataset

from gptcast.data.era5land_swvl1 import Era5LandSwvl1


class Era5LandSwvl1DataModule(LightningDataModule):
    """LightningDataModule for ERA5-Land swvl1 (soil moisture) forecasting.

    Goal: keep the same overall structure as the original MIARAD datamodule:
    - dataset returns dicts with keys: 'image', 'mask', 'file_path_'
    - supports `seq_len` + `stack_seq` to build a paper-style stacked input (v/h)

    Unlike MIARAD:
    - data is read from yearly NetCDF files under `base_dir`
    - train/val/test splits are provided explicitly via CSV files (avoid temporal leakage)
    """

    default_data_path = Path(__file__).parent.parent.parent.resolve() / "data"

    def __init__(
        self,
        *,
        base_dir: str,
        train_metadata_path_or_df: Union[str, DataFrame],
        val_metadata_path_or_df: Union[str, DataFrame],
        test_metadata_path_or_df: Union[str, DataFrame],
        clip_and_normalize: Tuple[float, float, float, float] = (0.0, 0.8, -1.0, 1.0),
        resize: Optional[Union[int, Tuple[int, int]]] = 256,
        crop: Optional[int] = 256,
        smart_crop: bool = False,
        max_mask_fraction: float = 0.0,
        smart_crop_attempts: int = 30,
        center_crop_val: bool = False,
        random_rotate90: bool = False,
        drop_incomplete: bool = True,
        max_open_years: int = 4,
        seq_len: int = 1,
        stack_seq: str = None,
        batch_size: int = 4,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self) -> None:
        # Dataset is assumed present locally (user provided).
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit") and self.data_train is None:
            self.data_train = Era5LandSwvl1(
                base_dir=self.hparams.base_dir,
                metadata_path_or_df=self.hparams.train_metadata_path_or_df,
                seq_len=self.hparams.seq_len,
                stack_seq=self.hparams.stack_seq,
                clip_and_normalize=self.hparams.clip_and_normalize,
                resize=self.hparams.resize,
                crop=self.hparams.crop,
                smart_crop=self.hparams.smart_crop,
                max_mask_fraction=self.hparams.max_mask_fraction,
                smart_crop_attempts=self.hparams.smart_crop_attempts,
                center_crop=False,
                random_rotate90=self.hparams.random_rotate90,
                drop_incomplete=self.hparams.drop_incomplete,
                max_open_years=self.hparams.max_open_years,
            )
            self.data_val = Era5LandSwvl1(
                base_dir=self.hparams.base_dir,
                metadata_path_or_df=self.hparams.val_metadata_path_or_df,
                seq_len=self.hparams.seq_len,
                stack_seq=self.hparams.stack_seq,
                clip_and_normalize=self.hparams.clip_and_normalize,
                resize=self.hparams.resize,
                crop=self.hparams.crop,
                smart_crop=False,  # keep validation simple/deterministic
                max_mask_fraction=self.hparams.max_mask_fraction,
                smart_crop_attempts=self.hparams.smart_crop_attempts,
                center_crop=self.hparams.center_crop_val,
                random_rotate90=False,  # keep validation deterministic
                drop_incomplete=self.hparams.drop_incomplete,
                max_open_years=self.hparams.max_open_years,
            )

        if stage in (None, "test") and self.data_test is None:
            self.data_test = Era5LandSwvl1(
                base_dir=self.hparams.base_dir,
                metadata_path_or_df=self.hparams.test_metadata_path_or_df,
                seq_len=self.hparams.seq_len,
                stack_seq=self.hparams.stack_seq,
                clip_and_normalize=self.hparams.clip_and_normalize,
                resize=self.hparams.resize,
                crop=self.hparams.crop,
                smart_crop=False,
                max_mask_fraction=self.hparams.max_mask_fraction,
                smart_crop_attempts=self.hparams.smart_crop_attempts,
                center_crop=self.hparams.center_crop_val,
                random_rotate90=False,
                drop_incomplete=self.hparams.drop_incomplete,
                max_open_years=self.hparams.max_open_years,
            )

    def train_dataloader(self) -> DataLoader:
        assert self.data_train is not None
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.data_val is not None
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        assert self.data_test is not None
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
