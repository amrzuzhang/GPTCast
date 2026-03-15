from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

from lightning import LightningDataModule
from pandas import DataFrame
from torch.utils.data import DataLoader, Dataset

from gptcast.data.ecmwf_s2s_download import ensure_ecmwf_guidance
from gptcast.data.era5land_hydro import Era5LandHydro
from gptcast.data.guidance_ecmwf import default_guidance_root, derive_candidate_init_dates


class Era5LandHydroDataModule(LightningDataModule):
    """Generic LightningDataModule for ERA5-Land state/forcing experiments."""

    default_data_path = Path(__file__).parent.parent.parent.resolve() / "data"

    def __init__(
        self,
        *,
        base_dir: str,
        train_metadata_path_or_df: Union[str, DataFrame],
        val_metadata_path_or_df: Union[str, DataFrame],
        test_metadata_path_or_df: Union[str, DataFrame],
        image_variable_key: str = "swvl1",
        forcing_variable_keys: Optional[Sequence[str]] = None,
        guidance_variable_keys: Optional[Sequence[str]] = None,
        guidance_dir: Optional[str] = None,
        guidance_target_offset: Optional[int] = None,
        guidance_auto_download: bool = False,
        guidance_download_metadata_paths: Optional[Sequence[str]] = None,
        guidance_weekdays: Optional[Sequence[int]] = None,
        guidance_min_year: int = 1995,
        guidance_lead_days: Optional[Sequence[int]] = None,
        guidance_area: Optional[Sequence[float]] = None,
        guidance_grid: str = "1.5/1.5",
        normalize_forcing: bool = False,
        clip_and_normalize: Optional[Tuple[float, float, float, float]] = None,
        resize: Optional[Union[int, Tuple[int, int]]] = 256,
        crop: Optional[int] = 256,
        smart_crop: bool = False,
        max_mask_fraction: float = 0.0,
        smart_crop_attempts: int = 30,
        center_crop_val: bool = False,
        random_rotate90: bool = False,
        drop_incomplete: bool = True,
        max_open_years: int = 8,
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
        if not self.hparams.guidance_auto_download or not self.hparams.guidance_variable_keys:
            return

        guidance_dir = (
            default_guidance_root(self.hparams.base_dir)
            if self.hparams.guidance_dir is None
            else Path(self.hparams.guidance_dir)
        )
        metadata_sources = list(self.hparams.guidance_download_metadata_paths or [])
        if not metadata_sources:
            metadata_sources = [
                self.hparams.train_metadata_path_or_df,
                self.hparams.val_metadata_path_or_df,
                self.hparams.test_metadata_path_or_df,
            ]

        init_dates = derive_candidate_init_dates(
            metadata_sources,
            weekdays=self.hparams.guidance_weekdays or (0, 3),
            min_year=int(self.hparams.guidance_min_year),
        )
        if not init_dates:
            return

        ensure_ecmwf_guidance(
            variables=self.hparams.guidance_variable_keys,
            init_dates=init_dates,
            out_root=guidance_dir,
            lead_days=self.hparams.guidance_lead_days or [max(1, int(self.hparams.guidance_target_offset or self.hparams.seq_len - 1))],
            area=self.hparams.guidance_area or (42.0, 105.0, 20.0, 125.0),
            grid=str(self.hparams.guidance_grid),
            overwrite=False,
        )

    def _build_dataset(self, metadata_path_or_df, *, center_crop: bool, smart_crop: bool, random_rotate90: bool):
        return Era5LandHydro(
            base_dir=self.hparams.base_dir,
            metadata_path_or_df=metadata_path_or_df,
            image_variable_key=self.hparams.image_variable_key,
            forcing_variable_keys=self.hparams.forcing_variable_keys,
            guidance_variable_keys=self.hparams.guidance_variable_keys,
            guidance_dir=self.hparams.guidance_dir,
            guidance_target_offset=self.hparams.guidance_target_offset,
            normalize_forcing=self.hparams.normalize_forcing,
            seq_len=self.hparams.seq_len,
            stack_seq=self.hparams.stack_seq,
            clip_and_normalize=self.hparams.clip_and_normalize,
            resize=self.hparams.resize,
            crop=self.hparams.crop,
            smart_crop=smart_crop,
            max_mask_fraction=self.hparams.max_mask_fraction,
            smart_crop_attempts=self.hparams.smart_crop_attempts,
            center_crop=center_crop,
            random_rotate90=random_rotate90,
            drop_incomplete=self.hparams.drop_incomplete,
            max_open_years=self.hparams.max_open_years,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit") and self.data_train is None:
            self.data_train = self._build_dataset(
                self.hparams.train_metadata_path_or_df,
                center_crop=False,
                smart_crop=self.hparams.smart_crop,
                random_rotate90=self.hparams.random_rotate90,
            )
            self.data_val = self._build_dataset(
                self.hparams.val_metadata_path_or_df,
                center_crop=self.hparams.center_crop_val,
                smart_crop=False,
                random_rotate90=False,
            )

        if stage in (None, "test") and self.data_test is None:
            self.data_test = self._build_dataset(
                self.hparams.test_metadata_path_or_df,
                center_crop=self.hparams.center_crop_val,
                smart_crop=False,
                random_rotate90=False,
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
