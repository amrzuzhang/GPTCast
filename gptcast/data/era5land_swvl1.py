from __future__ import annotations

import ast
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple, Union

import albumentations as A
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import xarray as xr
from torch.utils.data import Dataset


@dataclass(frozen=True)
class _YearCacheEntry:
    ds: xr.Dataset
    time_index: pd.DatetimeIndex


class Era5LandSwvl1(Dataset):
    """ERA5-Land volumetric soil water layer 1 (swvl1) dataset.

    This dataset is designed to mirror the MIARAD dataset interface as much as possible:
    - metadata is provided as a CSV with a `files` column containing a python-list-as-string
      (parsed via `ast.literal_eval`)
    - each element in `files` is a string key like: `era5land/SM_YYYYMMDDHHMM`

    Unlike MIARAD (which reads gzipped NetCDF from a tar), ERA5-Land is stored as one NetCDF per year:
      <base_dir>/<year>/volumetric_soil_water_layer_1.nc

    Missing values:
    - In this dataset layout, swvl1 is NaN over ocean for all timesteps (static spatial mask).
      We expose this as `mask` (True where invalid/ocean).
    - We do *not* time-interpolate. Ocean values are filled with cmin (typically 0) before scaling.
    """

    # The basename part of the key format used in the generated CSVs.
    date_fmt = "SM_%Y%m%d%H%M"

    @staticmethod
    def parse_metadata(csv_path: str) -> pd.DataFrame:
        return pd.read_csv(csv_path, index_col="id", converters={"files": ast.literal_eval})

    def __init__(
        self,
        *,
        base_dir: str,
        metadata_path_or_df: Union[str, pd.DataFrame],
        seq_len: int = 1,
        stack_seq: Optional[str] = None,
        clip_and_normalize: Optional[Tuple[float, float, float, float]] = (0.0, 0.8, -1.0, 1.0),
        resize: Optional[Union[int, Tuple[int, int]]] = 256,
        crop: Optional[int] = 256,
        smart_crop: bool = False,
        max_mask_fraction: float = 0.0,
        smart_crop_attempts: int = 30,
        center_crop: bool = False,
        random_rotate90: bool = False,
        drop_incomplete: bool = True,
        max_open_years: int = 4,
        var_name: str = "swvl1",
        yearly_filename: str = "volumetric_soil_water_layer_1.nc",
    ) -> None:
        super().__init__()
        assert seq_len >= 1
        if stack_seq is not None and stack_seq not in {"v", "h"}:
            raise ValueError("stack_seq must be one of {None, 'v', 'h'}")
        if max_open_years < 1:
            raise ValueError("max_open_years must be >= 1")

        self.base_dir = Path(base_dir)
        self.var_name = var_name
        self.yearly_filename = yearly_filename

        if isinstance(metadata_path_or_df, str):
            self.meta = self.parse_metadata(metadata_path_or_df)
        else:
            self.meta = metadata_path_or_df

        self.seq_len = int(seq_len)
        self.stack = stack_seq
        self.clip_and_normalize = clip_and_normalize
        self.resize = resize
        self.crop = crop
        self.smart_crop = bool(smart_crop)
        self.max_mask_fraction = float(max_mask_fraction)
        self.smart_crop_attempts = int(smart_crop_attempts)
        self.center_crop = bool(center_crop)
        self.random_rotate90 = bool(random_rotate90)
        self.drop_incomplete = bool(drop_incomplete)
        self.max_open_years = int(max_open_years)

        self._files: list[str] = []
        for flist in self.meta["files"]:
            self._files.extend(flist)

        # Keep the ordering from the CSV. The CSV we generated is chronological, which is what we want.
        self._timestamps: list[datetime] = [
            datetime.strptime(Path(f).name, self.date_fmt) for f in self._files
        ]

        # For seq_len>1 we must avoid windows that straddle time discontinuities.
        # The metadata CSV can have multiple rows (yearly or pre-segmented) and we flatten the
        # `files` lists; therefore we precompute valid start indices where *all* steps are
        # spaced by exactly 1 day.
        self._start_indices: list[int]
        if self.seq_len <= 1:
            self._start_indices = list(range(len(self._files)))
        else:
            expected_gap = timedelta(days=1)
            n = len(self._timestamps)
            max_start = n - self.seq_len
            valid: list[int] = []
            for start in range(0, max_start + 1):
                ok = True
                for j in range(self.seq_len - 1):
                    if (self._timestamps[start + j + 1] - self._timestamps[start + j]) != expected_gap:
                        ok = False
                        break
                if ok:
                    valid.append(start)
            self._start_indices = valid

        self._len = len(self._start_indices)

        # Lazily initialized per-process (DataLoader workers get their own dataset instance).
        self._ocean_mask_native: Optional[np.ndarray] = None  # bool (H, W), True for ocean/invalid
        self._year_cache: "OrderedDict[int, _YearCacheEntry]" = OrderedDict()

        # Transforms are applied after resizing and stacking along channel dimension (H,W,C).
        self._transform = None
        if self.crop is not None:
            crop_op = (
                A.CenterCrop(width=self.crop, height=self.crop)
                if self.center_crop
                else A.RandomCrop(width=self.crop, height=self.crop)
            )
            ops = [crop_op]
            if self.random_rotate90:
                ops.append(A.RandomRotate90())
            self._transform = A.Compose(ops)

    def __len__(self) -> int:
        return self._len

    def _nc_path(self, year: int) -> Path:
        return self.base_dir / str(year) / self.yearly_filename

    def _get_year_entry(self, year: int) -> _YearCacheEntry:
        # LRU cache via OrderedDict.
        if year in self._year_cache:
            entry = self._year_cache.pop(year)
            self._year_cache[year] = entry
            return entry

        # Evict if needed.
        while len(self._year_cache) >= self.max_open_years:
            _, old = self._year_cache.popitem(last=False)
            try:
                old.ds.close()
            except Exception:
                pass

        ds = xr.open_dataset(self._nc_path(year))
        # Convert to pandas index for fast get_loc.
        time_index = pd.to_datetime(ds["time"].values)
        entry = _YearCacheEntry(ds=ds, time_index=time_index)
        self._year_cache[year] = entry
        return entry

    def _init_ocean_mask(self, dt: datetime) -> None:
        if self._ocean_mask_native is not None:
            return
        entry = self._get_year_entry(dt.year)
        try:
            tidx = int(entry.time_index.get_loc(pd.Timestamp(dt)))
        except KeyError:
            # Should not happen with our generated CSVs, but keep it robust.
            tidx = int(entry.time_index.get_indexer([pd.Timestamp(dt)], method="nearest")[0])

        arr = entry.ds[self.var_name].isel(time=tidx).values  # (lat, lon)
        self._ocean_mask_native = np.isnan(arr)

    def _read_frame_native(self, dt: datetime) -> np.ndarray:
        entry = self._get_year_entry(dt.year)
        try:
            tidx = int(entry.time_index.get_loc(pd.Timestamp(dt)))
        except KeyError:
            tidx = int(entry.time_index.get_indexer([pd.Timestamp(dt)], method="nearest")[0])
        arr = entry.ds[self.var_name].isel(time=tidx).values
        return arr.astype(np.float32, copy=False)

    @staticmethod
    def _stack_channels_to_single(image_hwc: np.ndarray, stack: str) -> np.ndarray:
        # image_hwc: H x W x C
        c = image_hwc.shape[2]
        chw = np.transpose(image_hwc, (2, 0, 1))  # C,H,W
        slices = list(chw)  # list of HxW
        if stack == "v":
            out = np.concatenate(slices, axis=0)  # (C*H, W)
        elif stack == "h":
            out = np.concatenate(slices, axis=1)  # (H, C*W)
        else:
            raise ValueError(f"Invalid stack mode: {stack}")
        return out[:, :, None]  # add channel dim

    @staticmethod
    def _stack_mask(mask_hw: np.ndarray, c: int, stack: str) -> np.ndarray:
        if c <= 1:
            return mask_hw
        if stack == "v":
            return np.concatenate([mask_hw] * c, axis=0)
        if stack == "h":
            return np.concatenate([mask_hw] * c, axis=1)
        raise ValueError(f"Invalid stack mode: {stack}")

    def __getitem__(self, i: int) -> dict:
        if i < 0 or i >= self._len:
            raise IndexError(i)

        start = self._start_indices[i]
        t0 = self._timestamps[start]
        self._init_ocean_mask(t0)
        assert self._ocean_mask_native is not None

        # Read a sequence as (S, H, W) in native resolution.
        if self.seq_len == 1:
            frames = [self._read_frame_native(t0)]
        else:
            frames = [self._read_frame_native(self._timestamps[start + j]) for j in range(self.seq_len)]
        samples = np.stack(frames, axis=0)  # (S, H, W)

        mask = self._ocean_mask_native.copy()  # (H, W) bool

        # Fill missing/ocean with cmin before clipping/scaling.
        if self.clip_and_normalize is not None:
            cmin, cmax, nmin, nmax = self.clip_and_normalize
            samples = np.nan_to_num(samples, nan=float(cmin))
            samples = np.clip(samples, float(cmin), float(cmax))
            samples = (samples - float(cmin)) / (float(cmax) - float(cmin) + 1e-12)
            if float(nmin) != 0.0 or float(nmax) != 1.0:
                samples = samples * (float(nmax) - float(nmin)) + float(nmin)
        else:
            samples = np.nan_to_num(samples, nan=0.0)

        # Resize (optionally) so we can reuse the original model configs (paper-style crops).
        if self.resize is not None:
            if isinstance(self.resize, int):
                resize_hw = (self.resize, self.resize)
            else:
                resize_hw = (int(self.resize[0]), int(self.resize[1]))
            x = torch.from_numpy(samples).unsqueeze(1)  # (S,1,H,W)
            x = F.interpolate(x, size=resize_hw, mode="bilinear", align_corners=False)
            samples = x.squeeze(1).numpy()

            m = torch.from_numpy(mask.astype(np.float32))[None, None, ...]
            m = F.interpolate(m, size=resize_hw, mode="nearest")
            mask = m[0, 0].numpy().astype(bool)

        # Convert to HWC.
        image = np.transpose(samples, (1, 2, 0))  # (H, W, S)

        # Optional crop/rotate (applies consistently across all channels).
        if self._transform is not None:
            mask_u8 = mask.astype(np.uint8)
            if self.smart_crop:
                assert self.crop is not None
                best = None
                best_frac = float("inf")
                for _ in range(max(1, self.smart_crop_attempts)):
                    t = self._transform(image=image, mask=mask_u8)
                    m = t["mask"]
                    frac = float(m.mean()) if m.size else 1.0
                    if frac < best_frac:
                        best = t
                        best_frac = frac
                    if frac <= self.max_mask_fraction:
                        best = t
                        best_frac = frac
                        break
                if best is None:
                    # Should never happen, but keep the failure mode clear.
                    best = self._transform(image=image, mask=mask_u8)

                image = best["image"]
                mask = best["mask"].astype(bool)
            else:
                t = self._transform(image=image, mask=mask_u8)
                image = t["image"]
                mask = t["mask"].astype(bool)

        # Stack temporal channels into a single-channel big image (same trick as in the paper).
        if self.stack is not None:
            c = image.shape[2]
            image = self._stack_channels_to_single(image, self.stack)  # (H',W',1)
            mask = self._stack_mask(mask, c=c, stack=self.stack)  # (H',W')
        else:
            # Keep multi-channel (H,W,S) output. This is useful for notebook-style
            # visualization/verification, and mirrors MIARAD's `MiaradN` behaviour.
            pass

        example: dict = {
            "image": image.astype(np.float32, copy=False),
            "mask": mask,
            "file_path_": t0.strftime(self.date_fmt),
        }
        return example
