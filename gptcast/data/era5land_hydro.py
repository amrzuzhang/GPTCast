from __future__ import annotations

import ast
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import xarray as xr
from torch.utils.data import Dataset

from gptcast.data.guidance_ecmwf import ECMWF_GUIDANCE_VARIABLES, default_guidance_root


@dataclass(frozen=True)
class _YearCacheEntry:
    ds: xr.Dataset
    time_index: pd.DatetimeIndex


@dataclass(frozen=True)
class _VariableSpec:
    key: str
    category: str
    filename: str
    var_name: str
    clip_and_normalize: Optional[Tuple[float, float, float, float]] = None
    derived: bool = False
    dependencies: Tuple[str, ...] = ()


@dataclass(frozen=True)
class _GuidanceSpec:
    key: str
    provider: str
    var_name: str
    clip_and_normalize: Optional[Tuple[float, float, float, float]] = None


_LAYER_THICKNESS_M = {
    "swvl1": 0.07,
    "swvl2": 0.21,
    "swvl3": 0.72,
    "swvl4": 1.89,
}


ERA5LAND_HYDRO_VARIABLES: dict[str, _VariableSpec] = {
    "swvl1": _VariableSpec(
        key="swvl1",
        category="land_surface",
        filename="volumetric_soil_water_layer_1.nc",
        var_name="swvl1",
        clip_and_normalize=(0.0, 0.8, -1.0, 1.0),
    ),
    "swvl2": _VariableSpec(
        key="swvl2",
        category="land_surface",
        filename="volumetric_soil_water_layer_2.nc",
        var_name="swvl2",
        clip_and_normalize=(0.0, 0.8, -1.0, 1.0),
    ),
    "swvl3": _VariableSpec(
        key="swvl3",
        category="land_surface",
        filename="volumetric_soil_water_layer_3.nc",
        var_name="swvl3",
        clip_and_normalize=(0.0, 0.8, -1.0, 1.0),
    ),
    "swvl4": _VariableSpec(
        key="swvl4",
        category="land_surface",
        filename="volumetric_soil_water_layer_4.nc",
        var_name="swvl4",
        clip_and_normalize=(0.0, 0.8, -1.0, 1.0),
    ),
    "rzsm_0_100cm": _VariableSpec(
        key="rzsm_0_100cm",
        category="land_surface",
        filename="root_zone_soil_moisture_0_100cm.nc",
        var_name="rzsm_0_100cm",
        clip_and_normalize=(0.0, 0.8, -1.0, 1.0),
        derived=True,
        dependencies=("swvl1", "swvl2", "swvl3"),
    ),
    "soil_water_storage_0_100cm_mm": _VariableSpec(
        key="soil_water_storage_0_100cm_mm",
        category="land_surface",
        filename="soil_water_storage_0_100cm_mm.nc",
        var_name="soil_water_storage_0_100cm_mm",
        clip_and_normalize=(0.0, 400.0, -1.0, 1.0),
        derived=True,
        dependencies=("swvl1", "swvl2", "swvl3"),
    ),
    "precipitation": _VariableSpec(
        key="precipitation",
        category="atmosphere",
        filename="precipitation.nc",
        var_name="precipitation",
        clip_and_normalize=(0.0, 0.05, -1.0, 1.0),
    ),
    "evapotranspiration": _VariableSpec(
        key="evapotranspiration",
        category="land_surface",
        filename="evapotranspiration.nc",
        var_name="evapotranspiration",
        clip_and_normalize=(0.0, 0.005, -1.0, 1.0),
    ),
    "runoff": _VariableSpec(
        key="runoff",
        category="land_surface",
        filename="runoff.nc",
        var_name="runoff",
        clip_and_normalize=(0.0, 0.05, -1.0, 1.0),
        derived=True,
        dependencies=("surface_runoff", "sub_surface_runoff"),
    ),
    "surface_runoff": _VariableSpec(
        key="surface_runoff",
        category="land_surface",
        filename="surface_runoff.nc",
        var_name="surface_runoff",
        clip_and_normalize=(0.0, 0.05, -1.0, 1.0),
    ),
    "sub_surface_runoff": _VariableSpec(
        key="sub_surface_runoff",
        category="land_surface",
        filename="sub_surface_runoff.nc",
        var_name="sub_surface_runoff",
        clip_and_normalize=(0.0, 0.05, -1.0, 1.0),
    ),
    "2m_temperature": _VariableSpec(
        key="2m_temperature",
        category="atmosphere",
        filename="2m_temperature.nc",
        var_name="t2m",
        clip_and_normalize=(240.0, 330.0, -1.0, 1.0),
    ),
    "specific_humidity": _VariableSpec(
        key="specific_humidity",
        category="atmosphere",
        filename="specific_humidity.nc",
        var_name="specific_humidity",
        clip_and_normalize=(0.0, 0.03, -1.0, 1.0),
    ),
    "surface_pressure": _VariableSpec(
        key="surface_pressure",
        category="atmosphere",
        filename="surface_pressure.nc",
        var_name="sp",
        clip_and_normalize=(70000.0, 105000.0, -1.0, 1.0),
    ),
    "10m_u_component_of_wind": _VariableSpec(
        key="10m_u_component_of_wind",
        category="atmosphere",
        filename="10m_u_component_of_wind.nc",
        var_name="u10",
        clip_and_normalize=(-25.0, 25.0, -1.0, 1.0),
    ),
    "10m_v_component_of_wind": _VariableSpec(
        key="10m_v_component_of_wind",
        category="atmosphere",
        filename="10m_v_component_of_wind.nc",
        var_name="v10",
        clip_and_normalize=(-25.0, 25.0, -1.0, 1.0),
    ),
    "surface_solar_radiation_downwards_w_m2": _VariableSpec(
        key="surface_solar_radiation_downwards_w_m2",
        category="land_surface",
        filename="surface_solar_radiation_downwards_w_m2.nc",
        var_name="ssrd_w_m2",
        clip_and_normalize=(0.0, 1400.0, -1.0, 1.0),
    ),
    "surface_thermal_radiation_downwards_w_m2": _VariableSpec(
        key="surface_thermal_radiation_downwards_w_m2",
        category="land_surface",
        filename="surface_thermal_radiation_downwards_w_m2.nc",
        var_name="strd_w_m2",
        clip_and_normalize=(0.0, 700.0, -1.0, 1.0),
    ),
}


ECMWF_GUIDANCE_SPECS: dict[str, _GuidanceSpec] = {
    key: _GuidanceSpec(
        key=spec.key,
        provider="ecmwf",
        var_name=spec.out_var,
        clip_and_normalize=spec.clip_and_normalize,
    )
    for key, spec in ECMWF_GUIDANCE_VARIABLES.items()
}


class Era5LandHydro(Dataset):
    """Generic ERA5-Land hydro dataset with one main state field and optional forcing fields."""

    date_fmt = "SM_%Y%m%d%H%M"

    @staticmethod
    def parse_metadata(csv_path: str) -> pd.DataFrame:
        return pd.read_csv(csv_path, index_col="id", converters={"files": ast.literal_eval})

    def __init__(
        self,
        *,
        base_dir: str,
        metadata_path_or_df: Union[str, pd.DataFrame],
        image_variable_key: str = "swvl1",
        forcing_variable_keys: Optional[Sequence[str]] = None,
        guidance_variable_keys: Optional[Sequence[str]] = None,
        guidance_dir: Optional[str] = None,
        guidance_target_offset: Optional[int] = None,
        normalize_forcing: bool = False,
        seq_len: int = 1,
        stack_seq: Optional[str] = None,
        clip_and_normalize: Optional[Tuple[float, float, float, float]] = None,
        resize: Optional[Union[int, Tuple[int, int]]] = 256,
        crop: Optional[int] = 256,
        smart_crop: bool = False,
        max_mask_fraction: float = 0.0,
        smart_crop_attempts: int = 30,
        center_crop: bool = False,
        random_rotate90: bool = False,
        drop_incomplete: bool = True,
        max_open_years: int = 8,
    ) -> None:
        super().__init__()
        assert seq_len >= 1
        if stack_seq is not None and stack_seq not in {"v", "h"}:
            raise ValueError("stack_seq must be one of {None, 'v', 'h'}")
        if max_open_years < 1:
            raise ValueError("max_open_years must be >= 1")

        self.base_dir = Path(base_dir)
        self.image_spec = self._get_spec(image_variable_key)
        self.forcing_specs = [self._get_spec(k) for k in (forcing_variable_keys or [])]
        self.guidance_specs = [self._get_guidance_spec(k) for k in (guidance_variable_keys or [])]
        self.guidance_dir = (
            default_guidance_root(self.base_dir) if guidance_dir is None else Path(guidance_dir)
        )
        self.guidance_target_offset = guidance_target_offset
        self.normalize_forcing = bool(normalize_forcing)

        if isinstance(metadata_path_or_df, str):
            self.meta = self.parse_metadata(metadata_path_or_df)
        else:
            self.meta = metadata_path_or_df

        self.seq_len = int(seq_len)
        self.stack = stack_seq
        self.clip_and_normalize = (
            self.image_spec.clip_and_normalize if clip_and_normalize is None else clip_and_normalize
        )
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
        self._timestamps = [datetime.strptime(Path(f).name, self.date_fmt) for f in self._files]

        if self.seq_len <= 1:
            self._start_indices = list(range(len(self._files)))
        else:
            expected_gap = timedelta(days=1)
            n = len(self._timestamps)
            max_start = n - self.seq_len
            valid = []
            for start in range(0, max_start + 1):
                if all(
                    (self._timestamps[start + j + 1] - self._timestamps[start + j]) == expected_gap
                    for j in range(self.seq_len - 1)
                ):
                    valid.append(start)
            self._start_indices = valid

        if self.guidance_specs:
            self._start_indices = [idx for idx in self._start_indices if self._has_all_guidance(idx)]

        self._len = len(self._start_indices)
        self._image_mask_native: Optional[np.ndarray] = None
        self._year_cache: "OrderedDict[tuple[str, int, str], _YearCacheEntry]" = OrderedDict()
        self._guidance_cache: "OrderedDict[Path, _YearCacheEntry]" = OrderedDict()

    @staticmethod
    def _get_spec(key: str) -> _VariableSpec:
        if key not in ERA5LAND_HYDRO_VARIABLES:
            raise KeyError(
                f"Unknown ERA5-Land hydro variable key: {key!r}. "
                f"Available keys: {sorted(ERA5LAND_HYDRO_VARIABLES)}"
            )
        return ERA5LAND_HYDRO_VARIABLES[key]

    @staticmethod
    def _get_guidance_spec(key: str) -> _GuidanceSpec:
        if key not in ECMWF_GUIDANCE_SPECS:
            raise KeyError(
                f"Unknown guidance variable key: {key!r}. "
                f"Available keys: {sorted(ECMWF_GUIDANCE_SPECS)}"
            )
        return ECMWF_GUIDANCE_SPECS[key]

    def __len__(self) -> int:
        return self._len

    def _file_path(self, spec: _VariableSpec, year: int) -> Path:
        return self.base_dir / spec.category / str(year) / spec.filename

    def _get_year_entry(self, spec: _VariableSpec, year: int) -> _YearCacheEntry:
        cache_key = (spec.key, year, spec.filename)
        if cache_key in self._year_cache:
            entry = self._year_cache.pop(cache_key)
            self._year_cache[cache_key] = entry
            return entry

        while len(self._year_cache) >= self.max_open_years:
            _, old = self._year_cache.popitem(last=False)
            try:
                old.ds.close()
            except Exception:
                pass

        file_path = self._file_path(spec, year)
        if not file_path.exists():
            raise FileNotFoundError(
                f"Missing ERA5-Land file for variable {spec.key!r}: {file_path}\n"
                "If you want to use root-zone or forcing variables, rerun the hydrology downloader first."
            )

        ds = xr.open_dataset(file_path)
        time_index = pd.to_datetime(ds["time"].values)
        entry = _YearCacheEntry(ds=ds, time_index=time_index)
        self._year_cache[cache_key] = entry
        return entry

    def _guidance_file_path(self, spec: _GuidanceSpec, dt: datetime) -> Path:
        return self.guidance_dir / spec.key / f"{dt.year:04d}" / f"{dt:%Y%m%d}.nc"

    def _get_guidance_entry(self, spec: _GuidanceSpec, dt: datetime) -> _YearCacheEntry:
        file_path = self._guidance_file_path(spec, dt)
        if file_path in self._guidance_cache:
            entry = self._guidance_cache.pop(file_path)
            self._guidance_cache[file_path] = entry
            return entry

        while len(self._guidance_cache) >= self.max_open_years:
            _, old = self._guidance_cache.popitem(last=False)
            try:
                old.ds.close()
            except Exception:
                pass

        if not file_path.exists():
            raise FileNotFoundError(
                f"Missing guidance file for variable {spec.key!r}: {file_path}\n"
                "Download the requested ECMWF guidance files first, or enable guidance auto-download."
            )

        ds = xr.open_dataset(file_path)
        if "lead_day" in ds.coords:
            time_index = pd.Index(ds["lead_day"].values)
        elif "lead_day" in ds.dims:
            time_index = pd.Index(ds["lead_day"].values)
        else:
            raise RuntimeError(f"Guidance file does not contain lead_day coordinate: {file_path}")

        entry = _YearCacheEntry(ds=ds, time_index=time_index)
        self._guidance_cache[file_path] = entry
        return entry

    def _get_guidance_target_offset(self) -> int:
        if self.guidance_target_offset is None:
            return max(0, int(self.seq_len) - 1)
        offset = int(self.guidance_target_offset)
        if offset < 0 or offset >= int(self.seq_len):
            raise ValueError(
                f"guidance_target_offset must be in [0, seq_len-1], got {offset} for seq_len={self.seq_len}"
            )
        return offset

    def _has_all_guidance(self, start_idx: int) -> bool:
        init_dt = self._timestamps[start_idx]
        return all(self._guidance_file_path(spec, init_dt).exists() for spec in self.guidance_specs)

    def _read_frame_native(self, spec: _VariableSpec, dt: datetime) -> np.ndarray:
        if spec.derived:
            return self._read_derived_frame_native(spec, dt)

        entry = self._get_year_entry(spec, dt.year)
        try:
            tidx = int(entry.time_index.get_loc(pd.Timestamp(dt)))
        except KeyError:
            tidx = int(entry.time_index.get_indexer([pd.Timestamp(dt)], method="nearest")[0])
        arr = entry.ds[spec.var_name].isel(time=tidx).values
        return arr.astype(np.float32, copy=False)

    def _read_guidance_frame_native(self, spec: _GuidanceSpec, init_dt: datetime, lead_day: int) -> np.ndarray:
        entry = self._get_guidance_entry(spec, init_dt)
        try:
            lead_idx = int(entry.time_index.get_loc(int(lead_day)))
        except KeyError:
            lead_idx = int(entry.time_index.get_indexer([int(lead_day)])[0])
            if lead_idx < 0:
                raise KeyError(
                    f"Lead day {lead_day} not found in guidance file for {spec.key!r} at init {init_dt:%Y-%m-%d}"
                )
        arr = entry.ds[spec.var_name].isel(lead_day=lead_idx).values
        return arr.astype(np.float32, copy=False)

    def _read_derived_frame_native(self, spec: _VariableSpec, dt: datetime) -> np.ndarray:
        if spec.key == "rzsm_0_100cm":
            swvl = {dep: self._read_frame_native(self._get_spec(dep), dt) for dep in spec.dependencies}
            total_depth = sum(_LAYER_THICKNESS_M[k] for k in spec.dependencies)
            out = sum(swvl[k] * _LAYER_THICKNESS_M[k] for k in spec.dependencies) / total_depth
            return out.astype(np.float32, copy=False)
        if spec.key == "soil_water_storage_0_100cm_mm":
            swvl = {dep: self._read_frame_native(self._get_spec(dep), dt) for dep in spec.dependencies}
            out = 1000.0 * sum(swvl[k] * _LAYER_THICKNESS_M[k] for k in spec.dependencies)
            return out.astype(np.float32, copy=False)
        if spec.key == "runoff":
            sro = self._read_frame_native(self._get_spec("surface_runoff"), dt)
            ssro = self._read_frame_native(self._get_spec("sub_surface_runoff"), dt)
            return (sro + ssro).astype(np.float32, copy=False)
        raise ValueError(f"Unsupported derived variable: {spec.key}")

    def _init_image_mask(self, dt: datetime) -> None:
        if self._image_mask_native is not None:
            return
        arr = self._read_frame_native(self.image_spec, dt)
        self._image_mask_native = np.isnan(arr)

    @staticmethod
    def _normalize(arr: np.ndarray, spec: _VariableSpec, normalize: bool) -> np.ndarray:
        if not normalize or spec.clip_and_normalize is None:
            return np.nan_to_num(arr, nan=0.0).astype(np.float32, copy=False)
        cmin, cmax, nmin, nmax = spec.clip_and_normalize
        out = np.nan_to_num(arr, nan=float(cmin))
        out = np.clip(out, float(cmin), float(cmax))
        out = (out - float(cmin)) / (float(cmax) - float(cmin) + 1e-12)
        out = out * (float(nmax) - float(nmin)) + float(nmin)
        return out.astype(np.float32, copy=False)

    @staticmethod
    def _resize_sequence(samples: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
        x = torch.from_numpy(samples).unsqueeze(1)
        x = F.interpolate(x, size=size_hw, mode="bilinear", align_corners=False)
        return x.squeeze(1).numpy()

    @staticmethod
    def _resize_mask(mask: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
        m = torch.from_numpy(mask.astype(np.float32))[None, None, ...]
        m = F.interpolate(m, size=size_hw, mode="nearest")
        return m[0, 0].numpy().astype(bool)

    @staticmethod
    def _stack_channels_to_single(image_hwc: np.ndarray, stack: str) -> np.ndarray:
        c = image_hwc.shape[2]
        chw = np.transpose(image_hwc, (2, 0, 1))
        slices = list(chw)
        if stack == "v":
            out = np.concatenate(slices, axis=0)
        elif stack == "h":
            out = np.concatenate(slices, axis=1)
        else:
            raise ValueError(f"Invalid stack mode: {stack}")
        return out[:, :, None]

    @staticmethod
    def _pick_crop(
        *,
        mask: np.ndarray,
        crop: int,
        smart_crop: bool,
        max_mask_fraction: float,
        smart_crop_attempts: int,
        center_crop: bool,
    ) -> tuple[int, int]:
        h, w = mask.shape
        if crop is None or crop >= min(h, w):
            return 0, 0
        if center_crop:
            return max(0, (h - crop) // 2), max(0, (w - crop) // 2)
        if not smart_crop:
            return np.random.randint(0, h - crop + 1), np.random.randint(0, w - crop + 1)

        best = (0, 0)
        best_frac = float("inf")
        for _ in range(max(1, smart_crop_attempts)):
            y0 = np.random.randint(0, h - crop + 1)
            x0 = np.random.randint(0, w - crop + 1)
            frac = float(mask[y0:y0 + crop, x0:x0 + crop].mean())
            if frac < best_frac:
                best = (y0, x0)
                best_frac = frac
            if frac <= max_mask_fraction:
                return y0, x0
        return best

    @staticmethod
    def _crop(arr: np.ndarray, y0: int, x0: int, crop: Optional[int]) -> np.ndarray:
        if crop is None:
            return arr
        return arr[..., y0:y0 + crop, x0:x0 + crop]

    @staticmethod
    def _rotate_hw(arr: np.ndarray, k: int) -> np.ndarray:
        if k == 0:
            return arr
        return np.rot90(arr, k=k, axes=(-2, -1)).copy()

    def __getitem__(self, i: int) -> dict:
        if i < 0 or i >= self._len:
            raise IndexError(i)

        start = self._start_indices[i]
        t0 = self._timestamps[start]
        self._init_image_mask(t0)
        assert self._image_mask_native is not None

        image_frames = [self._read_frame_native(self.image_spec, self._timestamps[start + j]) for j in range(self.seq_len)]
        image_samples = np.stack(image_frames, axis=0)
        mask = self._image_mask_native.copy()
        image_samples = self._normalize(image_samples, self.image_spec, normalize=True)

        forcing = None
        if self.forcing_specs:
            forcing_groups = []
            for spec in self.forcing_specs:
                frames = [self._read_frame_native(spec, self._timestamps[start + j]) for j in range(self.seq_len)]
                samples = np.stack(frames, axis=0)
                samples = self._normalize(samples, spec, normalize=self.normalize_forcing)
                forcing_groups.append(samples)
            forcing = np.stack(forcing_groups, axis=0)  # (V, S, H, W)

        guidance = None
        if self.guidance_specs:
            lead_day = self._get_guidance_target_offset()
            guidance_groups = []
            for spec in self.guidance_specs:
                arr = self._read_guidance_frame_native(spec, t0, lead_day=lead_day)
                arr = self._normalize(arr[None, ...], spec, normalize=self.normalize_forcing)[0]
                guidance_groups.append(arr)
            guidance = np.stack(guidance_groups, axis=0)  # (V, H, W)

        if self.resize is not None:
            size_hw = (int(self.resize), int(self.resize)) if isinstance(self.resize, int) else (int(self.resize[0]), int(self.resize[1]))
            image_samples = self._resize_sequence(image_samples, size_hw)
            mask = self._resize_mask(mask, size_hw)
            if forcing is not None:
                v, s, _, _ = forcing.shape
                forcing = forcing.reshape(v * s, forcing.shape[-2], forcing.shape[-1])
                forcing = self._resize_sequence(forcing, size_hw).reshape(v, s, size_hw[0], size_hw[1])
            if guidance is not None:
                guidance = self._resize_sequence(guidance, size_hw)

        y0, x0 = self._pick_crop(
            mask=mask,
            crop=self.crop if self.crop is not None else min(mask.shape),
            smart_crop=self.smart_crop,
            max_mask_fraction=self.max_mask_fraction,
            smart_crop_attempts=self.smart_crop_attempts,
            center_crop=self.center_crop,
        )

        image_samples = self._crop(image_samples, y0, x0, self.crop)
        mask = self._crop(mask[None, ...], y0, x0, self.crop)[0]
        if forcing is not None:
            forcing = self._crop(forcing, y0, x0, self.crop)
        if guidance is not None:
            guidance = self._crop(guidance, y0, x0, self.crop)

        rot_k = np.random.randint(0, 4) if self.random_rotate90 else 0
        image_samples = self._rotate_hw(image_samples, rot_k)
        mask = self._rotate_hw(mask[None, ...], rot_k)[0].astype(bool)
        if forcing is not None:
            forcing = self._rotate_hw(forcing, rot_k)
        if guidance is not None:
            guidance = self._rotate_hw(guidance, rot_k)

        image = np.transpose(image_samples, (1, 2, 0))
        if self.stack is not None:
            image = self._stack_channels_to_single(image, self.stack)
            mask = (
                np.concatenate([mask] * image_samples.shape[0], axis=0)
                if self.stack == "v"
                else np.concatenate([mask] * image_samples.shape[0], axis=1)
            )

        example = {
            "image": image.astype(np.float32, copy=False),
            "mask": mask,
            "file_path_": t0.strftime(self.date_fmt),
        }

        if forcing is not None:
            forcing_hwc = np.transpose(forcing, (2, 3, 0, 1)).reshape(forcing.shape[-2], forcing.shape[-1], -1)
        else:
            forcing_hwc = None
        if guidance is not None:
            guidance_hwc = np.transpose(guidance, (1, 2, 0))
            example["guidance"] = guidance_hwc.astype(np.float32, copy=False)
            if forcing_hwc is None:
                forcing_hwc = guidance_hwc
            else:
                forcing_hwc = np.concatenate([forcing_hwc, guidance_hwc], axis=2)
        if forcing_hwc is not None:
            example["forcing"] = forcing_hwc.astype(np.float32, copy=False)

        return example
