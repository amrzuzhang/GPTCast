import numpy as np

def dbz_to_rainfall(dBZ, a=200., b=1.6, cutoff=0.04):
    rr = np.power(10, (dBZ - 10 * np.log10(a)) / (10 * b))
    rr[rr < cutoff] = 0
    return rr


def rainfall_to_dbz(rainfall, a=200., b=1.6, cutoff=0.):
    dbz = 10 * np.log10(a) + 10 * b * np.log10(rainfall)
    dbz[dbz < cutoff] = 0
    return dbz


def swvl1_norm_to_phys(
    x_norm: np.ndarray,
    *,
    clip: tuple[float, float] = (0.0, 0.8),
    norm: tuple[float, float] = (-1.0, 1.0),
) -> np.ndarray:
    """Convert ERA5-Land SWVL1 from model normalization space back to physical units.

    Training uses a simple clip + minmax mapping:
      phys in [clip[0], clip[1]]  ->  norm in [norm[0], norm[1]]

    Args:
        x_norm: Array in normalized space (typically [-1, 1]). Can be a numpy array
            or a numpy masked array; masks are preserved.
        clip: Physical clipping range for swvl1 (m3/m3).
        norm: Normalized range used during training.
    """
    cmin, cmax = float(clip[0]), float(clip[1])
    nmin, nmax = float(norm[0]), float(norm[1])

    if isinstance(x_norm, np.ma.MaskedArray):
        data = x_norm.data.astype(np.float32, copy=False)
        mask = x_norm.mask
    else:
        data = np.asarray(x_norm, dtype=np.float32)
        mask = None

    # [nmin,nmax] -> [0,1]
    x01 = (data - nmin) / (nmax - nmin + 1e-12)
    # [0,1] -> [cmin,cmax]
    x_phys = x01 * (cmax - cmin) + cmin
    x_phys = np.clip(x_phys, cmin, cmax)

    if mask is not None:
        return np.ma.masked_array(x_phys, mask=mask)
    return x_phys


def swvl1_phys_to_norm(
    x_phys: np.ndarray,
    *,
    clip: tuple[float, float] = (0.0, 0.8),
    norm: tuple[float, float] = (-1.0, 1.0),
) -> np.ndarray:
    """Convert ERA5-Land SWVL1 from physical units (m3/m3) to model normalization space."""
    cmin, cmax = float(clip[0]), float(clip[1])
    nmin, nmax = float(norm[0]), float(norm[1])

    if isinstance(x_phys, np.ma.MaskedArray):
        data = x_phys.data.astype(np.float32, copy=False)
        mask = x_phys.mask
    else:
        data = np.asarray(x_phys, dtype=np.float32)
        mask = None

    data = np.nan_to_num(data, nan=cmin)
    data = np.clip(data, cmin, cmax)
    x01 = (data - cmin) / (cmax - cmin + 1e-12)
    x_norm = x01 * (nmax - nmin) + nmin

    if mask is not None:
        return np.ma.masked_array(x_norm, mask=mask)
    return x_norm
