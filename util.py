import numpy as np
import torch
from numpy.typing import ArrayLike

LIGHTSPEED_ANGSTROM_S = 2.99792458e+18



def mag_to_flux(mag):
    """AB magnitude -> flux density f_nu in erg/s/cm^2/Hz (numpy or torch input)."""
    return 10 ** ((mag + 48.6) / -2.5)


def flux_to_mag(flux):
    """Flux density f_nu in erg/s/cm^2/Hz -> AB magnitude (numpy or torch input); non-positive flux gives NaN."""
    log10 = torch.log10 if isinstance(flux, torch.Tensor) else np.log10
    return -2.5 * log10(flux) - 48.6



# Filter transmission curves, one file per band
# (paths are relative to the repository root — run code from there)
FILTER_PATH = 'datasets/filters/'
PHOTO_FILTER_LIST = {
    'u': FILTER_PATH + 'u.par',
    'g': FILTER_PATH + 'g.par',
    'r': FILTER_PATH + 'r.par',
    'i': FILTER_PATH + 'i.par',
    'z': FILTER_PATH + 'Z.par',
    'Y': FILTER_PATH + 'Y.par',
    'J': FILTER_PATH + 'J.par',
    'H': FILTER_PATH + 'H.par',
    'K': FILTER_PATH + 'Ks.par',
}



def _interp_gpu(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """Linear interpolation equivalent to np.interp(..., left=0, right=0). xp must be 1D sorted."""
    idx = torch.searchsorted(xp.contiguous(), x.contiguous()).clamp(1, len(xp) - 1)
    x0, x1 = xp[idx - 1], xp[idx]
    y0, y1 = fp[idx - 1], fp[idx]
    result = y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return result.masked_fill((x < xp[0]) | (x > xp[-1]), 0.0)



def ab_mag(obs_frame_fluxes_erg_s_cm2_A: torch.Tensor,
           obs_frame_waves_A: torch.Tensor,
           bands: str | list,
           batch_size: int = 10_000,
           device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    """
    AB magnitudes from observed-frame SEDs by integrating over filter curves.

    Parameters
    ----------
    obs_frame_fluxes_erg_s_cm2_A : Tensor, shape (n_gal, n_lambda)
        Observed-frame f_lambda in erg/s/cm^2/A.
    obs_frame_waves_A : Tensor, shape (n_gal, n_lambda) or (n_lambda,)
        Observed-frame wavelengths in Angstrom, ascending along the last axis.
        A 1-D grid is shared by all SEDs.
    bands : str or list of str
        Keys of ``PHOTO_FILTER_LIST``, e.g. ``'ugrizYJHK'``.
    batch_size : int
        Number of SEDs integrated per pass on ``device``; all bands are
        computed in one pass per batch.
    device : torch.device or str
        Where the integration runs.

    Returns
    -------
    dict
        ``{band: Tensor of shape (n_gal,)}`` with AB magnitudes, on CPU.
    """

    bands = list(bands)
    for band in bands:
        if band not in PHOTO_FILTER_LIST:
            raise ValueError(f"band '{band}' not in {list(PHOTO_FILTER_LIST)}.")

    dtype = obs_frame_fluxes_erg_s_cm2_A.dtype

    shared_wave = obs_frame_waves_A.ndim == 1
    if shared_wave:
        obs_frame_waves_A = obs_frame_waves_A.unsqueeze(0)

    # Pre-load all filter curves onto target device: list of (filter_wav, filter_trans) pairs
    filter_curves = [(torch.from_numpy(arr[:, 0]).to(device=device, dtype=dtype),
                      torch.from_numpy(arr[:, 1]).to(device=device, dtype=dtype))
                     for arr in (np.genfromtxt(PHOTO_FILTER_LIST[b]) for b in bands)]

    results = []
    with torch.no_grad():
        for start in range(0, obs_frame_fluxes_erg_s_cm2_A.shape[0], batch_size):
            flux_b = obs_frame_fluxes_erg_s_cm2_A[start:start + batch_size].to(device)
            wave_b = (obs_frame_waves_A if shared_wave else obs_frame_waves_A[start:start + batch_size]).to(device)

            all_T = torch.stack([_interp_gpu(wave_b, fw, ft) for fw, ft in filter_curves]) # (n_bands, batch, n_lambda) or (n_bands, 1, n_lambda)

            # flux_b broadcasts from (batch, n_lambda) to (n_bands, batch, n_lambda)
            num = torch.trapezoid(flux_b * all_T * wave_b, x=wave_b, dim=-1) / LIGHTSPEED_ANGSTROM_S
            den = torch.trapezoid(all_T / wave_b, x=wave_b, dim=-1)
            results.append((-2.5 * torch.log10(num / den) - 48.6))  # (n_bands, batch)

    mags_all = torch.cat(results, dim=-1).cpu()  # (n_bands, n_gal)
    output = {band: mags_all[i] for i, band in enumerate(bands)}

    torch.cuda.empty_cache()
    return output



class ECDF_vectorized:
    """Vectorized empirical CDF along one axis of an N-d array.

    Parameters
    ----------
    sample : array_like
        Reference sample. ``axis`` is the sample dimension (e.g. index over
        spectra); all other axes are "bin" dimensions (e.g. wavelength).
        NaNs are not handled — mask them out before passing in.
    axis : int, default 0
        Axis along which ``sample`` enumerates draws.
    side : {'right', 'left'}, default 'right'
        Tie convention. ``'right'`` is the standard ECDF
        ``F(x) = #{X_i <= x} / n``; ``'left'`` gives ``#{X_i < x} / n``.
    """

    def __init__(self, sample: ArrayLike, axis: int = 0, side: str = "right"):
        if side not in ("right", "left"):
            raise ValueError("side must be 'right' or 'left'")
        sample = np.asarray(sample)
        if sample.ndim < 1:
            raise ValueError("sample must have at least one dimension")
        self._axis = axis
        self._side = side
        # Sample axis -> position 0, then sort.
        moved = np.moveaxis(sample, axis, 0)
        self._sorted = np.sort(moved, axis=0)
        self._n = self._sorted.shape[0]
        self._bin_shape = self._sorted.shape[1:]

    def __call__(self, x: ArrayLike) -> np.ndarray:
        """Evaluate the per-bin ECDF at ``x``.

        ``x`` must match the reference shape on the non-sample axes.
        """
        x = np.asarray(x)
        moved = np.moveaxis(x, self._axis, 0)
        if moved.shape[1:] != self._bin_shape:
            raise ValueError(
                f"Bin-shape mismatch: queries have {moved.shape[1:]}, "
                f"reference has {self._bin_shape}."
            )
        k = moved.shape[0]
        ref_flat = self._sorted.reshape(self._n, -1)   # (n, M)
        q_flat = moved.reshape(k, -1)                  # (k, M)
        out = np.empty_like(q_flat, dtype=float)
        for j in range(ref_flat.shape[1]):
            out[:, j] = np.searchsorted(
                ref_flat[:, j], q_flat[:, j], side=self._side
            )
        out /= self._n
        out = out.reshape(moved.shape)
        return np.moveaxis(out, 0, self._axis)


