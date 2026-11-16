# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Band-wise MSE weighted by the truth's local variance in each band.

Plain MSE cannot see fine-scale content the truth does not have: in a smooth field the
sub-150 km bands are a fraction of a percent of the error. This loss splits the error into
Gaussian-difference bands and divides each by the truth's own climatological variance in
that band at that point, so the same absolute error counts for more where the truth is
smooth. It is an error term: it removes content the truth does not support and leaves
supported content alone; it cannot add variance.

The grid must hold a regular ``ny x nx`` block in its first ``ny * nx`` nodes (row-major).
Nodes after the block (e.g. the boundary of a cutout) are not filtered and contribute zero.

The band filters live here as plain functions so that the script producing the statistics
file uses exactly the code the loss does.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

from anemoi.training.losses.base import BaseLoss

if TYPE_CHECKING:
    from torch.distributed.distributed_c10d import ProcessGroup

LOGGER = logging.getLogger(__name__)

# Gaussian kernels are cut at this many standard deviations.
TAPS_TRUNCATE = 3.0


def gaussian_sigma_px(band_km: float, grid_km: float) -> float:
    """Standard deviation, in grid points, of the Gaussian that halves the amplitude at ``band_km``.

    Half amplitude, not half power: the low-pass and its complement then cross at
    ``band_km``, so a band edge means the same thing seen from either side. With the
    half-power convention the complement only reaches half power at a much shorter
    wavelength and every band sits below its nominal range.
    """
    return math.sqrt(math.log(2.0) / 2.0) / math.pi * band_km / grid_km


def gaussian_taps(band_km: float, grid_km: float) -> torch.Tensor:
    """1-D taps of the Gaussian low-pass with half-amplitude wavelength ``band_km``."""
    sigma = gaussian_sigma_px(band_km, grid_km)
    radius = max(1, math.ceil(TAPS_TRUNCATE * sigma))
    x = torch.arange(-radius, radius + 1, dtype=torch.float64)
    taps = torch.exp(-0.5 * (x / sigma) ** 2)
    return (taps / taps.sum()).to(torch.float32)


def _extrapolate_pad(x: torch.Tensor, r: int, dim: int) -> torch.Tensor:
    """Pad ``r`` points on both sides of ``dim`` by odd reflection about the edge value."""
    n = x.shape[dim]
    lo = torch.arange(r, 0, -1, device=x.device)
    hi = torch.arange(n - 2, n - 2 - r, -1, device=x.device)
    return torch.cat(
        [
            2 * x.narrow(dim, 0, 1) - x.index_select(dim, lo),
            x,
            2 * x.narrow(dim, n - 1, 1) - x.index_select(dim, hi),
        ],
        dim=dim,
    )


def gaussian_lowpass(x: torch.Tensor, taps: torch.Tensor) -> torch.Tensor:
    """Separable low-pass of ``x`` (N, ny, nx).

    The field is continued past each edge by odd reflection, which extends constants and
    linear trends exactly. Zero padding, or renormalising the part of the kernel that is
    inside the domain, makes the low-pass one-sided there: the large-scale gradient then
    leaks into the fine bands along the edge, and for a smooth field that leak is far
    larger than its real fine-scale content.
    """
    k = taps.numel()
    r = k // 2
    taps = taps.to(device=x.device, dtype=x.dtype)
    x = F.conv2d(_extrapolate_pad(x, r, -1)[:, None], taps.view(1, 1, 1, k))
    return F.conv2d(_extrapolate_pad(x, r, -2), taps.view(1, 1, k, 1))[:, 0]


def band_decompose(
    x: torch.Tensor,
    taps: list[torch.Tensor],
    grid_shape: tuple[int, int],
    active: list[bool] | None = None,
) -> list[torch.Tensor | None]:
    """Split ``x`` (..., ny * nx, n_variables) into Gaussian-difference bands.

    Band 0 is ``x - G_0 x``, band k is ``G_(k-1) x - G_k x``; what is left, ``G_(K-1) x``, is
    not returned. Entries whose ``active`` flag is False are returned as None and the
    low-passes only they would need are not computed.
    """
    ny, nx = grid_shape
    if x.shape[-2] != ny * nx:
        msg = f"band_decompose expects {ny * nx} grid points, got {x.shape[-2]}"
        raise ValueError(msg)
    active = [True] * len(taps) if active is None else list(active)
    block = x.movedim(-1, -2).reshape(-1, ny, nx)
    needed = {i for k, on in enumerate(active) if on for i in (k, k + 1)}
    lows = {0: block} if 0 in needed else {}
    for i, t in enumerate(taps, start=1):
        if i in needed:
            lows[i] = gaussian_lowpass(block, t)

    def restore(b: torch.Tensor) -> torch.Tensor:
        return b.reshape(*x.shape[:-2], x.shape[-1], ny * nx).movedim(-1, -2)

    return [restore(lows[k] - lows[k + 1]) if on else None for k, on in enumerate(active)]


def band_labels(bands_km: list[float]) -> list[str]:
    """Human-readable labels, e.g. ``['lt22', '22-55', '55-150']``."""
    edges = [f"{b:g}" for b in bands_km]
    return [f"lt{edges[0]}"] + [f"{lo}-{hi}" for lo, hi in zip(edges[:-1], edges[1:], strict=True)]


def save_band_stats(
    path: str | Path,
    *,
    sigma2: np.ndarray,
    mean_sigma2: np.ndarray,
    variables: list[str],
    bands_km: list[float],
    grid_shape: tuple[int, int],
    grid_km: float,
    meta: dict | None = None,
) -> None:
    """Write the statistics file read by :class:`BandWeightedMSELoss`.

    Parameters
    ----------
    sigma2 : np.ndarray
        Truth variance per band, variable and (possibly coarsened) grid point,
        shape (n_bands, n_variables, cy, cx).
    mean_sigma2 : np.ndarray
        Its mean over the full-resolution block, shape (n_bands, n_variables).
    """
    taps = {f"taps_{k}": gaussian_taps(b, grid_km).numpy() for k, b in enumerate(bands_km)}
    np.savez(
        path,
        sigma2=np.asarray(sigma2, dtype=np.float32),
        mean_sigma2=np.asarray(mean_sigma2, dtype=np.float64),
        variables=np.asarray(variables),
        bands_km=np.asarray(bands_km, dtype=np.float64),
        grid_shape=np.asarray(grid_shape, dtype=np.int64),
        grid_km=np.float64(grid_km),
        meta=json.dumps(meta or {}),
        **taps,
    )


def load_band_stats(
    path: str | Path,
    variables: list[str],
    *,
    bands_km: list[float],
    grid_shape: tuple[int, int],
    grid_km: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read the statistics for ``variables`` and return them at full resolution.

    Returns
    -------
    sigma2 : torch.Tensor
        Shape (n_bands, n_variables, ny, nx), bilinearly upsampled if stored coarsened.
    mean_sigma2 : torch.Tensor
        Shape (n_bands, n_variables).
    """
    with np.load(path, allow_pickle=False) as f:
        file_vars = [str(v) for v in f["variables"]]
        missing = [v for v in variables if v not in file_vars]
        if missing:
            msg = f"Variables {missing} are not in the band statistics file {path}"
            raise ValueError(msg)
        if not np.allclose(f["bands_km"], bands_km) or tuple(f["grid_shape"]) != tuple(grid_shape):
            msg = (
                f"Band statistics file {path} was made for bands {f['bands_km'].tolist()} km on grid "
                f"{tuple(f['grid_shape'])}, not {list(bands_km)} km on {tuple(grid_shape)}"
            )
            raise ValueError(msg)
        for k, b in enumerate(bands_km):
            taps = gaussian_taps(b, grid_km).numpy()
            if f[f"taps_{k}"].shape != taps.shape or not np.allclose(f[f"taps_{k}"], taps, atol=1e-7):
                msg = f"Band statistics file {path} was made with a different {b:g} km filter"
                raise ValueError(msg)
        idx = [file_vars.index(v) for v in variables]
        sigma2 = torch.from_numpy(f["sigma2"][:, idx].astype(np.float32))
        mean_sigma2 = torch.from_numpy(f["mean_sigma2"][:, idx].astype(np.float32))
    if tuple(sigma2.shape[-2:]) != tuple(grid_shape):
        sigma2 = F.interpolate(sigma2, size=tuple(grid_shape), mode="bilinear", align_corners=False)
    return sigma2.clamp_min(0.0), mean_sigma2


def inverse_variance_weights(sigma2: torch.Tensor, mean_sigma2: torch.Tensor, eps_factor: float) -> torch.Tensor:
    """Weights ``w = u / <u> / <sigma2>`` with ``u = 1 / (sigma2 + eps_factor * median(sigma2))``.

    The floor is tied to the median because the truth variance is strongly skewed (its mean
    is 3 to 20 times its median, set by terrain): a floor at the mean leaves the weight
    almost uniform. With ``eps_factor = 1`` the smoother half of the domain carries close to
    the full weight and the rougher half is weighted down as ``1 / sigma2``.

    A spatially uniform error of variance E scores ``E / <sigma2>`` whatever ``eps_factor``
    is: it only moves weight between smooth and rough regions. Means and medians are taken
    over the block, unweighted.
    """
    median = sigma2.flatten(-2).median(dim=-1).values[..., None, None]
    u = 1.0 / (sigma2 + eps_factor * median)
    return u / u.mean(dim=(-2, -1), keepdim=True) / mean_sigma2[..., None, None]


class BandWeightedMSELoss(BaseLoss):
    """MSE of the error in Gaussian-difference bands, weighted by the truth's local band variance."""

    name: str = "band_mse"

    def __init__(
        self,
        stats_path: str,
        variables: list[str],
        grid_shape: list[int],
        grid_km: float,
        bands_km: list[float],
        band_weights: list[float] | None = None,
        eps_factor: float = 1.0,
        label: str = "band",
        ignore_nans: bool = False,
        **kwargs,
    ) -> None:
        """Band-weighted MSE.

        Parameters
        ----------
        stats_path : str
            File written by :func:`save_band_stats` with the truth's variance maps.
        variables : list[str]
            Variables the loss is applied to, in the order they arrive. Must equal the
            ``predicted_variables`` of the loss config (the schema checks it).
        grid_shape : list[int]
            (ny, nx) of the regular block at the start of the grid dimension.
        grid_km : float
            Grid spacing of the block.
        bands_km : list[float]
            Increasing half-amplitude wavelengths of the low-passes that separate the bands.
        band_weights : list[float], optional
            Weight per band, by default all 1. A band with weight 0 is not computed.
        eps_factor : float, optional
            Floor on the truth variance, as a multiple of its median over the domain. Small
            values concentrate the weight on smooth regions; large values give a uniform weight.
        label : str, optional
            Name used by ``BandLossMonitor`` when logging the band values.
        ignore_nans : bool, optional
            Allow nans in the loss and apply methods ignoring nans for measuring the loss.
        """
        super().__init__(ignore_nans=ignore_nans)
        del kwargs
        # The filters need every neighbour of a point, so the grid cannot arrive sharded.
        self.supports_sharding = False

        self.variables = list(variables)
        self.grid_shape = (int(grid_shape[0]), int(grid_shape[1]))
        self.bands_km = [float(b) for b in bands_km]
        self.band_weights = [1.0] * len(self.bands_km) if band_weights is None else [float(w) for w in band_weights]
        assert len(self.band_weights) == len(self.bands_km), "band_weights must have one entry per band"
        assert self.bands_km == sorted(self.bands_km), "bands_km must be increasing"
        self.active = [w != 0.0 for w in self.band_weights]
        self.labels = band_labels(self.bands_km)
        self.label = label
        self.eps_factor = float(eps_factor)

        self._taps = [gaussian_taps(b, grid_km) for b in self.bands_km]
        sigma2, mean_sigma2 = load_band_stats(
            stats_path,
            self.variables,
            bands_km=self.bands_km,
            grid_shape=self.grid_shape,
            grid_km=grid_km,
        )
        weights = inverse_variance_weights(sigma2, mean_sigma2, self.eps_factor)
        # (n_bands, n_variables, ny, nx) -> (n_bands, ny * nx, n_variables), active bands only.
        # Not persistent: the maps are large and must stay out of checkpoints.
        for k, on in enumerate(self.active):
            if on:
                self.register_buffer(f"weights_{k}", weights[k].flatten(-2).transpose(0, 1).contiguous(), persistent=False)

        self.last_components: dict[str, torch.Tensor] = {}
        self._logged_first_call = False
        LOGGER.info(
            "BandWeightedMSELoss '%s': %d variables %s, bands %s km (weights %s), eps_factor %g, grid %s, stats %s",
            self.label,
            len(self.variables),
            self.variables,
            self.labels,
            self.band_weights,
            self.eps_factor,
            self.grid_shape,
            stats_path,
        )

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        squash: bool = True,
        *,
        scaler_indices: tuple[int, ...] | None = None,
        without_scalers: list[str] | list[int] | None = None,
        grid_shard_slice: slice | None = None,
        group: ProcessGroup | None = None,
        squash_mode: str = "avg",
        **_kwargs,
    ) -> torch.Tensor:
        """Calculates the band-weighted loss.

        Parameters
        ----------
        pred : torch.Tensor
            Prediction tensor, shape (bs, output_times, ensemble, lat*lon, n_outputs)
        target : torch.Tensor
            Target tensor, shape (bs, output_times, ensemble, lat*lon, n_outputs)
        squash : bool, optional
            Average last dimension, by default True
        scaler_indices: tuple[int,...], optional
            Indices to subset the calculated scaler with, by default None
        without_scalers: list[str] | list[int] | None, optional
            list of scalers to exclude from scaling. Can be list of names or dimensions to exclude.
            By default None
        grid_shard_slice : slice, optional
            Must be None: this loss needs the full grid.
        group: ProcessGroup, optional
            Unused, the loss is never sharded.
        squash_mode : str, optional
            Reduction mode for the variable dimension, by default ``"avg"``

        Returns
        -------
        torch.Tensor
            Weighted loss
        """
        del group
        if grid_shard_slice is not None:
            msg = "BandWeightedMSELoss needs the full grid, but received a grid shard."
            raise RuntimeError(msg)
        n_block = self.grid_shape[0] * self.grid_shape[1]
        if pred.shape[-2] < n_block or pred.shape[-1] != len(self.variables):
            msg = (
                f"BandWeightedMSELoss '{self.label}' expects at least {n_block} grid points and "
                f"{len(self.variables)} variables, got {tuple(pred.shape)}"
            )
            raise ValueError(msg)
        if not self._logged_first_call:
            LOGGER.info(
                "BandWeightedMSELoss '%s' first call: pred %s %s, target %s %s",
                self.label,
                tuple(pred.shape),
                pred.dtype,
                tuple(target.shape),
                target.dtype,
            )
            self._logged_first_call = True

        # The band content is orders of magnitude below the field: keep it out of half precision.
        with torch.autocast(device_type=pred.device.type, enabled=False):
            error = pred[..., :n_block, :].float() - target[..., :n_block, :].float()
            bands = band_decompose(error, self._taps, self.grid_shape, self.active)

            loss = None
            for k, band in enumerate(bands):
                if band is None:
                    continue
                out = F.pad(getattr(self, f"weights_{k}") * band.square(), (0, 0, 0, pred.shape[-2] - n_block))
                out = self.scale(out, scaler_indices, without_scalers=without_scalers)
                value = self.reduce(out, squash, squash_mode=squash_mode)
                self.last_components[self.labels[k]] = value.detach()
                loss = self.band_weights[k] * value if loss is None else loss + self.band_weights[k] * value

        if loss is None:  # every band switched off
            shape = () if squash else (pred.shape[-1],)
            return torch.zeros(shape, dtype=torch.float32, device=pred.device)
        return loss
