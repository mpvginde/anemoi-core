# (C) Copyright 2024- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import logging
from functools import cached_property

import numpy as np
import torch
from einops import rearrange
from omegaconf import DictConfig
from rich.console import Console
from rich.tree import Tree

from anemoi.datasets import open_dataset
from anemoi.training.data.usable_indices import get_usable_indices
from anemoi.training.utils.time_indices import TimeIndices
from anemoi.training.utils.time_indices import normalize_time_indices

LOGGER = logging.getLogger(__name__)


def _as_dict(value: str | dict | DictConfig) -> str | dict:
    """Convert DictConfig payloads to plain dicts."""
    return dict(value) if isinstance(value, DictConfig) else value


def _normalize_dataset_config(dataset_config: str | dict | DictConfig) -> str | dict:
    """Normalize dataset payload to the open_dataset dictionary contract."""
    dataset_config = _as_dict(dataset_config)
    if not isinstance(dataset_config, dict):
        return dataset_config

    if "dataset" not in dataset_config:
        msg = "dataset_config must contain the 'dataset' key."
        raise ValueError(msg)

    if dataset_config["dataset"] is None:
        msg = "dataset_config.dataset cannot be None."
        raise ValueError(msg)

    invalid_inner_keys = {"start", "end"} & set(dataset_config)
    if invalid_inner_keys:
        invalid = ", ".join(sorted(invalid_inner_keys))
        msg = f"dataset_config cannot contain [{invalid}]. Use outer keys 'start' and 'end' instead."
        raise ValueError(msg)

    # Keep only explicitly set options to avoid passing None-valued kwargs
    # (e.g. select=None), which can trigger downstream subset selection issues.
    return {key: value for key, value in dataset_config.items() if value is not None}


def _normalize_reader_config(dataset_config: dict | DictConfig) -> dict:
    """Validate and normalize reader configuration."""
    normalized = dict(dataset_config)

    if "dataset" in normalized:
        msg = (
            "Invalid dataloader dataset schema: use 'dataset_config' (outer key) "
            "and 'dataset' inside it. The legacy outer 'dataset' key is no longer supported."
        )
        raise ValueError(msg)

    base_dataset_config = normalized.pop("dataset_config", None)
    if base_dataset_config is None:
        msg = "Missing required 'dataset_config' in dataset reader configuration."
        raise ValueError(msg)

    allowed_keys = {"start", "end", "trajectory", "overrides"}
    unknown_keys = set(normalized) - allowed_keys
    if unknown_keys:
        unknown = ", ".join(sorted(unknown_keys))
        allowed = ", ".join(sorted({"dataset_config", *allowed_keys}))
        msg = f"Unknown dataset reader option(s) [{unknown}]. Allowed top-level keys: {allowed}."
        raise ValueError(msg)

    normalized["dataset_config"] = base_dataset_config
    return normalized


def _read_trajectory_steps(data: object, sequence: int, positions: list[int]) -> np.ndarray:
    """Read forecast steps ``positions`` of initialisation ``sequence`` of a trajectories dataset.

    Returns an array of shape ``(variables, ensembles, steps, cells)``.
    """
    positions = normalize_time_indices(positions)
    if isinstance(positions, slice):
        # Read only the requested steps instead of the whole forecast; with per-step chunking
        # this avoids reading and decompressing unused steps. Only slices take this path: a list
        # index on the step axis is not applied correctly by anemoi-datasets.
        return data[sequence, :, :, positions, :]
    return data[sequence][:, :, positions, :]


class BaseAnemoiReader:
    """Anemoi data reader for native grid datasets.

    A sample is addressed by a 2-D anchor ``(sequence, position)``.
    Relative time offsets are applied *within* a single sequence, so a
    sample can never span two sequences.  Analysis datasets expose a single
    sequence (the whole time series); forecast datasets expose one sequence
    per initialisation (base date), with the forecast step as the position.
    """

    def __init__(
        self,
        dataset: str | dict | None = None,
        dataset_config: str | dict | None = None,
        start: datetime.datetime | int | None = None,
        end: datetime.datetime | int | None = None,
    ):
        """Initialize Anemoi data reader."""
        source = dataset_config if dataset_config is not None else dataset
        if source is None:
            msg = "Either dataset or dataset_config must be provided."
            raise ValueError(msg)
        self.data = open_dataset(_normalize_dataset_config(source), start=start, end=end)
        #: Sampling config used by :meth:`compute_anchors`.
        #: ``{"stride": 1}`` keeps every valid position;
        #: ``{"stride": None}`` uses stride = window size (non-overlapping).
        self.default_sampling: dict = {"stride": 1}

    # ------------------------------------------------------------------
    # Sequence / position geometry
    # ------------------------------------------------------------------

    @property
    def num_sequences(self) -> int:
        """Number of independent sequences in the dataset."""
        return 1

    def sequence_length(self, sequence: int = 0) -> int:  # noqa: ARG002
        """Return the number of positions in ``sequence``."""
        return len(self.data.dates)

    @property
    def missing_sequences(self) -> set[int]:
        """Return sequences that are entirely missing and must not be sampled."""
        return set()

    def missing_positions(self, sequence: int = 0) -> set[int]:  # noqa: ARG002
        """Return positions within ``sequence`` that are missing."""
        return set(self.missing)

    def compute_anchors(
        self,
        relative_indices: list[int] | np.ndarray,
        sampling: dict | None = None,
    ) -> np.ndarray:
        """Return the valid ``(sequence, position)`` anchors for a relative window.

        Parameters
        ----------
        relative_indices : list[int] | np.ndarray
            Relative offsets (in positions) requested around each anchor.
        sampling : dict | None
            Sampling configuration with key ``"stride"``.
            ``{"stride": None}`` uses stride = window size (non-overlapping);
            ``{"stride": 1}`` keeps every valid position;
            ``{"stride": 6}`` steps anchors by 6.
            Defaults to :attr:`default_sampling`.

        Returns
        -------
        np.ndarray
            Array of shape ``(n_anchors, 2)`` with ``(sequence, position)`` rows.
        """
        sampling = sampling or self.default_sampling

        rel = np.asarray(list(relative_indices), dtype=np.int64)
        window = int(rel.max()) - int(rel.min()) + 1

        # Resolve stride from sampling dict; None → window size (non-overlapping)
        raw_stride = sampling.get("stride") if isinstance(sampling, dict) else None
        stride = window if raw_stride is None else int(raw_stride)
        if stride < 1:
            msg = f"trajectory_sampling.stride must be >= 1, got {stride}."
            raise ValueError(msg)

        anchors: list[np.ndarray] = []
        for sequence in range(self.num_sequences):
            if sequence in self.missing_sequences:
                continue

            positions = get_usable_indices(
                self.missing_positions(sequence),
                self.sequence_length(sequence),
                rel,
            )

            if stride > 1 and positions.size:
                positions = positions[(positions - positions[0]) % stride == 0]

            if positions.size:
                seq_col = np.full(positions.size, sequence, dtype=np.int64)
                anchors.append(np.stack([seq_col, positions], axis=1))

        if not anchors:
            return np.empty((0, 2), dtype=np.int64)
        return np.concatenate(anchors, axis=0)

    # ------------------------------------------------------------------
    # Dataset properties
    # ------------------------------------------------------------------

    @property
    def dates(self) -> np.ndarray:
        """Return dataset dates."""
        return self.data.dates

    @property
    def grid_size(self) -> int:
        """Return dataset grid size."""
        return sum(self.data.grids)

    @property
    def statistics(self) -> dict:
        """Return dataset statistics."""
        return self.data.statistics

    def statistics_tendencies(
        self,
        timestep: int | str | datetime.timedelta | None = None,
    ) -> dict | None:
        """Return dataset tendency statistics."""
        if timestep is None:
            timestep = getattr(self, "timestep", None)
        if timestep is None:
            msg = "timestep must be provided to compute tendency statistics."
            raise ValueError(msg)
        try:
            return self.data.statistics_tendencies(timestep)
        except (KeyError, AttributeError):
            return None

    @property
    def variables(self) -> list[str]:
        """Return dataset variables."""
        return self.data.variables

    @property
    def missing(self) -> set[int]:
        """Return dataset missing values mask."""
        return self.data.missing

    @property
    def metadata(self) -> dict:
        """Return dataset metadata."""
        return self.data.metadata()

    @property
    def frequency(self) -> datetime.timedelta:
        """Return dataset frequency."""
        return self.data.frequency

    @property
    def supporting_arrays(self) -> dict:
        """Return dataset supporting_arrays."""
        return self.data.supporting_arrays()

    @property
    def name_to_index(self) -> dict[str, int]:
        """Return dataset name-to-index mapping."""
        return self.data.name_to_index

    @property
    def resolution(self) -> str:
        """Return dataset resolution."""
        return self.data.resolution

    @cached_property
    def cutout_mask(self) -> np.ndarray:
        """Return cutout mask."""
        cutout_mask = np.zeros(self.grid_size, dtype=bool)
        if len(self.data.grids) <= 1:
            err_msg = "Dataset `cutout_mask` property requires a cutout grid but does not have one."
            raise ValueError(err_msg)
        cutout_mask[: self.data.grids[0]] = True
        return cutout_mask

    @cached_property
    def boundary_mask(self) -> np.ndarray:
        """Return boundary mask, defined as the complement of the cutout mask."""
        return ~self.cutout_mask

    # ------------------------------------------------------------------
    # Sample loading
    # ------------------------------------------------------------------

    def get_sample(
        self,
        sequence: int,
        positions: TimeIndices,
        grid_shard_indices: np.ndarray | slice | None = None,
    ) -> torch.Tensor:
        """Get a sample from the dataset.

        For analysis datasets there is a single sequence, so ``sequence`` is
        ignored and ``positions`` index the time axis directly.
        """
        del sequence  # analysis datasets have a single sequence
        if isinstance(grid_shard_indices, slice):
            x = self.data[positions, :, :, grid_shard_indices]
        else:
            x = self.data[positions, :, :, :]
            if grid_shard_indices is not None:
                x = x[..., grid_shard_indices]

        x = rearrange(x, "dates variables ensemble gridpoints -> dates ensemble gridpoints variables")
        return torch.from_numpy(x)

    def __repr__(self) -> str:
        console = Console(record=True, width=120)
        with console.capture() as capture:
            console.print(self.tree())
        return capture.get()

    def tree(self, prefix: str = "") -> Tree:
        tree = Tree(prefix + " 💾 " + f"{self.__class__.__name__}")
        tree.add(f"Dataset: {self.data}")
        tree.add(f"Frequency: {self.frequency}")
        tree.add(f"Resolution: {self.resolution}")
        tree.add(f"Num variables: {len(self.name_to_index)}")
        return tree


class NativeGridDataset(BaseAnemoiReader):
    """Native grid (analysis) dataset.

    A single sequence covering the whole time series; relative offsets index
    the time axis directly.  This is the default analysis behaviour.
    """

    def __init__(
        self,
        dataset: str | dict | None = None,
        dataset_config: str | dict | None = None,
        start: datetime.datetime | int | None = None,
        end: datetime.datetime | int | None = None,
        sampling: dict | None = None,
    ) -> None:
        """Initialize NativeGridDataset."""
        super().__init__(dataset=dataset, dataset_config=dataset_config, start=start, end=end)
        if sampling is not None:
            self.default_sampling = sampling


class TrajectoryDataset(BaseAnemoiReader):
    """Trajectory dataset with an explicit lead-step axis.

    Wraps a 5-D ``trajectories``-layout dataset opened through
    :func:`anemoi.datasets.open_dataset` (on-disk shape
    ``(base_dates, variables, ensembles, steps, cells)``).  Each base date
    (forecast initialisation) is exposed as an independent sequence and the
    forecast step is the within-sequence position, so a training sample is
    always contained within a single forecast and never crosses initialisation
    boundaries.

    Step subsetting (``steps``, ``step_start``, ``step_end``,
    ``step_frequency``) and base-date subsetting (``start``/``end`` on the
    valid-time envelope, or ``base_start``/``base_end``) are handled by
    ``open_dataset`` via the dataset configuration.
    """

    def __init__(
        self,
        dataset: str | dict | None = None,
        dataset_config: str | dict | None = None,
        start: datetime.datetime | int | None = None,
        end: datetime.datetime | int | None = None,
        sampling: dict | None = None,
    ) -> None:
        source = dataset_config if dataset_config is not None else dataset
        if source is None:
            msg = "Either dataset or dataset_config must be provided."
            raise ValueError(msg)

        # Trajectory datasets derive their step frequency from the dataset itself.
        # Passing data.frequency would be misleading and is not supported.
        source_dict = _as_dict(source) if not isinstance(source, str) else {}
        if isinstance(source_dict, dict) and source_dict.get("frequency") is not None:
            msg = (
                "TrajectoryDataset does not accept a 'frequency' in dataset_config. "
                "The step frequency is read directly from the dataset. "
                "Set data.frequency: null in your config."
            )
            raise AssertionError(msg)

        # Trajectory datasets use base_start/base_end to filter by initialisation date;
        # passing start/end would trigger access to .dates which doesn't exist on them.
        open_kwargs: dict = {}
        if start is not None:
            open_kwargs["base_start"] = start
        if end is not None:
            open_kwargs["base_end"] = end
        self.data = open_dataset(_normalize_dataset_config(source), **open_kwargs)
        self.default_sampling = sampling if sampling is not None else {"stride": None}

    @property
    def num_sequences(self) -> int:
        """Number of forecast initialisations (base dates)."""
        return self.data.shape[0]

    def sequence_length(self, sequence: int = 0) -> int:  # noqa: ARG002
        """Return the number of forecast steps per initialisation."""
        return self.data.shape[-2]

    @property
    def missing_sequences(self) -> set[int]:
        """Return the base-date indices that are missing."""
        return set(self.data.missing)

    def missing_positions(self, sequence: int = 0) -> set[int]:  # noqa: ARG002
        """Forecast datasets do not track per-step missing values."""
        return set()

    @property
    def frequency(self) -> datetime.timedelta:
        """Return the step frequency (spacing between consecutive forecast steps)."""
        freq = self.data.step_frequency
        if freq is not None:
            return freq
        msg = (
            f"Cannot determine step frequency: data.step_frequency is None for dataset {self.data}. "
            "Ensure that the dataset configuration includes a valid step_frequency (e.g. '6H')."
        )
        raise ValueError(msg)

    def get_sample(
        self,
        sequence: int,
        positions: TimeIndices,
        grid_shard_indices: np.ndarray | slice | None = None,
    ) -> torch.Tensor:
        """Load forecast steps ``positions`` of initialisation ``sequence``."""
        if isinstance(positions, slice):
            positions = range(*positions.indices(self.sequence_length(sequence)))
        x = _read_trajectory_steps(self.data, sequence, np.asarray(positions).tolist())
        if grid_shard_indices is not None:
            x = x[..., grid_shard_indices]

        x = rearrange(x, "variables ensemble steps gridpoints -> steps ensemble gridpoints variables")
        return torch.from_numpy(x)

    def tree(self, prefix: str = "") -> Tree:
        tree = Tree(prefix + " 💾 " + f"{self.__class__.__name__}")
        tree.add(f"Dataset: {self.data}")
        tree.add(f"Step frequency: {self.frequency}")
        tree.add(f"Resolution: {self.resolution}")
        tree.add(f"Num variables: {len(self.name_to_index)}")
        tree.add(f"Num initialisations: {self.num_sequences}")
        tree.add(f"Steps per initialisation: {self.sequence_length()}")
        tree.add(f"Sampling: {self.default_sampling}")
        return tree


#: Grid points that a :class:`SourceOverride` can replace.
OVERRIDE_REGIONS = ("interior", "boundary", "all")
#: Time steps that a :class:`SourceOverride` can replace, relative to the anchor t_0 of a sample.
OVERRIDE_TIMES = ("inputs", "outputs", "from_t0", "all")

_EARTH_RADIUS_KM = 6371.0


def _latlon_to_xyz(latitudes: np.ndarray, longitudes: np.ndarray) -> np.ndarray:
    """Convert latitudes and longitudes in degrees to points on the unit sphere."""
    lat = np.deg2rad(np.asarray(latitudes, dtype=np.float64))
    lon = np.deg2rad(np.asarray(longitudes, dtype=np.float64))
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)


def _match_grid_points(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    source_latitudes: np.ndarray,
    source_longitudes: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Return the nearest source point of each grid point, and the largest of these distances in km."""
    from scipy.spatial import KDTree

    tree = KDTree(_latlon_to_xyz(source_latitudes, source_longitudes))
    chord, index = tree.query(_latlon_to_xyz(latitudes, longitudes))
    distance_km = 2 * _EARTH_RADIUS_KM * np.arcsin(np.clip(np.asarray(chord) / 2, 0.0, 1.0))
    largest_km = float(distance_km.max()) if distance_km.size else 0.0
    return np.asarray(index, dtype=np.int64), largest_km


def _check_override_options(region: str, times: str) -> None:
    """Raise if an override replaces unknown grid points or time steps, or the loss targets."""
    if region not in OVERRIDE_REGIONS:
        msg = f"Unknown override region '{region}'. Choose one of {OVERRIDE_REGIONS}."
        raise ValueError(msg)
    if times not in OVERRIDE_TIMES:
        msg = f"Unknown override times '{times}'. Choose one of {OVERRIDE_TIMES}."
        raise ValueError(msg)
    if region != "boundary" and times != "inputs":
        msg = (
            f"An override of region '{region}' can only replace the inputs (times: inputs), got '{times}'. "
            "The interior at later time steps holds the loss targets, which always come from the base dataset."
        )
        raise ValueError(msg)


class SourceOverride:
    """A dataset that replaces part of the samples of a :class:`CompositeNativeGridDataset`.

    It replaces the grid points of one ``region`` of the base dataset (the cutout ``interior``,
    the ``boundary`` or ``all`` of them) at the time steps selected by ``times``, relative to the
    anchor t_0 of each sample: ``inputs`` (offsets <= 0), ``outputs`` (offsets > 0), ``from_t0``
    (offsets >= 0) or ``all``.

    A gridded source is looked up on valid time. A trajectories source is looked up on the
    forecast initialised at t_0, at a lead time equal to the offset of the time step, so it can
    only replace time steps from t_0 onwards. Grid points are matched to the nearest source point,
    variables by name.

    ``clip_min`` maps replaced variables to a lower bound applied to the source values before they
    replace the base values (e.g. ``{"q_500": 0.0}`` for an analysis with negative humidity).
    """

    def __init__(
        self,
        base: BaseAnemoiReader,
        dataset_config: str | dict,
        region: str,
        times: str,
        variables: list[str] | None = None,
        max_distance_km: float = 1.0,
        clip_min: dict[str, float] | None = None,
    ) -> None:
        _check_override_options(region, times)
        self.region = region
        self.times = times
        self.data = open_dataset(_normalize_dataset_config(dataset_config))
        self.is_trajectory = len(self.data.shape) == 5

        if self.is_trajectory and times in ("inputs", "all"):
            msg = (
                f"Override {self.data} is a trajectories dataset, which has no forecast steps before its "
                f"initialisation t_0, so it cannot replace times '{times}'. Use 'outputs' or 'from_t0'."
            )
            raise ValueError(msg)

        ensembles = self.data.shape[2]
        if ensembles not in (1, base.data.shape[2]):
            msg = f"Override {self.data} has {ensembles} ensemble members, the base dataset {base.data.shape[2]}."
            raise ValueError(msg)

        self._match_grid(base, max_distance_km)
        self._select_variables(base, variables)
        self._set_clip_min(clip_min)
        self._index_times()

    def _match_grid(self, base: BaseAnemoiReader, max_distance_km: float) -> None:
        """Find the grid points of the base dataset that are replaced, and their nearest source points."""
        if self.region == "all":
            self.grid_indices = np.arange(base.grid_size)
        else:
            self.grid_indices = np.flatnonzero(base.cutout_mask if self.region == "interior" else base.boundary_mask)
        self.source_grid_indices, self.matched_distance_km = _match_grid_points(
            np.asarray(base.data.latitudes)[self.grid_indices],
            np.asarray(base.data.longitudes)[self.grid_indices],
            self.data.latitudes,
            self.data.longitudes,
        )
        if self.matched_distance_km > max_distance_km:
            msg = (
                f"Override {self.data} does not have the grid points of region '{self.region}' of the base dataset: "
                f"a grid point is {self.matched_distance_km:.3f} km from the nearest source point "
                f"(max_distance_km={max_distance_km})."
            )
            raise ValueError(msg)

    def _select_variables(self, base: BaseAnemoiReader, variables: list[str] | None) -> None:
        """Select the variables that are replaced, by default all that both datasets have."""
        source_name_to_index = self.data.name_to_index
        if variables is None:
            variables = [name for name in base.variables if name in source_name_to_index]
        else:
            variables = list(variables)
            unknown = [name for name in variables if name not in base.name_to_index or name not in source_name_to_index]
            if unknown:
                msg = f"Override variables {unknown} are not in both the base dataset and {self.data}."
                raise ValueError(msg)
        if not variables:
            msg = f"Override {self.data} has no variables in common with the base dataset."
            raise ValueError(msg)

        self.variables = variables
        self.base_variable_indices = np.array([base.name_to_index[name] for name in variables], dtype=np.int64)
        self.source_variable_indices = np.array([source_name_to_index[name] for name in variables], dtype=np.int64)
        self.kept_variables = [name for name in base.variables if name not in set(variables)]
        if self.kept_variables:
            LOGGER.warning(
                "Override %s (region %s, times %s) does not replace %s: they keep the values of the base dataset.",
                self.data,
                self.region,
                self.times,
                self.kept_variables,
            )

    def _set_clip_min(self, clip_min: dict[str, float] | None) -> None:
        """Set the lower bounds per replaced variable, as a vector over the replaced variables."""
        self.clip_min = dict(_as_dict(clip_min) or {})
        unknown = [name for name in self.clip_min if name not in self.variables]
        if unknown:
            msg = f"Override clip_min variables {unknown} are not replaced by {self.data} (variables: {self.variables})."
            raise ValueError(msg)
        #: Number of source values raised to their lower bound, counted per process (for diagnostics).
        self.clipped_count = 0
        if not self.clip_min:
            self._clip_columns = None
            return
        self._clip_columns = np.array([self.variables.index(name) for name in self.clip_min], dtype=np.int64)
        self._clip_values = np.array(list(self.clip_min.values()), dtype=np.float64)

    def _clip(self, x: np.ndarray) -> np.ndarray:
        """Raise values below ``clip_min`` to it, in place; ``x`` has the variables last. NaNs are kept."""
        if self._clip_columns is None:
            return x
        columns = x[..., self._clip_columns]
        bounds = self._clip_values.astype(x.dtype)
        below = columns < bounds
        self.clipped_count += int(np.count_nonzero(below))
        x[..., self._clip_columns] = np.where(below, bounds, columns)
        return x

    def _index_times(self) -> None:
        """Index the times (base dates for trajectories) and steps, in seconds, that can be read."""
        missing = set(self.data.missing)
        source_times = self.data.base_dates if self.is_trajectory else self.data.dates
        source_times = np.asarray(source_times).astype("datetime64[s]").astype(np.int64)
        self._time_index = {int(t): i for i, t in enumerate(source_times) if i not in missing}
        self._available_times = np.fromiter(self._time_index, dtype=np.int64, count=len(self._time_index))
        if self.is_trajectory:
            steps = np.asarray(self.data.steps).astype("timedelta64[s]").astype(np.int64)
            self._step_index = {int(s): i for i, s in enumerate(steps)}

    def selects(self, offset: int) -> bool:
        """Return whether the override replaces the time step at relative ``offset`` from t_0."""
        if self.times == "inputs":
            return offset <= 0
        if self.times == "outputs":
            return offset > 0
        if self.times == "from_t0":
            return offset >= 0
        return True

    def available(self, anchor_times: np.ndarray, offsets: list[int]) -> np.ndarray:
        """Return which anchors can be read at all ``offsets``.

        Anchors are given by the valid time of their t_0 and offsets as lead times, both in seconds.
        """
        if not offsets:
            return np.ones(len(anchor_times), dtype=bool)
        if self.is_trajectory:
            if any(offset not in self._step_index for offset in offsets):
                return np.zeros(len(anchor_times), dtype=bool)
            return np.isin(anchor_times, self._available_times)
        valid_times = anchor_times[:, None] + np.asarray(offsets, dtype=np.int64)[None, :]
        return np.isin(valid_times, self._available_times).all(axis=1)

    def read(self, anchor_time: int, offsets: list[int], grid_points: np.ndarray) -> np.ndarray:
        """Read the source at ``offsets`` from ``anchor_time`` (seconds) and at the source ``grid_points``.

        Returns an array of shape ``(times, ensemble, grid_points, variables)``.
        """
        if self.is_trajectory:
            steps = [self._step_index[offset] for offset in offsets]
            x = _read_trajectory_steps(self.data, self._time_index[anchor_time], steps)
            x = x[self.source_variable_indices][..., grid_points]
            x = rearrange(x, "variables ensemble steps gridpoints -> steps ensemble gridpoints variables")
        else:
            dates = normalize_time_indices([self._time_index[anchor_time + offset] for offset in offsets])
            x = self.data[dates, :, :, :]
            x = x[:, self.source_variable_indices][..., grid_points]
            x = rearrange(x, "dates variables ensemble gridpoints -> dates ensemble gridpoints variables")
        return self._clip(np.ascontiguousarray(x))

    def tree(self, prefix: str = "") -> Tree:
        tree = Tree(f"{prefix}{self.region} grid points, {self.times} time steps")
        tree.add(f"Dataset: {self.data}")
        tree.add(f"Lookup: {'forecast initialised at t_0' if self.is_trajectory else 'valid time'}")
        tree.add(f"Grid points: {len(self.grid_indices)} (largest source distance: {self.matched_distance_km:.3f} km)")
        tree.add(f"Variables replaced: {len(self.variables)}")
        if self.clip_min:
            tree.add(f"Lower bounds: {self.clip_min}")
        tree.add(f"Variables kept from base: {self.kept_variables}")
        return tree


class CompositeNativeGridDataset(NativeGridDataset):
    """Native grid dataset whose samples are partly read from other datasets.

    The base dataset (``dataset_config``) is read as usual and provides the grid, the statistics
    and the targets. Each :class:`SourceOverride` then replaces one region of the grid at some of
    the time steps of every sample. Overrides are applied in order, so a later one wins.

    This lets a limited area model trained on ``cutout(CERRA, ERA5)`` be fine-tuned with the data it
    gets operationally: IFS analyses as inputs on the interior and the boundary and, on the boundary
    at later time steps, the IFS forecast initialised at t_0, which then provides the lateral
    boundary conditions during the rollout. The interior at later time steps holds the loss targets
    and always comes from the base dataset.
    """

    def __init__(
        self,
        dataset: str | dict | None = None,
        dataset_config: str | dict | None = None,
        start: datetime.datetime | int | None = None,
        end: datetime.datetime | int | None = None,
        sampling: dict | None = None,
        overrides: list[dict] | None = None,
    ) -> None:
        """Initialize CompositeNativeGridDataset."""
        super().__init__(dataset=dataset, dataset_config=dataset_config, start=start, end=end, sampling=sampling)
        if not overrides:
            msg = "CompositeNativeGridDataset requires at least one override."
            raise ValueError(msg)
        self.overrides = [SourceOverride(self, **_as_dict(override)) for override in overrides]
        self._times = np.asarray(self.data.dates).astype("datetime64[s]").astype(np.int64)
        self._frequency_seconds = int(self.frequency.total_seconds())
        #: Relative offsets (in positions) of the time steps of a sample, set by :meth:`compute_anchors`.
        self._relative_indices: list[int] | None = None

    def _replaced_steps(self, override: SourceOverride) -> tuple[list[int], list[int]]:
        """Return the time steps of a sample that ``override`` replaces, and their offsets from t_0 in seconds."""
        steps = [i for i, offset in enumerate(self._relative_indices) if override.selects(offset)]
        return steps, [self._relative_indices[i] * self._frequency_seconds for i in steps]

    def compute_anchors(
        self,
        relative_indices: list[int] | np.ndarray,
        sampling: dict | None = None,
    ) -> np.ndarray:
        """Return the valid anchors of the base dataset at which every override can be read."""
        anchors = super().compute_anchors(relative_indices, sampling)
        self._relative_indices = [int(offset) for offset in relative_indices]

        anchor_times = self._times[anchors[:, 1]]
        keep = np.ones(len(anchors), dtype=bool)
        for i, override in enumerate(self.overrides):
            available = override.available(anchor_times, self._replaced_steps(override)[1])
            LOGGER.info(
                "Override %d (region %s, times %s) can be read at %d of %d anchors.",
                i,
                override.region,
                override.times,
                int(available.sum()),
                len(anchors),
            )
            keep &= available
        return anchors[keep]

    def get_sample(
        self,
        sequence: int,
        positions: TimeIndices,
        grid_shard_indices: np.ndarray | slice | None = None,
    ) -> torch.Tensor:
        """Read the base sample and replace the parts that the overrides cover."""
        if self._relative_indices is None:
            msg = "compute_anchors must be called before get_sample: it sets the time steps of a sample."
            raise RuntimeError(msg)

        # A freshly read array, which is modified in place below.
        x = super().get_sample(sequence, positions, grid_shard_indices)

        if isinstance(positions, slice):
            positions = range(*positions.indices(len(self._times)))
        positions = np.atleast_1d(np.asarray(positions)).tolist()
        if len(positions) != len(self._relative_indices):
            msg = f"Expected {len(self._relative_indices)} time steps, got positions {positions}."
            raise ValueError(msg)
        anchor_time = int(self._times[positions[0] - self._relative_indices[0]])

        # Position in the (possibly sharded) sample of each grid point of the base dataset, -1 if not loaded.
        loaded = np.arange(self.grid_size)
        if grid_shard_indices is not None:
            loaded = loaded[grid_shard_indices]
        local_index = np.full(self.grid_size, -1, dtype=np.int64)
        local_index[loaded] = np.arange(len(loaded))

        for override in self.overrides:
            steps, offsets = self._replaced_steps(override)
            local = local_index[override.grid_indices]
            in_shard = local >= 0
            if not steps or not in_shard.any():
                continue

            values = override.read(anchor_time, offsets, override.source_grid_indices[in_shard])
            values = torch.from_numpy(np.ascontiguousarray(values)).to(x.dtype)
            grid = torch.from_numpy(local[in_shard])
            variables = torch.from_numpy(override.base_variable_indices)
            for k, step in enumerate(steps):
                x[step][:, grid[:, None], variables[None, :]] = values[k]

        return x

    def tree(self, prefix: str = "") -> Tree:
        tree = super().tree(prefix)
        for i, override in enumerate(self.overrides):
            tree.add(override.tree(prefix=f"Override {i}: "))
        return tree


def create_dataset(dataset_config: dict, **_kwargs) -> BaseAnemoiReader:
    """Factory function to create a data reader based on the dataset configuration."""
    dataset_config = _normalize_reader_config(dataset_config)
    trajectory_config = _as_dict(dataset_config.pop("trajectory", None))
    overrides = dataset_config.pop("overrides", None)

    if trajectory_config is not None and overrides:
        msg = "A dataset reader cannot have both 'trajectory' and 'overrides'."
        raise ValueError(msg)

    if overrides:
        LOGGER.info("Creating CompositeNativeGridDataset...")
        return CompositeNativeGridDataset(**dataset_config, overrides=list(overrides))

    if trajectory_config is not None:
        sampling = trajectory_config.get("sampling") if isinstance(trajectory_config, dict) else None
        LOGGER.info("Creating TrajectoryDataset...")
        return TrajectoryDataset(**dataset_config, sampling=_as_dict(sampling))

    LOGGER.info("Creating NativeGridDataset...")
    return NativeGridDataset(**dataset_config)
