# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for CompositeNativeGridDataset: samples partly read from other (gridded or trajectories) datasets."""

import datetime
from collections.abc import Iterator
from unittest.mock import patch

import numpy as np
import pytest
import torch
from omegaconf import DictConfig
from omegaconf import OmegaConf
from pydantic import TypeAdapter
from pydantic import ValidationError

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.data.data_reader import CompositeNativeGridDataset
from anemoi.training.data.data_reader import create_dataset
from anemoi.training.data.multidataset import MultiDataset
from anemoi.training.schemas.dataloader import CompositeDatasetSchema
from anemoi.training.schemas.dataloader import NativeDatasetSchema
from anemoi.training.schemas.dataloader import TrajectoryDatasetSchema
from anemoi.training.tasks import Forecaster
from anemoi.training.utils.masks import Boolean1DMask

SIX_HOURS = datetime.timedelta(hours=6)

# Base grid: a cutout of 4 interior (LAM) points followed by 3 boundary (global) points.
INTERIOR = [(50.0, 4.0), (50.1, 4.1), (50.2, 4.2), (50.3, 4.3)]
BOUNDARY = [(40.0, 0.0), (40.0, 1.0), (40.0, 2.0)]
# Interior source (e.g. IFS analysis on the CERRA grid): denser, in another order.
INTERIOR_SOURCE = [(51.0, 5.0), INTERIOR[3], INTERIOR[0], (52.0, 6.0), INTERIOR[2], INTERIOR[1]]
# Global source grid (e.g. N320), shared by the boundary analysis and the forecast.
GLOBAL = [BOUNDARY[2], (0.0, 0.0), BOUNDARY[0], (10.0, 5.0), BOUNDARY[1]]

VARIABLES = ["a", "b", "c"]
RELATIVE_INDICES = [-1, 0, 1, 2]  # multistep_input=2, rollout 2, timestep 6h


def _dates(start: str, count: int, hours: int = 6) -> np.ndarray:
    return np.datetime64(start, "s") + np.arange(count) * np.timedelta64(hours, "h")


class FakeGridded:
    """Fake gridded dataset with the layout (dates, variables, ensemble, grid)."""

    def __init__(
        self,
        dates: np.ndarray,
        variables: list[str],
        points: list[tuple[float, float]],
        grids: list[int] | None = None,
        missing: set[int] | None = None,
        seed: int = 0,
    ) -> None:
        self.dates = np.asarray(dates, dtype="datetime64[s]")
        self.frequency = SIX_HOURS
        self.variables = list(variables)
        self.name_to_index = {name: i for i, name in enumerate(variables)}
        self.latitudes = np.array([lat for lat, _ in points])
        self.longitudes = np.array([lon for _, lon in points])
        self.grids = grids or [len(points)]
        self.missing = missing or set()
        self.resolution = "fake"
        self.statistics = {}
        self.shape = (len(self.dates), len(variables), 1, len(points))
        self.values = np.random.default_rng(seed).standard_normal(self.shape).astype(np.float32)

    def metadata(self) -> dict:
        return {}

    def supporting_arrays(self) -> dict:
        return {}

    def __getitem__(self, key: object) -> np.ndarray:
        return np.array(self.values[key])  # a copy, like a zarr read

    def __repr__(self) -> str:
        return "FakeGridded"


class FakeTrajectories:
    """Fake trajectories dataset with the layout (base_dates, variables, ensemble, steps, grid)."""

    def __init__(
        self,
        base_dates: np.ndarray,
        steps_hours: list[int],
        variables: list[str],
        points: list[tuple[float, float]],
        missing: set[int] | None = None,
        seed: int = 0,
    ) -> None:
        self.base_dates = np.asarray(base_dates, dtype="datetime64[s]")
        self.steps = np.array([np.timedelta64(hours, "h") for hours in steps_hours])
        self.variables = list(variables)
        self.name_to_index = {name: i for i, name in enumerate(variables)}
        self.latitudes = np.array([lat for lat, _ in points])
        self.longitudes = np.array([lon for _, lon in points])
        self.missing = missing or set()
        self.shape = (len(self.base_dates), len(variables), 1, len(self.steps), len(points))
        self.values = np.random.default_rng(seed).standard_normal(self.shape).astype(np.float32)

    def __getitem__(self, key: object) -> np.ndarray:
        return np.array(self.values[key])

    def __repr__(self) -> str:
        return "FakeTrajectories"


def _make_sources() -> dict[str, object]:
    """Fakes for the base cutout, the interior and boundary analyses and the boundary forecast.

    The base has 12 six-hourly dates from 2020-01-01T00. The boundary analysis misses 2020-01-02T06,
    and the forecast (initialised at 00 and 12 UTC, steps 0-24h) misses its 2020-01-03T00 run.
    """
    base_dates = _dates("2020-01-01T00:00:00", 12)
    return {
        "base.zarr": FakeGridded(base_dates, VARIABLES, INTERIOR + BOUNDARY, grids=[4, 3], seed=1),
        "an-interior.zarr": FakeGridded(
            _dates("2019-12-31T18:00:00", 14),
            ["c", "a"],
            INTERIOR_SOURCE,
            seed=2,
        ),
        "an-boundary.zarr": FakeGridded(base_dates, VARIABLES, GLOBAL, missing={5}, seed=3),
        "fc.zarr": FakeTrajectories(
            _dates("2020-01-01T00:00:00", 6, hours=12),
            [0, 6, 12, 18, 24],
            ["b", "a", "c"],
            GLOBAL,
            missing={4},
            seed=4,
        ),
    }


def _override(dataset: str, region: str, times: str, **kwargs) -> dict:
    return {"dataset_config": {"dataset": dataset}, "region": region, "times": times, **kwargs}


PRODUCTION_OVERRIDES = [
    _override("an-interior.zarr", "interior", "inputs"),
    _override("an-boundary.zarr", "boundary", "inputs"),
    _override("fc.zarr", "boundary", "outputs"),
]


@pytest.fixture
def sources() -> dict[str, object]:
    return _make_sources()


@pytest.fixture
def open_fake(sources: dict[str, object]) -> Iterator[None]:
    """Patch open_dataset to open the fakes by name."""

    def _open(config: str | dict, **_kwargs) -> object:
        return sources[config["dataset"] if isinstance(config, dict) else config]

    with patch("anemoi.training.data.data_reader.open_dataset", side_effect=_open):
        yield


def _make_composite(overrides: list[dict]) -> CompositeNativeGridDataset:
    return CompositeNativeGridDataset(dataset_config={"dataset": "base.zarr"}, overrides=overrides)


def _point_index(source: object, point: tuple[float, float]) -> int:
    matches = np.flatnonzero((source.latitudes == point[0]) & (source.longitudes == point[1]))
    assert len(matches) == 1
    return int(matches[0])


def _expected_gridded(source: FakeGridded, date: np.datetime64, points: list, variables: list[str]) -> np.ndarray:
    """Values of ``source`` at ``date``, shape (points, variables)."""
    t = int(np.flatnonzero(source.dates == date)[0])
    cells = [_point_index(source, p) for p in points]
    return source.values[t][[source.name_to_index[v] for v in variables]][:, 0, cells].T


def _expected_forecast(source: FakeTrajectories, base: np.datetime64, lead_hours: int, points: list) -> np.ndarray:
    """Values of the forecast initialised at ``base`` at ``lead_hours``, shape (points, VARIABLES)."""
    b = int(np.flatnonzero(source.base_dates == base)[0])
    s = int(np.flatnonzero(source.steps == np.timedelta64(lead_hours, "h"))[0])
    cells = [_point_index(source, p) for p in points]
    return source.values[b][[source.name_to_index[v] for v in VARIABLES]][:, 0, s, cells].T


def _sample(ds: CompositeNativeGridDataset, anchor: int, grid_shard_indices: object = None) -> np.ndarray:
    positions = [anchor + offset for offset in RELATIVE_INDICES]
    return ds.get_sample(0, positions, grid_shard_indices).numpy()


@pytest.mark.usefixtures("open_fake")
class TestCompositeSample:
    """Each region and time step of a sample comes from the configured source."""

    def test_sample_mixes_sources(self, sources: dict[str, object]) -> None:
        ds = _make_composite(PRODUCTION_OVERRIDES)
        ds.compute_anchors(RELATIVE_INDICES)
        anchor = 4  # t_0 = 2020-01-02T00
        x = _sample(ds, anchor)
        base = sources["base.zarr"]
        t0 = base.dates[anchor]

        assert x.shape == (4, 1, 7, 3)
        for slot, offset in enumerate(RELATIVE_INDICES):
            valid_time = base.dates[anchor + offset]
            if offset <= 0:
                # Interior inputs: "a" and "c" from the interior analysis, "b" kept from the base.
                np.testing.assert_array_equal(
                    x[slot, 0, :4][:, [0, 2]],
                    _expected_gridded(sources["an-interior.zarr"], valid_time, INTERIOR, ["a", "c"]),
                )
                np.testing.assert_array_equal(x[slot, 0, :4, 1], base.values[anchor + offset, 1, 0, :4])
                # Boundary inputs from the boundary analysis.
                np.testing.assert_array_equal(
                    x[slot, 0, 4:],
                    _expected_gridded(sources["an-boundary.zarr"], valid_time, BOUNDARY, VARIABLES),
                )
            else:
                # Interior targets from the base, boundary from the forecast initialised at t_0.
                np.testing.assert_array_equal(x[slot, 0, :4], base.values[anchor + offset, :, 0, :4].T)
                np.testing.assert_array_equal(
                    x[slot, 0, 4:],
                    _expected_forecast(sources["fc.zarr"], t0, 6 * offset, BOUNDARY),
                )

    def test_from_t0_takes_the_t0_boundary_from_the_forecast(self, sources: dict[str, object]) -> None:
        overrides = [*PRODUCTION_OVERRIDES[:2], _override("fc.zarr", "boundary", "from_t0")]
        ds = _make_composite(overrides)
        ds.compute_anchors(RELATIVE_INDICES)
        anchor = 4
        x = _sample(ds, anchor)
        base = sources["base.zarr"]
        t0 = base.dates[anchor]

        # t_-1 boundary from the analysis, t_0 boundary from forecast step 0.
        np.testing.assert_array_equal(
            x[0, 0, 4:],
            _expected_gridded(sources["an-boundary.zarr"], base.dates[anchor - 1], BOUNDARY, VARIABLES),
        )
        np.testing.assert_array_equal(x[1, 0, 4:], _expected_forecast(sources["fc.zarr"], t0, 0, BOUNDARY))
        np.testing.assert_array_equal(x[2, 0, 4:], _expected_forecast(sources["fc.zarr"], t0, 6, BOUNDARY))

    def test_forecast_only_keeps_base_inputs(self, sources: dict[str, object]) -> None:
        ds = _make_composite([_override("fc.zarr", "boundary", "outputs")])
        ds.compute_anchors(RELATIVE_INDICES)
        anchor = 4
        x = _sample(ds, anchor)
        base = sources["base.zarr"]

        np.testing.assert_array_equal(x[:2, 0], np.transpose(base.values[anchor - 1 : anchor + 1, :, 0], (0, 2, 1)))
        np.testing.assert_array_equal(
            x[3, 0, 4:],
            _expected_forecast(sources["fc.zarr"], base.dates[anchor], 12, BOUNDARY),
        )

    @pytest.mark.parametrize(
        "grid_shard_indices",
        [
            pytest.param(slice(2, 6), id="slice-across-the-cutout-split"),
            pytest.param(slice(0, 3), id="slice-interior-only"),
            pytest.param(np.array([0, 5, 6]), id="index-array"),
        ],
    )
    def test_grid_shard_matches_full_sample(self, grid_shard_indices: object) -> None:
        ds = _make_composite(PRODUCTION_OVERRIDES)
        ds.compute_anchors(RELATIVE_INDICES)
        full = _sample(ds, 2)
        shard = _sample(ds, 2, grid_shard_indices)
        np.testing.assert_array_equal(shard, full[:, :, grid_shard_indices])

    def test_slice_positions(self) -> None:
        ds = _make_composite(PRODUCTION_OVERRIDES)
        ds.compute_anchors(RELATIVE_INDICES)
        sliced = ds.get_sample(0, slice(3, 7, 1)).numpy()
        np.testing.assert_array_equal(sliced, _sample(ds, 4))

    def test_explicit_variables(self, sources: dict[str, object]) -> None:
        ds = _make_composite([_override("fc.zarr", "boundary", "outputs", variables=["c"])])
        ds.compute_anchors(RELATIVE_INDICES)
        x = _sample(ds, 4)
        base = sources["base.zarr"]
        assert ds.overrides[0].kept_variables == ["a", "b"]
        np.testing.assert_array_equal(x[2, 0, 4:, :2], base.values[5, :2, 0, 4:].T)
        np.testing.assert_array_equal(
            x[2, 0, 4:, 2],
            _expected_forecast(sources["fc.zarr"], base.dates[4], 6, BOUNDARY)[:, 2],
        )

    def test_get_sample_before_compute_anchors_raises(self) -> None:
        ds = _make_composite(PRODUCTION_OVERRIDES)
        with pytest.raises(RuntimeError, match="compute_anchors"):
            ds.get_sample(0, [3, 4, 5, 6])


@pytest.mark.usefixtures("open_fake")
class TestCompositeAnchors:
    """Anchors are kept only where every override has the data it replaces."""

    def test_anchors_need_every_source(self) -> None:
        ds = _make_composite(PRODUCTION_OVERRIDES)
        anchors = ds.compute_anchors(RELATIVE_INDICES)
        # Base anchors are positions 1..9. The forecast is initialised at 00/12 UTC: 2, 4, 6, 8.
        # The boundary analysis misses position 5 (an input of 6), the forecast its run at 8.
        np.testing.assert_array_equal(anchors, [[0, 2], [0, 4]])

    def test_anchors_follow_the_rollout(self) -> None:
        ds = _make_composite(PRODUCTION_OVERRIDES)
        # Rollout 4: leads up to 24h, the last forecast step.
        np.testing.assert_array_equal(ds.compute_anchors([-1, 0, 1, 2, 3, 4]), [[0, 2], [0, 4]])
        # Rollout 5: a 30h lead does not exist.
        assert len(ds.compute_anchors([-1, 0, 1, 2, 3, 4, 5])) == 0

    def test_gridded_only_overrides_keep_all_times_of_day(self) -> None:
        ds = _make_composite(PRODUCTION_OVERRIDES[:1])
        anchors = ds.compute_anchors(RELATIVE_INDICES)
        np.testing.assert_array_equal(anchors[:, 1], np.arange(1, 10))


@pytest.mark.usefixtures("open_fake")
class TestCompositeConfiguration:
    """Invalid override configurations are rejected."""

    @pytest.mark.parametrize("times", ["inputs", "all"])
    def test_trajectories_cannot_replace_inputs(self, times: str) -> None:
        with pytest.raises(ValueError, match="no forecast steps before its initialisation"):
            _make_composite([_override("fc.zarr", "boundary", times)])

    @pytest.mark.parametrize(("region", "times"), [("interior", "outputs"), ("all", "from_t0"), ("interior", "all")])
    def test_interior_targets_cannot_be_replaced(self, region: str, times: str) -> None:
        with pytest.raises(ValueError, match="loss targets"):
            _make_composite([_override("an-interior.zarr", region, times)])

    def test_unknown_region(self) -> None:
        with pytest.raises(ValueError, match="Unknown override region"):
            _make_composite([_override("an-boundary.zarr", "edge", "inputs")])

    def test_source_without_the_grid_points_raises(self) -> None:
        # The interior source does not have the boundary points.
        with pytest.raises(ValueError, match="does not have the grid points of region 'boundary'"):
            _make_composite([_override("an-interior.zarr", "boundary", "inputs")])

    def test_unknown_variable(self) -> None:
        with pytest.raises(ValueError, match=r"\['b'\] are not in both"):
            _make_composite([_override("an-interior.zarr", "interior", "inputs", variables=["a", "b"])])

    def test_no_overrides(self) -> None:
        with pytest.raises(ValueError, match="at least one override"):
            _make_composite([])


@pytest.mark.usefixtures("open_fake")
class TestCreateCompositeDataset:
    """create_dataset and the schema select CompositeNativeGridDataset when overrides are given."""

    def test_create_dataset_selects_composite_reader(self) -> None:
        config = DictConfig({"dataset_config": {"dataset": "base.zarr"}, "overrides": PRODUCTION_OVERRIDES})
        ds = create_dataset(config)
        assert isinstance(ds, CompositeNativeGridDataset)
        assert [override.region for override in ds.overrides] == ["interior", "boundary", "boundary"]

    def test_create_dataset_rejects_trajectory_with_overrides(self) -> None:
        config = {
            "dataset_config": {"dataset": "base.zarr"},
            "trajectory": {"sampling": None},
            "overrides": PRODUCTION_OVERRIDES,
        }
        with pytest.raises(ValueError, match="both 'trajectory' and 'overrides'"):
            create_dataset(config)

    def test_schema_round_trip(self) -> None:
        adapter = TypeAdapter(CompositeDatasetSchema | NativeDatasetSchema | TrajectoryDatasetSchema)
        config = {"dataset_config": {"dataset": "base.zarr"}, "end": 2020, "overrides": PRODUCTION_OVERRIDES}
        parsed = adapter.validate_python(config)
        assert isinstance(parsed, CompositeDatasetSchema)

        # The training config is validated, dumped to OmegaConf and then given to create_dataset.
        ds = create_dataset(OmegaConf.create(parsed.model_dump(by_alias=True)))
        assert isinstance(ds, CompositeNativeGridDataset)
        assert ds.overrides[0].variables == ["a", "c"]

    def test_schema_without_overrides_is_not_composite(self) -> None:
        adapter = TypeAdapter(CompositeDatasetSchema | NativeDatasetSchema | TrajectoryDatasetSchema)
        parsed = adapter.validate_python({"dataset_config": {"dataset": "base.zarr"}})
        assert not isinstance(parsed, CompositeDatasetSchema)

    def test_schema_rejects_unknown_times(self) -> None:
        with pytest.raises(ValidationError):
            CompositeDatasetSchema(
                dataset_config={"dataset": "base.zarr"},
                overrides=[_override("fc.zarr", "boundary", "later")],
            )


@pytest.mark.usefixtures("open_fake")
def test_rollout_takes_the_boundary_from_the_forecast(sources: dict[str, object]) -> None:
    """With forecast overrides, advancing the rollout puts the forecast on the boundary."""
    ds = _make_composite(PRODUCTION_OVERRIDES)
    multi = MultiDataset({"data": ds}, {"data": RELATIVE_INDICES})
    batch = {"data": multi[1]["data"][None]}  # anchor 4, t_0 = 2020-01-02T00
    t0 = sources["base.zarr"].dates[4]

    task = Forecaster(multistep_input=2, multistep_output=1, timestep="6h", rollout={"start": 2, "maximum": 2})
    variable_groups = DictConfig({"forcing": [], "diagnostic": [], "target": []})
    data_indices = {"data": IndexCollection(variable_groups, ds.name_to_index)}
    output_mask = {"data": Boolean1DMask({"cutout_mask": torch.from_numpy(ds.cutout_mask)}, "cutout_mask")}

    x = task.get_inputs(batch, data_indices=data_indices)
    t0_input = x["data"][:, 1].clone()
    y_pred = {"data": torch.zeros((1, 1, 1, 7, 3))}
    x = task.advance_input(
        x,
        y_pred,
        batch,
        rollout_step=0,
        data_indices=data_indices,
        output_mask=output_mask,
        grid_shard_slice={"data": None},
    )

    torch.testing.assert_close(x["data"][:, 0], t0_input)
    torch.testing.assert_close(x["data"][0, 1, 0, :4], torch.zeros(4, 3))
    np.testing.assert_array_equal(
        x["data"][0, 1, 0, 4:].numpy(),
        _expected_forecast(sources["fc.zarr"], t0, 6, BOUNDARY),
    )


def _write_trajectories_zarr(path: object, source: FakeTrajectories, variables: list[str]) -> str:
    """Write ``source`` as a trajectories-layout zarr, with ``variables`` as its variable names."""
    import zarr

    _, _, ensembles, steps, cells = source.shape
    root = zarr.open_group(str(path), mode="w")
    root.create_dataset("data", data=source.values, chunks=(1, len(variables), ensembles, 1, cells))
    root.create_dataset("base_dates", data=source.base_dates)
    root.create_dataset("steps", data=source.steps.astype("timedelta64[us]"))
    root.create_dataset("latitudes", data=source.latitudes)
    root.create_dataset("longitudes", data=source.longitudes)
    root.attrs.update(
        layout="trajectories",
        variables=list(variables),
        frequency="12h",
        steps={"start": "0h", "end": f"{6 * (steps - 1)}h", "frequency": "6h"},
        resolution="fake",
        missing_dates=[],
        field_shape=[cells],
        start_base_date=str(source.base_dates[0]),
        end_base_date=str(source.base_dates[-1]),
    )
    return str(path)


class TestRenameTrajectories:
    """``rename`` in the dataset_config of a trajectories override, applied by the real open_dataset.

    The forecast stores variable "c" as "c_fc" (like IFS ``tcw`` for the base's ``tciwv``).
    """

    @pytest.fixture
    def fc_zarr(self, tmp_path: object, sources: dict[str, object]) -> str:
        pytest.importorskip(
            "anemoi.datasets.usage.trajectories.store",
            reason="this anemoi-datasets cannot open trajectories-layout datasets",
        )
        return _write_trajectories_zarr(tmp_path / "fc.zarr", sources["fc.zarr"], ["b", "a", "c_fc"])

    @pytest.fixture(autouse=True)
    def open_fake_or_zarr(self, sources: dict[str, object]) -> Iterator[None]:
        """Open the fakes by name and anything else with the real open_dataset."""
        from anemoi.datasets import open_dataset

        def _open(config: str | dict, **kwargs) -> object:
            name = config["dataset"] if isinstance(config, dict) else config
            return sources[name] if name in sources else open_dataset(config, **kwargs)

        with patch("anemoi.training.data.data_reader.open_dataset", side_effect=_open):
            yield

    @staticmethod
    def _fc_override(fc_zarr: str, rename: dict | None = None, **kwargs) -> dict:
        dataset_config = {"dataset": fc_zarr} if rename is None else {"dataset": fc_zarr, "rename": rename}
        return {"dataset_config": dataset_config, "region": "boundary", "times": "outputs", **kwargs}

    def test_renamed_variable_is_replaced(self, fc_zarr: str, sources: dict[str, object]) -> None:
        override = self._fc_override(fc_zarr, rename={"c_fc": "c"}, variables=VARIABLES)
        ds = _make_composite([override])
        assert ds.overrides[0].is_trajectory
        assert ds.overrides[0].variables == VARIABLES
        assert ds.overrides[0].kept_variables == []

        ds.compute_anchors(RELATIVE_INDICES)
        anchor = 4  # t_0 = 2020-01-02T00, a forecast base date
        x = _sample(ds, anchor)
        t0 = sources["base.zarr"].dates[anchor]
        for slot, offset in enumerate(RELATIVE_INDICES):
            if offset > 0:
                np.testing.assert_array_equal(
                    x[slot, 0, 4:],
                    _expected_forecast(sources["fc.zarr"], t0, 6 * offset, BOUNDARY),
                )

    def test_old_name_is_unknown(self, fc_zarr: str) -> None:
        override = self._fc_override(fc_zarr, rename={"c_fc": "c"}, variables=["a", "c_fc"])
        with pytest.raises(ValueError, match=r"\['c_fc'\] are not in both"):
            _make_composite([override])

    def test_without_rename_the_variable_is_not_replaced(self, fc_zarr: str) -> None:
        # By default the variable silently keeps its base value; an explicit list raises.
        ds = _make_composite([self._fc_override(fc_zarr)])
        assert ds.overrides[0].kept_variables == ["c"]
        with pytest.raises(ValueError, match=r"\['c'\] are not in both"):
            _make_composite([self._fc_override(fc_zarr, variables=VARIABLES)])
