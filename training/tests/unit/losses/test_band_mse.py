# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import DictConfig
from pydantic import TypeAdapter
from pydantic import ValidationError

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.losses import BandWeightedMSELoss
from anemoi.training.losses import CombinedLoss
from anemoi.training.losses import get_loss_function
from anemoi.training.losses.band_mse import band_decompose
from anemoi.training.losses.band_mse import band_labels
from anemoi.training.losses.band_mse import gaussian_lowpass
from anemoi.training.losses.band_mse import gaussian_taps
from anemoi.training.losses.band_mse import inverse_variance_weights
from anemoi.training.losses.band_mse import load_band_stats
from anemoi.training.losses.band_mse import save_band_stats
from anemoi.training.losses.variable_mapper import LossVariableMapper
from anemoi.training.schemas.training import BandWeightedMSELossSchema
from anemoi.training.schemas.training import CombinedLossSchema
from anemoi.training.schemas.training import LossSchemas
from anemoi.training.utils.index_space import IndexSpace

GRID_SHAPE = (24, 32)
GRID_KM = 5.5
BANDS_KM = [22.0, 55.0, 150.0]
N_BLOCK = GRID_SHAPE[0] * GRID_SHAPE[1]
N_BOUNDARY = 5  # nodes after the regular block, as in a cutout
VARIABLES = ["z_500", "msl"]
TARGET = "anemoi.training.losses.band_mse.BandWeightedMSELoss"


def _taps() -> list[torch.Tensor]:
    return [gaussian_taps(b, GRID_KM) for b in BANDS_KM]


def _write_stats(path: Path, sigma2: np.ndarray, variables: list[str] = VARIABLES) -> str:
    save_band_stats(
        path,
        sigma2=sigma2,
        mean_sigma2=sigma2.reshape(*sigma2.shape[:2], -1).mean(-1),
        variables=variables,
        bands_km=BANDS_KM,
        grid_shape=GRID_SHAPE,
        grid_km=GRID_KM,
    )
    return str(path)


@pytest.fixture
def stats_path(tmp_path: Path) -> str:
    """Statistics with a strong smooth-to-rough contrast, at full resolution."""
    rng = np.random.default_rng(0)
    sigma2 = np.exp(rng.normal(size=(len(BANDS_KM), len(VARIABLES), *GRID_SHAPE)) * 1.5)
    return _write_stats(tmp_path / "stats.npz", sigma2)


def _band_config(stats_path: str, **overrides) -> dict:
    config = {
        "_target_": TARGET,
        "scalers": ["node_weights"],
        "predicted_variables": VARIABLES,
        "variables": VARIABLES,
        "stats_path": stats_path,
        "grid_shape": list(GRID_SHAPE),
        "grid_km": GRID_KM,
        "bands_km": BANDS_KM,
    }
    config.update(overrides)
    return config


def _node_weights() -> dict:
    weights = torch.zeros(N_BLOCK + N_BOUNDARY)
    weights[:N_BLOCK] = 1.0 / N_BLOCK
    return {"node_weights": (3, weights)}


def test_gaussian_taps_are_normalised_and_widen() -> None:
    taps = _taps()
    for t in taps:
        torch.testing.assert_close(t.sum(), torch.tensor(1.0))
        assert t.numel() % 2 == 1
    assert taps[0].numel() < taps[1].numel() < taps[2].numel()


def test_lowpass_returns_constant_field_unchanged() -> None:
    x = torch.full((3, *GRID_SHAPE), 7.0)
    for t in _taps():
        torch.testing.assert_close(gaussian_lowpass(x, t), x)


def test_lowpass_returns_linear_trend_unchanged() -> None:
    """A large-scale gradient must not leak into the bands, also at the edges."""
    y, x = torch.meshgrid(torch.arange(GRID_SHAPE[0]), torch.arange(GRID_SHAPE[1]), indexing="ij")
    ramp = (3.0 * x - 2.0 * y + 5.0)[None].float()
    for t in _taps():
        torch.testing.assert_close(gaussian_lowpass(ramp, t), ramp, atol=1e-3, rtol=0)
    bands = band_decompose(ramp.reshape(1, 1, 1, N_BLOCK, 1), _taps(), GRID_SHAPE)
    for band in bands:
        assert band.abs().max() < 1e-3


def test_band_decompose_of_constant_is_zero() -> None:
    x = torch.full((1, 1, 1, N_BLOCK, 2), -3.0)
    for band in band_decompose(x, _taps(), GRID_SHAPE):
        torch.testing.assert_close(band, torch.zeros_like(x), atol=1e-5, rtol=0)


def test_band_decompose_telescopes() -> None:
    x = torch.randn(2, 1, 1, N_BLOCK, 3)
    taps = _taps()
    bands = band_decompose(x, taps, GRID_SHAPE)
    block = x.movedim(-1, -2).reshape(-1, *GRID_SHAPE)
    remainder = gaussian_lowpass(block, taps[-1]).reshape(2, 1, 1, 3, N_BLOCK).movedim(-1, -2)
    torch.testing.assert_close(sum(bands) + remainder, x, atol=1e-5, rtol=0)


def test_band_decompose_skips_inactive_bands() -> None:
    x = torch.randn(1, 1, 1, N_BLOCK, 2)
    full = band_decompose(x, _taps(), GRID_SHAPE)
    partial = band_decompose(x, _taps(), GRID_SHAPE, active=[True, False, True])
    assert partial[1] is None
    torch.testing.assert_close(partial[0], full[0])
    torch.testing.assert_close(partial[2], full[2])


def test_band_labels() -> None:
    assert band_labels(BANDS_KM) == ["lt22", "22-55", "55-150"]


@pytest.mark.parametrize("eps_factor", [0.01, 1.0, 100.0])
def test_weights_keep_their_mean_for_any_floor(eps_factor: float) -> None:
    sigma2 = torch.rand(3, 2, *GRID_SHAPE) + 0.01
    mean_sigma2 = sigma2.mean(dim=(-2, -1))
    weights = inverse_variance_weights(sigma2, mean_sigma2, eps_factor)
    torch.testing.assert_close(weights.mean(dim=(-2, -1)), 1.0 / mean_sigma2, rtol=1e-4, atol=0)


def test_small_floor_puts_the_weight_on_smooth_points() -> None:
    sigma2 = torch.ones(1, 1, *GRID_SHAPE)
    sigma2[..., : GRID_SHAPE[1] // 2] = 0.01  # smooth half
    mean_sigma2 = sigma2.mean(dim=(-2, -1))
    # the floor is a multiple of the median, here that of the smooth half (0.01)
    sharp = inverse_variance_weights(sigma2, mean_sigma2, 0.01)
    flat = inverse_variance_weights(sigma2, mean_sigma2, 1.0e4)
    assert sharp[..., 0].mean() / sharp[..., -1].mean() > 30
    assert flat[..., 0].mean() / flat[..., -1].mean() < 1.02


def test_stats_round_trip_and_coarse_upsampling(tmp_path: Path) -> None:
    coarse = np.random.default_rng(1).random((3, 2, 6, 8)).astype(np.float32) + 0.1
    path = _write_stats(tmp_path / "coarse.npz", coarse)
    sigma2, mean_sigma2 = load_band_stats(path, ["msl"], bands_km=BANDS_KM, grid_shape=GRID_SHAPE, grid_km=GRID_KM)
    assert sigma2.shape == (3, 1, *GRID_SHAPE)
    assert mean_sigma2.shape == (3, 1)
    torch.testing.assert_close(mean_sigma2[:, 0], torch.from_numpy(coarse[:, 1].reshape(3, -1).mean(-1)))


def test_stats_made_for_other_bands_or_variables_are_refused(stats_path: str) -> None:
    with pytest.raises(ValueError, match="was made for bands"):
        load_band_stats(stats_path, VARIABLES, bands_km=[22.0, 55.0, 200.0], grid_shape=GRID_SHAPE, grid_km=GRID_KM)
    with pytest.raises(ValueError, match="different 22 km filter"):
        load_band_stats(stats_path, VARIABLES, bands_km=BANDS_KM, grid_shape=GRID_SHAPE, grid_km=11.0)
    with pytest.raises(ValueError, match="are not in the band statistics file"):
        load_band_stats(stats_path, ["10u"], bands_km=BANDS_KM, grid_shape=GRID_SHAPE, grid_km=GRID_KM)


def test_uniform_statistics_give_the_band_error_over_the_variance(tmp_path: Path) -> None:
    variance = 0.25
    path = _write_stats(tmp_path / "uniform.npz", np.full((3, 2, *GRID_SHAPE), variance))
    loss = BandWeightedMSELoss(path, VARIABLES, list(GRID_SHAPE), GRID_KM, BANDS_KM)
    scaler = _node_weights()["node_weights"]
    loss.add_scaler(scaler[0], scaler[1], name="node_weights")

    pred = torch.randn(1, 1, 1, N_BLOCK + N_BOUNDARY, 2)
    target = torch.randn(1, 1, 1, N_BLOCK + N_BOUNDARY, 2)
    bands = band_decompose((pred - target)[..., :N_BLOCK, :], _taps(), GRID_SHAPE)
    expected = sum(b.square().mean() for b in bands) / variance

    torch.testing.assert_close(loss(pred, target), expected, rtol=1e-4, atol=0)
    assert list(loss.last_components) == ["lt22", "22-55", "55-150"]
    torch.testing.assert_close(sum(loss.last_components.values()), expected, rtol=1e-4, atol=0)


def test_band_weights_select_bands(stats_path: str) -> None:
    scaler = _node_weights()["node_weights"]
    losses = []
    for band_weights in ([1.0, 1.0, 1.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]):
        loss = BandWeightedMSELoss(stats_path, VARIABLES, list(GRID_SHAPE), GRID_KM, BANDS_KM, band_weights)
        loss.add_scaler(scaler[0], scaler[1], name="node_weights")
        losses.append(loss)
    assert not hasattr(losses[1], "weights_1")

    pred = torch.randn(1, 1, 1, N_BLOCK + N_BOUNDARY, 2)
    target = torch.randn(1, 1, 1, N_BLOCK + N_BOUNDARY, 2)
    values = [loss(pred, target) for loss in losses]
    torch.testing.assert_close(values[0], values[1] + values[2] / 2.0 + values[3], rtol=1e-4, atol=0)


def test_boundary_nodes_do_not_contribute(stats_path: str) -> None:
    loss = BandWeightedMSELoss(stats_path, VARIABLES, list(GRID_SHAPE), GRID_KM, BANDS_KM)
    loss.add_scaler(3, torch.full((N_BLOCK + N_BOUNDARY,), 1.0 / N_BLOCK), name="node_weights")
    pred = torch.randn(1, 1, 1, N_BLOCK + N_BOUNDARY, 2)
    target = torch.randn(1, 1, 1, N_BLOCK + N_BOUNDARY, 2)
    moved = pred.clone()
    moved[..., N_BLOCK:, :] += 100.0
    torch.testing.assert_close(loss(pred, target), loss(moved, target))


def test_weights_stay_out_of_the_state_dict(stats_path: str) -> None:
    loss = BandWeightedMSELoss(stats_path, VARIABLES, list(GRID_SHAPE), GRID_KM, BANDS_KM)
    assert loss.weights_0.shape == (N_BLOCK, len(VARIABLES))
    assert not any(key.startswith("weights_") for key in loss.state_dict())


def test_gradients_are_finite_under_autocast(stats_path: str) -> None:
    loss = BandWeightedMSELoss(stats_path, VARIABLES, list(GRID_SHAPE), GRID_KM, BANDS_KM)
    scaler = _node_weights()["node_weights"]
    loss.add_scaler(scaler[0], scaler[1], name="node_weights")
    pred = torch.randn(1, 1, 1, N_BLOCK + N_BOUNDARY, 2, requires_grad=True)
    target = torch.randn(1, 1, 1, N_BLOCK + N_BOUNDARY, 2)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        value = loss(pred, target)
    assert value.dtype == torch.float32
    value.backward()
    assert torch.isfinite(pred.grad).all()
    assert pred.grad[..., :N_BLOCK, :].abs().sum() > 0
    assert pred.grad[..., N_BLOCK:, :].abs().sum() == 0


def test_a_grid_shard_is_refused(stats_path: str) -> None:
    loss = BandWeightedMSELoss(stats_path, VARIABLES, list(GRID_SHAPE), GRID_KM, BANDS_KM)
    assert loss.supports_sharding is False
    x = torch.zeros(1, 1, 1, N_BLOCK, 2)
    with pytest.raises(RuntimeError, match="needs the full grid"):
        loss(x, x, grid_shard_slice=slice(0, 10))


@pytest.fixture
def data_indices() -> IndexCollection:
    data_config = {"forcing": ["lsm"], "diagnostic": [], "target": []}
    return IndexCollection(DictConfig(data_config), {"z_500": 0, "lsm": 1, "10u": 2, "msl": 3})


def test_combined_with_zero_weight_equals_mse(stats_path: str, data_indices: IndexCollection) -> None:
    mse_config = {"_target_": "anemoi.training.losses.MSELoss", "scalers": ["node_weights"]}
    combined = get_loss_function(
        DictConfig(
            {
                "_target_": "anemoi.training.losses.combined.CombinedLoss",
                "scalers": ["node_weights"],
                "loss_weights": [1.0, 0.0],
                "losses": [mse_config, _band_config(stats_path)],
            },
        ),
        scalers=_node_weights(),
        data_indices=data_indices,
    )
    mse = get_loss_function(DictConfig(mse_config), scalers=_node_weights(), data_indices=data_indices)

    assert isinstance(combined, CombinedLoss)
    band = combined.losses[1]
    assert isinstance(band, LossVariableMapper)
    assert isinstance(band.loss, BandWeightedMSELoss)
    assert band.predicted_variables == VARIABLES
    # one leaf without sharding support makes the trainer gather the grid for the whole loss
    assert [leaf.supports_sharding for leaf in combined.iter_leaf_losses()] == [True, False]

    pred = torch.randn(1, 1, 1, N_BLOCK + N_BOUNDARY, 3)  # model output: z_500, 10u, msl
    target = torch.randn(1, 1, 1, N_BLOCK + N_BOUNDARY, 4)  # full data: z_500, lsm, 10u, msl
    layouts = {"pred_layout": IndexSpace.MODEL_OUTPUT, "target_layout": IndexSpace.DATA_FULL}
    torch.testing.assert_close(combined(pred, target, **layouts), mse(pred, target, **layouts))
    assert set(band.loss.last_components) == {"lt22", "22-55", "55-150"}


def test_mapper_selects_the_variables_and_scatters_per_variable_values(
    stats_path: str,
    data_indices: IndexCollection,
) -> None:
    loss = get_loss_function(
        DictConfig(_band_config(stats_path)),
        scalers=_node_weights(),
        data_indices=data_indices,
    )
    pred = torch.randn(1, 1, 1, N_BLOCK + N_BOUNDARY, 3)
    target = torch.randn(1, 1, 1, N_BLOCK + N_BOUNDARY, 4)
    layouts = {"pred_layout": IndexSpace.MODEL_OUTPUT, "target_layout": IndexSpace.DATA_FULL}

    per_variable = loss(pred, target, squash=False, **layouts)
    assert per_variable.shape == (3,)
    assert per_variable[1] == 0  # 10u is not in the loss
    torch.testing.assert_close(loss(pred, target, **layouts), per_variable[[0, 2]].mean(), rtol=1e-4, atol=0)

    # changing the unselected variable changes nothing
    moved = pred.clone()
    moved[..., 1] += 10.0
    torch.testing.assert_close(loss(moved, target, **layouts), loss(pred, target, **layouts))


def test_schema_keeps_every_argument(stats_path: str) -> None:
    config = _band_config(stats_path, band_weights=[1.0, 0.0, 0.0], eps_factor=0.1, label="B")
    schema = BandWeightedMSELossSchema(**config)
    dumped = schema.model_dump(by_alias=True)
    for key, value in config.items():
        assert dumped[key] == value, key

    combined = {
        "_target_": "anemoi.training.losses.combined.CombinedLoss",
        "scalers": ["node_weights"],
        "loss_weights": [1.0, 0.5],
        "losses": [{"_target_": "anemoi.training.losses.MSELoss", "scalers": ["node_weights"]}, config],
    }
    validated = TypeAdapter(LossSchemas).validate_python(combined)
    assert isinstance(validated, CombinedLossSchema)
    assert isinstance(validated.losses[1], BandWeightedMSELossSchema)
    assert validated.losses[1].stats_path == stats_path
    assert validated.loss_weights == [1.0, 0.5]


def test_schema_refuses_inconsistent_configs(stats_path: str) -> None:
    with pytest.raises(ValidationError, match="identical to predicted_variables"):
        BandWeightedMSELossSchema(**_band_config(stats_path, variables=["msl", "z_500"]))
    with pytest.raises(ValidationError, match="one entry per band"):
        BandWeightedMSELossSchema(**_band_config(stats_path, band_weights=[1.0, 1.0]))


def test_monitor_logs_every_band_of_every_group(stats_path: str, data_indices: IndexCollection) -> None:
    from unittest.mock import MagicMock

    from anemoi.training.diagnostics.callbacks.band_loss import BandLossMonitor

    combined = get_loss_function(
        DictConfig(
            {
                "_target_": "anemoi.training.losses.combined.CombinedLoss",
                "scalers": ["node_weights"],
                "loss_weights": [1.0, 0.0, 0.0],
                "losses": [
                    {"_target_": "anemoi.training.losses.MSELoss", "scalers": ["node_weights"]},
                    _band_config(stats_path, label="A"),
                    _band_config(
                        stats_path,
                        label="B",
                        predicted_variables=["msl"],
                        variables=["msl"],
                        band_weights=[1.0, 0.0, 0.0],
                    ),
                ],
            },
        ),
        scalers=_node_weights(),
        data_indices=data_indices,
    )
    pred = torch.randn(1, 1, 1, N_BLOCK + N_BOUNDARY, 3)
    target = torch.randn(1, 1, 1, N_BLOCK + N_BOUNDARY, 4)
    combined(pred, target, pred_layout=IndexSpace.MODEL_OUTPUT, target_layout=IndexSpace.DATA_FULL)

    module = MagicMock()
    module.loss = {"data": combined}
    trainer = MagicMock()
    trainer.global_step = 20
    trainer.is_global_zero = True
    trainer.callback_metrics = {"train_multi_dataset_loss_step": torch.tensor(0.5), "val_x": torch.tensor(1.0)}

    monitor = BandLossMonitor(config=DictConfig({}), every_n_steps=10)
    monitor.on_train_batch_end(trainer, module)
    logged = trainer.logger.log_metrics.call_args.args[0]
    assert set(logged) == {
        "band_loss/data/A/lt22",
        "band_loss/data/A/22-55",
        "band_loss/data/A/55-150",
        "band_loss/data/B/lt22",
    }
    assert all(value > 0 for value in logged.values())

    trainer.global_step = 25
    trainer.logger.log_metrics.reset_mock()
    monitor.on_train_batch_end(trainer, module)
    trainer.logger.log_metrics.assert_not_called()


def test_variable_scaler_is_filtered_to_the_selected_variables(stats_path: str, data_indices: IndexCollection) -> None:
    """A level ramp given for all model outputs must reach the band loss as the selected entries only."""
    scalers = {**_node_weights(), "level": (4, torch.tensor([2.0, 7.0, 11.0]))}  # z_500, 10u, msl
    plain = get_loss_function(DictConfig(_band_config(stats_path)), scalers=scalers, data_indices=data_indices)
    scaled = get_loss_function(
        DictConfig(_band_config(stats_path, scalers=["node_weights", "level"])),
        scalers=scalers,
        data_indices=data_indices,
    )
    pred = torch.randn(1, 1, 1, N_BLOCK + N_BOUNDARY, 3)
    target = torch.randn(1, 1, 1, N_BLOCK + N_BOUNDARY, 4)
    layouts = {"pred_layout": IndexSpace.MODEL_OUTPUT, "target_layout": IndexSpace.DATA_FULL}
    expected = plain(pred, target, squash=False, **layouts) * torch.tensor([2.0, 0.0, 11.0])
    torch.testing.assert_close(scaled(pred, target, squash=False, **layouts), expected, rtol=1e-4, atol=0)


def test_variable_scaling_report_handles_a_loss_without_variable_scaler(
    stats_path: str,
    data_indices: IndexCollection,
) -> None:
    """The trainer logs the variable scaling of every sub-loss at start-up (smoke run 22812622 died here)."""
    from anemoi.training.losses.utils import print_variable_scaling

    combined = get_loss_function(
        DictConfig(
            {
                "_target_": "anemoi.training.losses.combined.CombinedLoss",
                "scalers": ["node_weights"],
                "loss_weights": [1.0, 0.0],
                "losses": [{"_target_": "anemoi.training.losses.MSELoss", "scalers": ["node_weights"]},
                           _band_config(stats_path)],
            },
        ),
        scalers=_node_weights(),
        data_indices=data_indices,
    )
    report = print_variable_scaling(combined, data_indices)
    band = report["LossVariableMapper_2"]  # both sub-losses are wrapped; the band loss is the second
    assert band["z_500"] == band["msl"] == 1.0
    assert band["total_sum"] == 2.0


def test_monitor_starts_its_log_line_on_a_new_line(
    stats_path: str,
    data_indices: IndexCollection,
    capsys: pytest.CaptureFixture,
) -> None:
    """The progress bar ends with a carriage return; the monitor's log record must not be glued to it."""
    from unittest.mock import MagicMock

    from anemoi.training.diagnostics.callbacks.band_loss import BandLossMonitor

    loss = get_loss_function(DictConfig(_band_config(stats_path)), scalers=_node_weights(), data_indices=data_indices)
    loss(
        torch.randn(1, 1, 1, N_BLOCK + N_BOUNDARY, 3),
        torch.randn(1, 1, 1, N_BLOCK + N_BOUNDARY, 4),
        pred_layout=IndexSpace.MODEL_OUTPUT,
        target_layout=IndexSpace.DATA_FULL,
    )
    module = MagicMock()
    module.loss = {"data": loss}
    trainer = MagicMock(global_step=10, is_global_zero=True, callback_metrics={})
    BandLossMonitor(config=DictConfig({}), every_n_steps=10).on_train_batch_end(trainer, module)
    assert capsys.readouterr().err.startswith("\n")
