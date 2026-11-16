# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import sys
from typing import Iterator

from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback

from anemoi.training.losses.band_mse import BandWeightedMSELoss

LOGGER = logging.getLogger(__name__)


class BandLossMonitor(Callback):
    """Log the per-band values of every ``BandWeightedMSELoss`` in the training loss.

    A ``CombinedLoss`` returns one number, so the band terms cannot be read from the logged
    training loss. This callback logs the value each band last contributed, before its band
    weight and before the ``loss_weights`` of the combined loss, every ``every_n_steps``
    optimiser steps.

    Metric names are ``band_loss/<dataset>/<label>/<band>``. The same values are written to
    the job log, next to the training loss of that step, so that loss weights can be
    calibrated from a log file alone.
    """

    def __init__(self, config: DictConfig, every_n_steps: int = 100) -> None:
        """Initialise the band loss monitor.

        Parameters
        ----------
        config : DictConfig
            Full job configuration (passed automatically by ``get_callbacks``).
        every_n_steps : int, optional
            Logging interval in optimiser steps, by default 100.
        """
        super().__init__()
        self.config = config
        self.every_n_steps = every_n_steps

    @staticmethod
    def _iter_band_losses(module: LightningModule) -> Iterator[tuple[str, BandWeightedMSELoss]]:
        """Yield (dataset_name, loss) for every band loss found in the training loss."""
        for dataset_name, loss in module.loss.items():
            for leaf in loss.iter_leaf_losses():
                # leaves are wrapped in a LossVariableMapper when variables are selected
                inner = getattr(leaf, "loss", leaf)
                if isinstance(inner, BandWeightedMSELoss):
                    yield dataset_name, inner

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        *args,
        **kwargs,
    ) -> None:
        if trainer.global_step % self.every_n_steps != 0:
            return
        # one model group is enough; log once to avoid duplicates
        if trainer.logger is None or not trainer.is_global_zero:
            return

        metrics: dict[str, float] = {}
        for dataset_name, loss in self._iter_band_losses(pl_module):
            for band, value in loss.last_components.items():
                metrics[f"band_loss/{dataset_name}/{loss.label}/{band}"] = float(value.float().mean())

        if metrics:
            trainer.logger.log_metrics(metrics, step=trainer.global_step)
            train_loss = {
                name: float(value)
                for name, value in trainer.callback_metrics.items()
                if name.startswith("train_") and name.endswith("_step")
            }
            if trainer.progress_bar_callback is not None:
                # the progress bar (stderr) ends its line with a carriage return, not a newline:
                # without this the log record is appended to the bar's text in the job log
                sys.stderr.write("\n")
                sys.stderr.flush()
            LOGGER.info("BandLossMonitor step %d: %s | %s", trainer.global_step, metrics, train_loss)
