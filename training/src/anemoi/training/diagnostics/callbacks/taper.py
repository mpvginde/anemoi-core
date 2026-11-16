# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from typing import Iterator

import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback

LOGGER = logging.getLogger(__name__)


class TaperMonitor(Callback):
    """Log the per-head decoder distance-taper length-scale during training.

    The learnable decoder taper (see decoder-facetting investigation) adds a
    per-head pre-softmax logit bias to the decoder attention. This callback scans
    the model for the taper parameters and logs their effective per-head value to
    the experiment logger every ``every_n_steps`` optimiser steps, so it is easy to
    see whether the model actually moves the length-scale away from its init.

    - For ``taper_kind: rbf`` it logs sigma_h = exp(taper_log_sigma_h).
    - For ``taper_kind: alibi`` it logs slope_h = softplus(taper_raw_slope_h).

    Metric names are ``taper/<param_path>/head_<h>`` plus ``.../mean``.
    """

    def __init__(self, config: DictConfig, every_n_steps: int = 100) -> None:
        """Initialise the taper monitor.

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
    def _iter_taper_values(module: LightningModule) -> Iterator[tuple[str, torch.Tensor]]:
        """Yield (param_path, per_head_value) for every taper parameter found."""
        for name, param in module.named_parameters():
            if name.endswith("taper_log_sigma"):
                yield name, param.detach().float().exp()
            elif name.endswith("taper_raw_slope"):
                yield name, torch.nn.functional.softplus(param.detach().float())

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        *args,
        **kwargs,
    ) -> None:
        if trainer.global_step % self.every_n_steps != 0:
            return
        # parameters are identical across ranks; log once to avoid duplicates
        if trainer.logger is None or not trainer.is_global_zero:
            return

        metrics: dict[str, float] = {}
        for name, values in self._iter_taper_values(pl_module):
            prefix = "taper/" + name.replace(".", "/")
            for head, value in enumerate(values.tolist()):
                metrics[f"{prefix}/head_{head}"] = value
            metrics[f"{prefix}/mean"] = float(values.mean())

        if metrics:
            trainer.logger.log_metrics(metrics, step=trainer.global_step)
