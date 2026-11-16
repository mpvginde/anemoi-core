# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""The trainer gathers a grid-sharded batch when a loss cannot work on shards.

Prediction and target have different variable counts (model output vs full data), so each
must be gathered with its own shard shapes. Using the prediction's for both crashed the
band-weighted loss smoke run (job 22813474) with a GPU memory access fault.
"""
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from anemoi.training.train.methods.base import BaseTrainingModule

GRID, N_PRED, N_TARGET = 10, 3, 5


def _worker(rank: int, world: int, sizes: list[int], tmp: str) -> None:
    dist.init_process_group("gloo", init_method=f"file://{tmp}/pg", rank=rank, world_size=world)
    full_pred = torch.arange(GRID * N_PRED, dtype=torch.float32).reshape(1, 1, 1, GRID, N_PRED)
    full_target = 1000 + torch.arange(GRID * N_TARGET, dtype=torch.float32).reshape(1, 1, 1, GRID, N_TARGET)
    start = sum(sizes[:rank])
    shard = slice(start, start + sizes[rank])
    module = SimpleNamespace(
        grid_shard_slice={"data": shard},
        grid_shard_shapes={"data": sizes},
        loss_supports_sharding=False,
        metrics_support_sharding=True,
        grid_dim=-2,
        model_comm_group=dist.group.WORLD,
    )
    pred, target, grid_slice = BaseTrainingModule._prepare_tensors_for_loss(
        module,
        full_pred[..., shard, :],
        full_target[..., shard, :],
        dataset_name="data",
    )
    ok = torch.equal(pred, full_pred) and torch.equal(target, full_target) and grid_slice is None
    Path(tmp, f"rank{rank}").write_text("ok" if ok else f"pred {tuple(pred.shape)} target {tuple(target.shape)}")
    dist.destroy_process_group()


@pytest.mark.parametrize("sizes", [[5, 5], [6, 4]], ids=["equal shards", "unequal shards"])
def test_prediction_and_target_are_gathered_with_their_own_shapes(tmp_path: Path, sizes: list[int]) -> None:
    # fork: the children reuse the parent's imports (a fresh import of methods.base alone is circular)
    mp.start_processes(_worker, args=(2, sizes, str(tmp_path)), nprocs=2, join=True, start_method="fork")
    assert [Path(tmp_path, f"rank{r}").read_text() for r in range(2)] == ["ok", "ok"]
