# Band-limiting the latent on the hidden mesh (decoder-facetting §18.8).
#
# Why this exists. The decoder reconstructs the 5.4 km output from a hidden mesh whose
# spacing is 13.9 km (Nyquist wavelength 27.8 km), so any near-Nyquist content in the
# latent is rendered onto the fine grid as sub-Nyquist structure. Probes showed the
# processor generates that content (smoothing the latent after the processor removes 62%
# of the residual grid-scale floor, smoothing it at the encoder output only 6%), and that
# the decoder also has a full-resolution data-side path (`out = out + x_skip[1]`) able to
# carry genuine fine structure. Band-limiting the latent therefore removes the model's
# ability to *invent* fine structure without deleting anything physical — unlike
# TruncatedConnection, which low-passes the carried state itself.
#
# OPT-IN: constructed only when `model.latent_filter` is present in the config. Absent,
# the model is bit-for-bit unchanged.

import logging

import torch
from torch import Tensor
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.models.distributed.graph import gather_tensor
from anemoi.models.distributed.graph import shard_tensor
from anemoi.models.distributed.shapes import change_channels_in_shape
from anemoi.models.distributed.shapes import get_shard_shapes

LOGGER = logging.getLogger(__name__)


class LatentLowPass(nn.Module):
    """Iterated self+neighbour average over the hidden mesh's shortest edges.

    One iteration is a linear low-pass whose stopband is the modes that alternate sign
    between adjacent nodes — i.e. exactly the near-Nyquist content the decode aliases
    down. `edge_quantile` keeps only the shortest edges because the hidden mesh is
    multi-scale: its long edges would otherwise smooth non-locally.

    The operator is fixed (no learnable parameters) and linear, so it is differentiable
    and adds no capacity — the model adapts to it rather than learning around it.
    """

    def __init__(self, graph, iterations: int = 1, edge_quantile: float = 0.75) -> None:
        super().__init__()
        self.iterations = int(iterations)

        edge_index = graph.edge_index
        edge_length = torch.as_tensor(graph["edge_length"]).reshape(-1).float()
        num_nodes = int(edge_index.max()) + 1

        keep = edge_length <= torch.quantile(edge_length, edge_quantile)
        loop = torch.arange(num_nodes)
        src = torch.cat([edge_index[0][keep], loop])
        dst = torch.cat([edge_index[1][keep], loop])
        deg = torch.zeros(num_nodes).index_add_(0, dst, torch.ones(src.numel()))

        self.num_nodes = num_nodes
        self.register_buffer("src", src, persistent=False)
        self.register_buffer("dst", dst, persistent=False)
        self.register_buffer("inv_deg", (1.0 / deg).unsqueeze(-1), persistent=False)
        LOGGER.info(
            "LatentLowPass: %d hidden nodes, %d/%d edges kept (quantile %.2f), "
            "mean degree %.2f, %d iteration(s)",
            num_nodes, int(keep.sum()), edge_index.shape[1], edge_quantile,
            float(deg.mean()), self.iterations,
        )

    def _smooth(self, x: Tensor) -> Tensor:
        for _ in range(self.iterations):
            x = torch.zeros_like(x).index_add_(0, self.dst, x[self.src]) * self.inv_deg
        return x

    def forward(
        self,
        x: Tensor,
        model_comm_group: ProcessGroup | None = None,
        shard_shapes: list | None = None,
    ) -> Tensor:
        """x is (batch*nodes, channels), possibly sharded along dim 0.

        A neighbour average needs every node, so a sharded latent is gathered, smoothed
        and re-sharded. Done in float32 with autocast disabled: index_add_ over ~1.7M
        edges accumulates badly in fp16.
        """
        sharded = model_comm_group is not None and model_comm_group.size() > 1
        if sharded:
            # shard_shapes comes from the model as get_shard_shapes(x_hidden_latent, ...),
            # so its trailing dim is the NODE-ATTRIBUTE width, not the latent's channel
            # count. gather/shard validate total element counts, so the channels must be
            # substituted first -- every mapper does the same (mapper.py:414). Without this
            # the collective raises "Tensor input and output of _broadcast_oop must have
            # the same number of elements".
            base = shard_shapes if shard_shapes is not None else get_shard_shapes(
                x, 0, model_comm_group
            )
            shapes = change_channels_in_shape(base, x.shape[-1])
            x = gather_tensor(x, 0, shapes, model_comm_group)

        assert x.shape[0] % self.num_nodes == 0, (
            f"latent leading dim {x.shape[0]} is not a multiple of "
            f"{self.num_nodes} hidden nodes"
        )
        with torch.autocast(device_type=x.device.type, enabled=False):
            dtype = x.dtype
            out = torch.cat(
                [self._smooth(chunk.float()) for chunk in x.float().chunk(
                    x.shape[0] // self.num_nodes, dim=0)]
            ).to(dtype)

        if sharded:
            out = shard_tensor(out, 0, shapes, model_comm_group)
        return out
