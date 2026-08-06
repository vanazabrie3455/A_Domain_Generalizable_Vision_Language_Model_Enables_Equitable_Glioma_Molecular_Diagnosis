import os
from dataclasses import dataclass
from datetime import timedelta
from typing import TypeVar

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset, DistributedSampler

ModuleType = TypeVar("ModuleType", bound=nn.Module)
ItemType = TypeVar("ItemType")


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def primary(self) -> bool:
        return self.rank == 0


def environment_context() -> DistributedContext:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    return DistributedContext(rank, local_rank, world_size, device)


def initialize(timeout_minutes: int = 30) -> DistributedContext:
    context = environment_context()
    if context.world_size > 1 and not dist.is_initialized():
        backend = "nccl" if context.device.type == "cuda" else "gloo"
        if context.device.type == "cuda":
            torch.cuda.set_device(context.device)
        dist.init_process_group(
            backend=backend,
            rank=context.rank,
            world_size=context.world_size,
            timeout=timedelta(minutes=timeout_minutes),
        )
    return context


def shutdown() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def wrap_model(model: ModuleType, context: DistributedContext) -> nn.Module:
    model = model.to(context.device)
    if context.world_size == 1:
        return model
    device_ids = [context.local_rank] if context.device.type == "cuda" else None
    return DistributedDataParallel(model, device_ids=device_ids, broadcast_buffers=True)


def distributed_sampler(
    dataset: Dataset[ItemType],
    context: DistributedContext,
    shuffle: bool,
    seed: int,
) -> DistributedSampler[ItemType] | None:
    if context.world_size == 1:
        return None
    return DistributedSampler(
        dataset,
        num_replicas=context.world_size,
        rank=context.rank,
        shuffle=shuffle,
        seed=seed,
        drop_last=False,
    )


def reduce_mean(value: Tensor, context: DistributedContext) -> Tensor:
    if context.world_size == 1:
        return value
    result = value.clone()
    dist.all_reduce(result, op=dist.ReduceOp.SUM)
    return result / context.world_size


def gather_variable_tensor(value: Tensor, context: DistributedContext) -> list[Tensor]:
    if context.world_size == 1:
        return [value]
    local_size = torch.tensor([value.shape[0]], device=value.device, dtype=torch.long)
    size_buffers = [torch.zeros_like(local_size) for _ in range(context.world_size)]
    dist.all_gather(size_buffers, local_size)
    sizes = [int(item.item()) for item in size_buffers]
    maximum = max(sizes)
    padded_shape = (maximum, *value.shape[1:])
    padded = torch.zeros(padded_shape, dtype=value.dtype, device=value.device)
    padded[: value.shape[0]] = value
    gathered = [torch.zeros_like(padded) for _ in range(context.world_size)]
    dist.all_gather(gathered, padded)
    return [tensor[:size].cpu() for tensor, size in zip(gathered, sizes, strict=True)]
