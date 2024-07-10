import contextlib
import os
from datetime import timedelta

import torch
from lightning import Fabric


@contextlib.contextmanager
def main_process_first(
    local_rank: int,
    description="",
    active=True,
    sync_at_end: bool = False,
    infinite_barrier=False,
):
    """
    Context manager that executes the wrapped code on the main process first and then on all other processes. Useful for e.g. dataset preprocessing.
    Inspiration from Huggingface: https://github.com/huggingface/transformers/pull/12351/files
    Infinite barrier is useful for very long running data preprocessing tasks the waiting processes would run into NCCL timeout.
    Inspiration for infinite barrier from: https://github.com/Lightning-AI/pytorch-lightning/blob/e330da5870fae34339170b942095a2600fa7a95e/src/lightning/fabric/utilities/distributed.py#L403C1-L429C64
    """

    if torch.distributed.is_available() and active:
        if infinite_barrier:
            assert torch.distributed.is_gloo_available(), "InfiniteBarrier only works with GLOO backend"
            # Create a new process group with the GLOO backend with a very high timeout that makes the barrier effectively wait forever
            infinite_gloo_pg = torch.distributed.new_group(backend="gloo", timeout=timedelta(days=10))
            barrier_fn = infinite_gloo_pg.monitored_barrier
        else:
            barrier_fn = torch.distributed.barrier
        try:
            if local_rank > 0:
                print(f"Process {local_rank} | {description} | Waiting for main process...")
                barrier_fn()
            yield
        finally:
            if local_rank == 0:
                print(f"Main process | {description} | Done. Executing on parallel processes now...")
                barrier_fn()
            if sync_at_end:
                print(f"Process {local_rank} | {description} | Syncing at the end...")
                barrier_fn()
            if infinite_barrier:
                torch.distributed.destroy_process_group(infinite_gloo_pg)
    else:
        yield


@contextlib.contextmanager
def one_process_after_another(fabric: Fabric):
    """
    Context manager that staggers execution of wrapped code across processes. Useful for e.g. debugging, profiling and to reduce peak memeory / system load in some distributed workload scenarios.
    """
    try:
        for i in range(fabric.world_size):
            if fabric.global_rank == i:
                yield
            fabric.barrier()
    finally:
        print("All processes done")


def get_rank() -> int:
    """
    Wrapper around torch.distributed.get_rank() that returns 0 if torch.distributed is not available as well as looking for LightningAI's `LOCAL_RANK` environment variable..
    """
    if not torch.distributed.is_available():
        return 0  # Training on CPU
    if not torch.distributed.is_initialized():
        # LOCAL_RANK from pytorch-lightning
        rank = os.environ.get("LOCAL_RANK") or os.environ.get("RANK")
        if rank is not None:
            return int(rank)
        else:
            return 0
    else:
        return torch.distributed.get_rank()
