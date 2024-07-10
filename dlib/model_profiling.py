import time

import torch
import torch.distributed as dist
import wandb
from tqdm.asyncio import tqdm


def clean_name(n):
    """Remove common wrapper prefixes from module names for cleaner logging."""
    return (
        n.replace("_forward_module.", "")
        .replace("_original_module.", "")
        .replace("_checkpoint_wrapped_module.", "")
        .replace("_fsdp_wrapped_module.", "")
        .replace("_orig_mod.", "")
    )


def log_stats(result, prefix, name):
    counts, bins, l2_norm, l1_norm = result
    wandb.run._log(
        {
            f"{prefix}/{name}": wandb.Histogram(np_histogram=(counts.tolist(), bins.tolist())),
            f"{prefix}-norm/{name}": l2_norm.item(),
            # f"{prefix}-l1-norm/{name}": l1_norm.item(), # don't log, it's too much clutter and a bit annoying
        },
        commit=False,
    )


NUM_BINS = 64


def sharded_calculate_and_log_stats(tensor: torch.Tensor, name: str, prefix: str, device: torch.device, rank: int):
    """
    Calculate tensor statistics in a work-then-reduce fashion.
    We first calculate local stats for parameter shards on each rank and then reduce to rank0, where we log.
    """

    def get_sharded_min_max(tensor: torch.Tensor):
        """
        Need to get overall min/max across all ranks before calculating histogram.
        """
        tensor_is_sharded_elsewhere = tensor is None or tensor.numel() == 0
        if not tensor_is_sharded_elsewhere:
            min, inverted_max = tensor.min().to(torch.float32), tensor.max().to(torch.float32) * -1
            to_reduce = torch.stack([min, inverted_max])
        else:
            # use a large positive number here in case max is negative
            to_reduce = torch.full([2], 1337.0, device=device, dtype=torch.float32)

        # inverted_max hack to fuse into single all_reduce call
        dist.all_reduce(to_reduce, op=dist.ReduceOp.MIN)
        min, max = to_reduce[0], to_reduce[1] * -1

        return min, max

    min, max = get_sharded_min_max(tensor)
    tensor_is_sharded_elsewhere = tensor is None or tensor.numel() == 0
    if not tensor_is_sharded_elsewhere:
        tensor = tensor.reshape(-1).to(torch.float32)
        counts = torch.histc(tensor, NUM_BINS, min=min, max=max)
        # trick: use squared l2 norms so that we can reduce across shards w/ SUM, do sqrt later
        squared_l2_norm = torch.norm(tensor, p=2.0).pow(2.0).unsqueeze(0)
        l1_norm = torch.norm(tensor, p=1.0).unsqueeze(0)
    else:
        counts, squared_l2_norm, l1_norm = (
            torch.zeros(NUM_BINS, device=device),
            torch.zeros(1, device=device),
            torch.zeros(1, device=device),
        )

    # assert its zero-dim tensor
    # assert squared_l2_norm.size() == torch.Size([])
    # assert l1_norm.size() == torch.Size([])
    # Concatenate all stats to reduce in a single all_reduce call
    counts_and_squared_l2_norm_and_l1_norm = torch.cat([counts, squared_l2_norm, l1_norm])
    dist.reduce(counts_and_squared_l2_norm_and_l1_norm, dst=0, op=dist.ReduceOp.SUM)
    counts, squared_l2_norm, l1_norm = counts_and_squared_l2_norm_and_l1_norm.split([NUM_BINS, 1, 1])

    if rank == 0:
        bins = torch.linspace(min, max, NUM_BINS + 1)
        result = [counts, bins, squared_l2_norm.sqrt(), l1_norm]
        log_stats(result, prefix, name)


def calculate_and_log_stats(tensor: torch.Tensor, name: str, prefix: str):
    tensor = tensor.reshape(-1).to(torch.float32)
    min, max = tensor.min(), tensor.max()
    counts = torch.histc(tensor, NUM_BINS, min=min, max=max)
    bins = torch.linspace(min, max, NUM_BINS + 1)
    l2_norm = tensor.norm(2.0)
    l1_norm = tensor.norm(1.0)
    result = [counts, bins, l2_norm, l1_norm]
    log_stats(result, prefix, name)


def log_param_stats(
    param: torch.nn.Parameter,
    name: str,
    rank: int,
    device: torch.device = None,
    log_weights=True,
    log_grads=True,
    sharded_grads=True,
    sharded_weights=True,
):
    weight = param.data
    grad = param.grad
    device = device if device is not None else weight.device

    if log_weights:
        if sharded_weights:
            sharded_calculate_and_log_stats(weight, name, prefix="parameters", device=device, rank=rank)
        elif rank == 0:
            calculate_and_log_stats(weight, name, prefix="parameters")

    if log_grads:
        if sharded_grads:
            sharded_calculate_and_log_stats(grad, name, prefix="gradients", device=device, rank=rank)
        elif rank == 0:
            calculate_and_log_stats(grad, name, prefix="gradients")


@torch.no_grad()
def log_model_stats_to_wandb(
    model: torch.nn.Module,
    log_weights=True,
    log_grads=True,
    rank=None,
    sharded_grads=True,
    sharded_weights=True,
) -> float:
    """
    Log parameters and gradients of a model to wandb. Handles sharded parameters for FSDP (we calculate tensor histograms and norms in the local rank and then gather to rank0).

    Should be called after backward pass and before `optimizer.zero_grad()`. We set `commit=False` for logging to `wandb` for efficiency so the actual logging to the wandb servers happens on your next `wandb.log` call.

    TODO: we can make this very efficient by first caluclating the local results for ALL parameters and then reducing them in a single all_reduce call. This will reduce the number of all_reduce calls from 2*num_tensors down to 1 for logging both grads and params.

    Returns: time spent in this function (seconds).
    """
    tracking_t0 = time.perf_counter()
    if dist.is_initialized() and rank is None:
        rank = dist.get_rank()
    else:
        rank = rank if rank is not None else 0
    for n, p in tqdm(model.named_parameters(), desc="Logging model states", leave=False):
        if p.requires_grad:
            log_param_stats(
                p,
                name=clean_name(n),
                rank=rank,
                log_grads=log_grads,
                log_weights=log_weights,
                sharded_grads=sharded_grads,
                sharded_weights=sharded_weights,
            )
    tracking_t1 = time.perf_counter()
    return tracking_t1 - tracking_t0
