import os
from typing import TYPE_CHECKING

import torch
from lightning import Fabric
from print_on_steroids import logger as printer

from dlib.utils import num_parameters

if TYPE_CHECKING:
    from args import TrainingArgs


def pretty_print_important_args(fabric: Fabric, args: "TrainingArgs"):
    printer.debug("-----Training Duration-----")
    printer.info(
        f"Training for {args.training_goal} steps, "
        f"eval every {args.eval_interval} steps ({args.eval_interval * args.gradient_accumulation_steps} iters), "
        f"save every {args.save_interval} steps ({args.save_interval * args.gradient_accumulation_steps} iters), "
        f"log every {args.log_interval} steps, "
        f"model profiling every {args.model_profiling_interval} steps, "
        if args.model_profiling_interval
        else "model proflining disabled, ",
        f"warmup for {args.warmup_period} steps, "
        f"lr decay from {args.max_lr} to {args.min_lr} until step {args.lr_decay_period}.",
    )
    printer.info(
        f"{args.batch_size=}, "
        f"split into {args.micro_batch_size=} on {args.num_devices=} (=> {args.micro_batch_size * args.num_devices} iter batch size). "
        f"and {args.gradient_accumulation_steps=}."
    )
    printer.info(
        f"Training for {args.training_goal} steps corresponds to "
        f"{args.training_goal * args.batch_size:,} samples, "
        f"{args.training_goal * args.batch_size * args.block_size / 1_000_000_000:,}B tokens"
    )
    printer.debug("---------------------")


def print_trainable_param_info(fabric: Fabric, model: torch.nn.Module):
    num_total_params = num_parameters(model, requires_grad=None)
    num_trainable_params = num_parameters(model, requires_grad=True)
    num_nontrainable_params = num_parameters(model, requires_grad=False)

    printer.debug("-----Param Analysis-----")
    printer.info(f"Number of trainable parameters: {num_trainable_params:,}")
    printer.info(f"Number of non trainable parameters: {num_nontrainable_params:,}")
    printer.info(f"Total parameters {num_total_params:,}")
    printer.info(
        f"Percentage of trainable parameters: {100 * num_trainable_params / (num_nontrainable_params + num_trainable_params):.2f}%"
    )
    printer.debug("---------------------")


def print_and_log_eval_results(
    fabric: Fabric,
    state: dict,
    val_metrics: dict[str, float],
    val_time: float = 0,
    preserve_tqdm=False,
):
    printer.success(
        f"Eval step {state['iter_num']}: {val_metrics['loss']=:.4f}, ",
        f"{val_metrics['perplexity']=:.2f}, "
        f"{val_metrics['per_token_nll']=:.2f}, "
        f"{val_metrics['per_doc_nll']=:.2f}, "
        f"val time: {val_time * 1000:.2f}ms",
    )

    fabric.log_dict(
        {
            "val/loss": val_metrics["loss"],
            "val/per_token_nll": val_metrics["per_token_nll"],
            "val/per_doc_nll": val_metrics["per_doc_nll"],
            "val/ppl": val_metrics["perplexity"],
            "trainer/optimizer_step": state["step_count"],
            "trainer/iter": state["iter_num"],
        }
    )


def print_and_log_train_results(
    fabric: Fabric,
    state: dict,
    loss: float,
    grad_norm: float,
    iter_time: float = 0,
    global_step_time: float = 0,
):
    printer.info(
        f"iter {state['iter_num']} step {state['step_count']}: loss {loss:.4f}, {grad_norm=:.4f}"
        f"iter time: {(iter_time) * 1000:.2f}ms global step time: {(global_step_time):.2f}s",
    )
    fabric.log_dict(
        {
            "train/loss": loss,
            "train/grad_norm": grad_norm,
            "trainer/optimizer_step": state["step_count"],
            "trainer/iter": state["iter_num"],
        }
    )


def log_slurm_info():
    # The info doesn't always seem to be in the same environment variable, so we just check all of them
    gpu_identifiers = (
        os.environ.get("SLURM_GPUS")
        or os.environ.get("SLURM_GPUS_PER_TASK")
        or os.environ.get("SLURM_JOB_GPUS")
        or os.environ.get("SLURM_STEP_GPUS")
        or len(os.environ.get("CUDA_VISIBLE_DEVICES", []))
    )
    printer.debug("-----SLURM Info-----")
    printer.info(
        f"Detected SLURM environment. SLURM Job ID: {os.environ.get('SLURM_JOB_ID')}, "
        f"SLURM Host Name: {os.environ.get('SLURM_JOB_NODELIST')}, "
        f"SLURM Job Name: {os.environ.get('SLURM_JOB_NAME')}, "
        f"SLURM GPUS: {gpu_identifiers}"
    )
    printer.debug("---------------------")


def print_mem_stats(fabric: Fabric, model: torch.nn.Module, speed_monitor, args: "TrainingArgs"):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    printer.info(f"expected bf16 memory usage from params: {num_params * 2 / 1e9:.2f} GB")
    bf16_bytes = num_params * 2
    fp32_bytes = num_params * 4
    if args.precision == "bf16-true":
        total = bf16_bytes + bf16_bytes + bf16_bytes * 2
        if args.fsdp_sharding_strategy == "FULL_SHARD":
            sharded = total / fabric.world_size
        elif args.fsdp_sharding_strategy == "SHARD_GRAD_OP":
            sharded = (bf16_bytes + bf16_bytes * 2) / fabric.world_size + bf16_bytes
        else:
            sharded = None
        optim_state_ckpt_bytes = bf16_bytes * 2
    if args.precision == "bf16-mixed":
        # master weights + bf16 weights + fp32 adam states + fp32 grads (converted from bf16)
        total = fp32_bytes + bf16_bytes + fp32_bytes * 2 + fp32_bytes
        if args.fsdp_sharding_strategy == "FULL_SHARD":
            sharded = total / fabric.world_size
        elif args.fsdp_sharding_strategy == "SHARD_GRAD_OP":
            sharded = (fp32_bytes * 2 + fp32_bytes) / fabric.world_size + bf16_bytes + fp32_bytes
        else:
            sharded = None
        optim_state_ckpt_bytes = fp32_bytes * 2
    printer.info(f"Expected {args.precision} total memory usage from params + grad + adam state: {total / 1e9:.2f} GB")
    if sharded:
        printer.info(
            f"Expected {args.precision} {args.fsdp_sharding_strategy} sharded memory usage from params + grad + adam state: {sharded / 1e9:.2f} GB"
        )
    printer.info(f"Expected {args.precision} peak checkpointing CPU RAM usage: {optim_state_ckpt_bytes / 1e9:.2f} GB")
    printer.info(f"TFLOP / sec available: {speed_monitor.hardware_flops_per_sec_promised / 1e12:.2f}")
    printer.info(f"Device type: {torch.cuda.get_device_name(fabric.device).lower()}")
    printer.info(f"Device memory: {torch.cuda.get_device_properties(fabric.device).total_memory / 1e9:.2f} GB")
    printer.info("----------------------------")
