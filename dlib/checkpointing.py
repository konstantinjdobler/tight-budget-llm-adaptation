import contextlib
import gc
import json
import os
import shutil
import time
import typing
from concurrent.futures import Future
from functools import partial
from pathlib import Path
from typing import Optional, TypedDict, Union

import lightning as L
import torch
import torch.distributed
import torch.distributed.checkpoint
import torch.distributed.checkpoint as dist_cp
from huggingface_hub import HfApi
from lightning.pytorch.loggers import WandbLogger as LightningWandbLogger
from print_on_steroids import logger as printer
from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE, Metadata
from torch.distributed.checkpoint.planner import SavePlanner
from torch.distributed.checkpoint.storage import StorageWriter
from torch.distributed.fsdp.api import (
    FullOptimStateDictConfig,
    FullStateDictConfig,
    ShardedOptimStateDictConfig,
    ShardedStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoTokenizer, MistralForCausalLM


class State(TypedDict):
    model: MistralForCausalLM
    optimizer: torch.optim.Optimizer
    hparams: dict
    iter_num: int
    step_count: int


CKPT_DIR_PATTERN = "iter-*-ckpt*"


def clean_name(n):
    """Remove common wrapper prefixes from module names."""
    return (
        n.replace("_forward_module.", "")
        .replace("_original_module.", "")
        .replace("_checkpoint_wrapped_module.", "")
        .replace("_fsdp_wrapped_module.", "")
        .replace("_orig_mod.", "")
    )


try:
    from torch.distributed.checkpoint._storage_utils import _storage_setup
except ImportError:
    pass


def my_modded_dist_save(
    state_dict: STATE_DICT_TYPE,
    *,
    checkpoint_id: Union[str, os.PathLike, None] = None,
    storage_writer: Optional[StorageWriter] = None,
    planner: Optional[SavePlanner] = None,
    process_group: Optional[torch.distributed.ProcessGroup] = None,
) -> Metadata:
    # torch._C._log_api_usage_once("torch.distributed.checkpoint.save")

    no_dist = True  # not (dist.is_available() and dist.is_initialized())
    # if no_dist:
    #     warnings.warn(
    #         "torch.distributed is unavailable or uninitialized, assuming the intent is to save in a single process."
    #     )

    # with _profile():
    storage_writer = typing.cast(StorageWriter, _storage_setup(storage_writer, checkpoint_id, reader=False))

    return torch.distributed.checkpoint.state_dict_saver._save_state_dict(
        state_dict=torch.distributed.checkpoint.state_dict_saver._stateful_to_state_dict(state_dict),
        storage_writer=storage_writer,
        process_group=process_group,
        no_dist=no_dist,
        planner=planner,
    )


"""
They changed the signature of torch.distributed.checkpoint.save in 2.4.0 nightly so that we cannot do no_dist=True anymore (why? who knows).
"""
if "2.4.0.dev" in torch.__version__:
    no_dist_dist_save = my_modded_dist_save
else:
    no_dist_dist_save = partial(torch.distributed.checkpoint.save, no_dist=True)


def dlib_save_checkpoint_hf(
    fabric: L.Fabric,
    state: State,
    out_dir: Path,
    tags: list[str] = [],
    state_dict_type=StateDictType.FULL_STATE_DICT,
    cpu_offload=True,
    pure_torch=False,
):
    """
    Must be called on ALL ranks!
    """
    from lightning.fabric.wrappers import _unwrap_objects

    state = _unwrap_objects(state)
    assert state_dict_type in [StateDictType.FULL_STATE_DICT]
    wandb_logger: LightningWandbLogger = fabric.logger
    # # if state_dict_type == StateDictType.SHARDED_STATE_DICT:
    # # wandb_logger.experiment.id is only inited on rank0, so broadcast it to other ranks
    # wandb_run_id = str(wandb_logger.experiment.id) if fabric.is_global_zero else None

    # wandb_run_id = fabric.broadcast(wandb_run_id, src=0)
    # printer.info(f"wandb_run_id: {wandb_run_id}")
    ckpt_dir = None
    if fabric.is_global_zero:
        wandb_run_id = str(wandb_logger.experiment.id)
        ckpt_dir = out_dir / wandb_run_id / f"step-{state['step_count']:07d}-ckpt"
        printer.info(f"Saving checkpoint weights to {str(ckpt_dir)!r}")

        os.makedirs(ckpt_dir, exist_ok=True)
        previous_ckpts = sorted(ckpt_dir.glob(CKPT_DIR_PATTERN))
        if len(previous_ckpts) > 0:
            printer.warning(f"Deleting previous checkpoints: {previous_ckpts}")
            for p in previous_ckpts:
                if p.is_dir():
                    shutil.rmtree(p)
                else:
                    p.unlink()
    ckpt_dir = fabric.broadcast(ckpt_dir, src=0)
    fabric.barrier()

    state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    optim_state_dict_config = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)

    ###########################
    ### Save states to disk ###
    ###########################
    # Save all keys separately to reduce peak memory usage
    metadata = {k: v for k, v in state.items() if k not in ["model", "optimizer"]}
    model = state["model"]
    if isinstance(model, DistributedDataParallel):
        printer.info("Unwrapping model from DDP")
        model = model.module
        context = contextlib.nullcontext()
    else:
        if state["hparams"]["fsdp_sharding_strategy"] == "NO_SHARD":
            context = contextlib.nullcontext()
        else:
            context = FSDP.state_dict_type(model, state_dict_type, state_dict_config, optim_state_dict_config)

    optimizer = state["optimizer"]
    with context:

        def clean_keys(save_payload):
            for k in list(save_payload.keys()):
                clean_k = clean_name(k)
                if clean_k != k:
                    save_payload[clean_k] = save_payload.pop(k)

        def save(save_key, save_payload, clean=False):
            if fabric.is_global_zero:
                if clean:
                    clean_keys(save_payload)
                save_path = ckpt_dir / f"{save_key}.pt"
                os.makedirs(save_path.parent, exist_ok=True)
                printer.info(f"Saving {save_key} to {str(save_path)!r}...")
                torch.save(save_payload, save_path)
                printer.debug(f"Saving done {str(save_path)!r}", rank0_only=False)

        save("metadata", metadata, clean=False)
        model_sd = model.state_dict()
        CLEAN = True
        if fabric.is_global_zero:
            if CLEAN:
                clean_keys(model_sd)
            kwargs = dict(safe_serialization=False, max_shard_size="999GB") if pure_torch else {}
            printer.info(f"Saving HF model to {str(ckpt_dir)!r}...")
            model.save_pretrained(ckpt_dir, state_dict=model_sd, **kwargs)
        del model_sd
        fabric.barrier()
        collected_ram = gc.collect()
        printer.debug(f"Collected RAM: {collected_ram}", rank0_only=False)

    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_optimizer_state_dict

    optim_sd_opts = StateDictOptions(cpu_offload=True, full_state_dict=True)
    optim_sd = get_optimizer_state_dict(model, optimizer, options=optim_sd_opts)
    # get optim_sd device
    try:
        key = next(iter(optim_sd["state"].keys()))
        printer.debug(f"optim_sd device: {optim_sd['state'][key]['exp_avg'].device}", rank0_only=False)
    except Exception as e:
        printer.error(f"Error getting optim_sd device: {e}")
    fabric.barrier()
    if fabric.is_global_zero:
        # ensure enough shards when small world size
        # threads = max(8 // fabric.world_size, 1)
        threads = 8
        no_dist_dist_save(
            state_dict=optim_sd,  # IMPORTANT for some reason: do not wrap in {"optimizer": "optim_sd"} => causes CUDA OOM
            storage_writer=dist_cp.FileSystemWriter(ckpt_dir / "optimizer", single_file_per_rank=True, thread_count=threads),
            # no_dist=True,
        )
        del optim_sd
    gc.collect()
    fabric.barrier()

    ###########################
    ### Log states to HF Hub ##
    ###########################
    future = None
    if fabric.is_global_zero:
        repo_id = f"kd-shared/{wandb_logger.experiment.name}"
        if not repo_id.endswith(wandb_run_id):
            repo_id += f"-{wandb_run_id}"
        printer.info(f"Logging to HF Hub under repo_id: {repo_id}")
        api = HfApi()
        api.create_repo(repo_id, repo_type="model", exist_ok=True, private=True)
        prev_notes = wandb_logger.experiment.notes or ""
        link_str = f"HF checkpoints: https://huggingface.co/{repo_id}"
        if link_str not in prev_notes:
            wandb_logger.experiment.notes = f"{prev_notes}\n{link_str}".lstrip()
        metainfo_for_ckpt = {
            "hparams": {k: str(v) if isinstance(v, Path) else v for k, v in metadata["hparams"].items()},
            "step_count": metadata["step_count"],
            "iter_num": metadata["iter_num"],
        }
        if not Path(ckpt_dir / "tokenizer.json").exists():
            tokenizer = AutoTokenizer.from_pretrained(metadata["hparams"]["tokenizer_path"])
            tokenizer.save_pretrained(ckpt_dir)
        if not Path(ckpt_dir / "README.md").exists():
            readme = f"""
# {wandb_logger.experiment.name}

W&B run url: {wandb_logger.experiment.url}
W&B run ID: {wandb_run_id}

## Metadata

```json
{json.dumps(metainfo_for_ckpt["hparams"], indent=4, ensure_ascii=False)}
```
"""
            (ckpt_dir / "README.md").write_text(readme)

        # Upload as future in background so main process is not blocked
        future: Future = api.upload_folder(
            folder_path=ckpt_dir,
            repo_id=repo_id,
            commit_message=f"step-{state['step_count']}",
            commit_description=json.dumps(metainfo_for_ckpt, indent=4, ensure_ascii=False),
            run_as_future=True,
        )

        def callback(f):
            printer.info(f"Upload done: {f.result()}")
            # Can create tags only after commit is done
            tags.append("latest")
            tags.append(f"step-{state['step_count']}")
            for tag in tags:
                api.create_tag(repo_id, tag=tag, exist_ok=True)

        future.add_done_callback(callback)
    return future


def load_optimizer_checkpoint(
    local_checkpoint_path: Path,
    fabric: L.Fabric,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    fix_compile=False,
):
    """
    `local_checkpoint_path` should be a directory containing `optimizer/` (when using `torch.distributed.checkpoint`) or `optimizer.pt`.

    `fix_compile` is a hack to fix key names since `torch.compile` adds a `_orig_mod` prefix to all keys.

    We load first load on CPU, then move to GPU&scatter.
    When doing sharded loading without cpu offload of the optimizer checkpoint with FSDP, CUDA memory usage seemed to be higher than before resuming the training.
    Probably loading on CPU first prevents memory fragementation?
    https://github.com/huggingface/transformers/issues/26186#issuecomment-1725035726
    """
    printer.info(f"Resuming training from {local_checkpoint_path}")
    checkpoint_type = get_checkpoint_type(local_checkpoint_path)

    if checkpoint_type == StateDictType.FULL_STATE_DICT:
        torch_load = None
        if fabric.is_global_zero:
            t0 = time.perf_counter()
            torch_load = torch.load(str(local_checkpoint_path / "optimizer.pt"), mmap=True)

            for p in list(torch_load["state"].keys()):
                new_key: str = p
                # because we load into model before compiling, need to always rm _orig_mod
                if True or not fix_compile:
                    new_key = new_key.replace("_orig_mod.", "")
                # new_key = "_forward_module." + new_key
                torch_load["state"][new_key] = torch_load["state"].pop(p)
            for pg in torch_load["param_groups"]:
                new_pg = [p for p in pg["params"]]
                for i, p in enumerate(pg["params"]):
                    if True or not fix_compile:
                        p = p.replace("_orig_mod.", "")
                    # new_pg[i] = "_forward_module." + p
                pg["params"] = new_pg
            print(torch_load)
            print(torch_load.keys())
            print([n for n, p in model.named_parameters()])
            from torch.distributed.fsdp.fully_sharded_data_parallel import _get_param_to_fqns

            print(_get_param_to_fqns(model))
            printer.info(f"Time to torch load optimizer: {time.perf_counter() - t0:.02f} seconds.")

        fabric.barrier()
        # fabric.save
        t1 = time.perf_counter()

        sharded_osd = FSDP.scatter_full_optim_state_dict(torch_load, model, optim=optimizer)
        optimizer.load_state_dict(sharded_osd)
        printer.info(f"Time to scatter optimizer state dict: {time.perf_counter() - t1:.02f} seconds.")
    elif checkpoint_type == StateDictType.SHARDED_STATE_DICT:
        t1 = time.perf_counter()

        from lightning.fabric.wrappers import _unwrap_objects
        from torch.distributed.checkpoint.state_dict import StateDictOptions, get_optimizer_state_dict

        unwrapped_model, unwrapped_optim = _unwrap_objects([model, optimizer])
        # NOTE: SUPER IMPORTANT to use cpu_offload=True, otherwise we take ~>10GB more CUDA RAM during training (which breaks training if optimized for throughput previously)
        optim_sd = get_optimizer_state_dict(
            unwrapped_model, unwrapped_optim, options=StateDictOptions(cpu_offload=True, full_state_dict=False)
        )

        dist_cp.state_dict_loader.load(
            optim_sd,
            storage_reader=dist_cp.filesystem.FileSystemReader(local_checkpoint_path / "optimizer"),
        )
        printer.info("load done", rank0_only=False)
        with FSDP.state_dict_type(
            model,
            StateDictType.SHARDED_STATE_DICT,
            ShardedStateDictConfig(offload_to_cpu=True),
            ShardedOptimStateDictConfig(offload_to_cpu=True),
        ):
            optim_sd = FSDP.optim_state_dict_to_load(unwrapped_model, unwrapped_optim, optim_state_dict=optim_sd)
        printer.info("optim_state_dict_to_load done", rank0_only=False)

        optimizer.load_state_dict(optim_sd)
        printer.info(f"Time to load optimizer state dict: {time.perf_counter() - t1:.02f} seconds.")
        # [KD ~ 13/01/24] set_optimizer_state_dict is bugged with KeyError when using activation checkpointing
        # set_optimizer_state_dict(unwrapped_model, unwrapped_optim, optim_state_dict=optim_sd)
        fabric.barrier()


def get_checkpoint_type(ckpt_dir: Path):
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint path {ckpt_dir} does not exist")
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(
            f"Checkpoint path {ckpt_dir} is not a directory. Checkpoint should be directory containing `model*`, `optimizer*` and `metadata*`"
        )
    if (ckpt_dir / "optimizer").exists() and (ckpt_dir / "optimizer").is_dir():
        return StateDictType.SHARDED_STATE_DICT
    elif (ckpt_dir / "optimizer.pt").exists() and (ckpt_dir / "optimizer.pt").is_file():
        return StateDictType.FULL_STATE_DICT
    else:
        raise FileNotFoundError(f"Checkpoint path {ckpt_dir} does not contain `model.pt` or `model` directory")
