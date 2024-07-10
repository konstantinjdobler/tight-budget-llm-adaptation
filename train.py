import dataclasses
import os
import time
from pathlib import Path

import lightning as L
import simple_parsing
import torch
import transformers
import transformers.models
from lightning.fabric.plugins.environments import LightningEnvironment, SLURMEnvironment
from lightning.fabric.strategies import FSDPStrategy
from lightning.fabric.strategies.strategy import (
    _Sharded,
)
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.loggers import WandbLogger as PytorchLightningWandbLogger
from print_on_steroids import logger as printer
from print_on_steroids.print import graceful_exceptions
from torch.distributed.fsdp import StateDictType
from torch.utils.data import DataLoader
from tqdm.asyncio import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, MistralForCausalLM, PreTrainedModel
from transformers.models.mistral.modeling_mistral import (
    MistralRMSNorm,
    MistralRotaryEmbedding,
)

from args import TrainingArgs as Args
from dlib import (
    SpeedMonitorFabric,
    get_dataloaders,
    log_model_stats_to_wandb,
    measure_model_flops,
    pretty_str_from_dict,
    wait_for_debugger,
)
from dlib.checkpointing import State, dlib_save_checkpoint_hf, load_optimizer_checkpoint
from dlib.dist_utils import main_process_first
from dlib.get_optimizer import get_optimizer
from dlib.lr_schedules import CosineDecayScheduler, InfiniteLRScheduler, LRScheduler
from helpers.entrypoints import deepfocus_init_, wechsel_init_
from helpers.mokey_patch_fa_packing import monkey_patch_packing_mistral
from helpers.printers import log_slurm_info, pretty_print_important_args, print_mem_stats, print_trainable_param_info

WANDB_PROJECT = "tight-budget-llm-adaptation"
WANDB_ENTITY = "konstantinjdobler"

print("import done")


def setup(args: Args) -> None:
    print("setup", os.environ.get("LOCAL_RANK"))
    args.out_dir = (args.out_dir / args.run_name).resolve()
    IS_ON_SLURM = SLURMEnvironment().detect()
    cluster_environment = None
    if IS_ON_SLURM:
        # do this as workaround check, since fabric.local_rank is not available yet
        if os.environ.get("LOCAL_RANK") is None:
            printer.info("Disabling SLURMEnvironment (we use lightning's native DDP launcher)")
            log_slurm_info()
        cluster_environment = LightningEnvironment()

    # Distributed setup
    precision = args.precision
    if args.use_fsdp:
        assert args.accelerator == "cuda"
        from transformers.models.mistral.modeling_mistral import MistralDecoderLayer

        activation_checkpointing_policy = {MistralDecoderLayer} if args.activation_checkpointing else None
        # need to create for FSDP meta device init because it's not already implemented for MistralRMSNorm & MistralRotaryEmbedding
        MistralRMSNorm.reset_parameters = lambda self: None
        MistralRotaryEmbedding.reset_parameters = lambda self: None

        strategy = FSDPStrategy(
            auto_wrap_policy={MistralDecoderLayer},
            activation_checkpointing_policy=activation_checkpointing_policy,
            state_dict_type="full",
            sync_module_states=True,  # Make sure all ranks have the same model weights
            use_orig_params=True,
            sharding_strategy=args.fsdp_sharding_strategy,
            cluster_environment=cluster_environment,
        )
    else:
        strategy = "auto"

    csv_logger = CSVLogger(
        args.out_dir.parent,
        args.out_dir.name,
        flush_logs_every_n_steps=args.gradient_accumulation_steps * 10,
    )

    ############# Construct W&B Logger ##############
    if args.offline or args.fast_dev_run:
        os.environ["WANDB_MODE"] = "dryrun"

    wandb_logger = PytorchLightningWandbLogger(
        name=args.run_name,
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        tags=args.wandb_tags,
    )

    fabric = L.Fabric(
        devices=args.num_devices,
        strategy=strategy,
        precision=precision,
        loggers=[wandb_logger, csv_logger],
    )
    with graceful_exceptions(extra_message=f"Rank: {fabric.global_rank}"):
        fabric.launch(main, args)


def main(fabric: L.Fabric, args: Args):
    if args.debug:
        if fabric.local_rank == 0:
            wait_for_debugger()
        fabric.barrier()
    if fabric.global_rank == 0:
        fabric.logger.log_hyperparams(dataclasses.asdict(args))
        fabric.logger.experiment.log_code(".")
        if not args.offline:
            if args.run_name is None:
                printer.warning("No run name specified with `--run_name`. Using W&B default (randomly generated name).")
            else:
                assert fabric.logger.version is not None
                # Append id to name for easier recognition in W&B UI
                fabric.logger.experiment.name = args.run_name + "-" + fabric.logger.version
    printer.config(mode="dev", verbosity="debug", rank=fabric.global_rank, print_rank0_only=True)
    printer.debug(args)
    pretty_print_important_args(fabric, args)

    if fabric.global_rank == 0:
        args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    param_precision = torch.bfloat16 if args.precision == "bf16-true" else torch.float32
    init_device = fabric.device if (fabric.is_global_zero or not args.use_fsdp) else torch.device("meta")
    saved_checkpoint_revision = None
    if args.saved_checkpoint_path and "@" in args.saved_checkpoint_path:
        args.saved_checkpoint_path, saved_checkpoint_revision = args.saved_checkpoint_path.split("@")
    if args.saved_checkpoint_path and not Path(args.saved_checkpoint_path).exists():
        # download from HF Hub
        from huggingface_hub import snapshot_download

        local_ckpt_path = args.out_dir.parent / "hfhub" / args.saved_checkpoint_path
        if saved_checkpoint_revision:
            local_ckpt_path = local_ckpt_path.with_name(
                f"{local_ckpt_path.stem}@{saved_checkpoint_revision}{local_ckpt_path.suffix}"
            )

        if fabric.is_global_zero and not local_ckpt_path.exists():
            printer.info(f"Downloading checkpoint from HF Hub: {args.saved_checkpoint_path} to {local_ckpt_path}")
            snapshot_download(
                args.saved_checkpoint_path,
                revision=saved_checkpoint_revision,
                local_dir=local_ckpt_path,
                local_dir_use_symlinks=False,
                max_workers=args.preprocessing_workers,
            )
        args.saved_checkpoint_path = str(local_ckpt_path)
        fabric.barrier()

    fabric.seed_everything(args.seed)  # same seed for every process to init model (FSDP)
    load_from_path = args.saved_checkpoint_path or args.model_path
    load_from_revision = None
    if "@" in args.model_path.name:
        base_name, load_from_revision = args.model_path.name.split("@")
        load_from_path = args.model_path.with_name(base_name)
        args.model_path = load_from_path
        print(f"Using revision {load_from_revision} for model {args.model_path} loading.")
    with init_device:
        # Tri Dao FA2 breaks compile
        attn_impl = "sdpa" if args.compile else "flash_attention_2"

        need_attn_impl_monkeypatch = attn_impl == "flash_attention_2" and args.precision == "bf16-mixed"
        if need_attn_impl_monkeypatch:
            # transformers bug: PyPI FA2 asserts not float32 weights, but if we use bf16-mixed later it's fine since it gets casted
            # https://github.com/huggingface/transformers/issues/28052#issuecomment-1870034307
            def _autoset_attn_implementation_monkeypatch(cls, config, *args, **kwargs):  # type: ignore
                config._attn_implementation = attn_impl
                return config

            old_autoset_attn_implementation = PreTrainedModel._autoset_attn_implementation
            PreTrainedModel._autoset_attn_implementation = classmethod(_autoset_attn_implementation_monkeypatch)
        if args.use_additional_flash_attn_kernels and not args.compile:
            from flash_attn.ops.rms_norm import RMSNorm as FlashRMSNorm

            prevMistralRMSNorm = transformers.models.mistral.modeling_mistral.MistralRMSNorm
            transformers.models.mistral.modeling_mistral.MistralRMSNorm = FlashRMSNorm
            printer.success("Using FlashRMSNorm instead of MistralRMSNorm.")

        if attn_impl == "sdpa":
            printer.warning("Using torch-native SDPA instead of Tri Dao FlashAttention2.")
            # Force FlashAttention in torch-native SDPA
            torch.backends.cuda.enable_math_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_flash_sdp(True)
        else:
            printer.info(f"Using Attention {attn_impl} implementation.")
            if args.decontaminated_packing:
                assert attn_impl == "flash_attention_2"
                printer.info("Monkey-patching packing functionality for Tri Dao FlashAttention")
                monkey_patch_packing_mistral()

        model: MistralForCausalLM = MistralForCausalLM.from_pretrained(
            load_from_path,
            revision=load_from_revision,
            attn_implementation=attn_impl,
            torch_dtype=param_precision,
            low_cpu_mem_usage=init_device.type != "meta",
            use_cache=False,
            return_dict=True,
        )

        """Revert monkey-patches only necessary for model construction."""
        if need_attn_impl_monkeypatch:
            PreTrainedModel._autoset_attn_implementation = old_autoset_attn_implementation
        if args.use_additional_flash_attn_kernels and not args.compile:
            assert isinstance(model.model.norm, FlashRMSNorm)
            transformers.models.mistral.modeling_mistral.MistralRMSNorm = prevMistralRMSNorm
        printer.debug(model.config)

    printer.success(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")
    printer.debug(model)

    fabric.barrier()
    ######### Model Modifications ##########
    if not args.saved_checkpoint_path and not load_from_revision:
        """Handle tokenizer switching: resize embs + init. Don't do when resuming from checkpoint mid-training."""
        device = model.model.embed_tokens.weight.device if fabric.is_global_zero else torch.device("meta")
        source_wte = source_lm_head = None
        if args.tokenizer_path != args.model_path:
            new_vocab_size = len(AutoTokenizer.from_pretrained(args.tokenizer_path))
            source_wte, source_lm_head = model.model.embed_tokens.weight.detach().clone(), model.lm_head.weight.detach().clone()
            printer.warning(
                f"Resizing model embeddings (size {model.get_input_embeddings().weight.size(0)}) to match tokenizer vocab size ({new_vocab_size}): {args.model_path} -> {args.tokenizer_path}"
            )
            model.resize_token_embeddings(new_vocab_size)

        if fabric.is_global_zero:
            # for FSDP, this is synced via sync_module_states=True
            do_model_modifications(fabric, args, model, device, source_wte, source_lm_head)

        if not args.use_fsdp:
            rank0_wte = fabric.broadcast(model.get_input_embeddings().weight.data, 0)
            rank0_lm_head = fabric.broadcast(model.get_output_embeddings().weight.data, 0)

            model.get_input_embeddings().weight.data = rank0_wte
            model.get_output_embeddings().weight.data = rank0_lm_head

            if args.train_only_embeddings:
                for n, p in model.named_parameters():
                    p.requires_grad = False
                model.get_input_embeddings().weight.requires_grad = True
                model.get_output_embeddings().weight.requires_grad = True
    fabric.barrier()

    print_trainable_param_info(fabric, model)
    parameter_lookup = {k: (p.shape, p.requires_grad) for k, p in model.named_parameters()}
    fabric.barrier()

    fwd_bwd_flops = 0.0
    if fabric.is_global_zero:
        printer.info("------TFLOP & Mem Stats------")
        fwd_bwd_flops = measure_model_flops(
            fabric,
            args.micro_batch_size,
            args.block_size,
            lambda: AutoModelForCausalLM.from_config(
                config=AutoConfig.from_pretrained(args.model_path),
                torch_dtype=param_precision,
                # attn_implementation=attn_impl, # meta device + Tri Dao FA2 not good
            ),
            parameter_lookup=parameter_lookup,
            num_layers=len(model.get_decoder().layers),
            hidden_size=model.get_decoder().get_input_embeddings().weight.shape[-1],
        )[0]
    fabric.broadcast(fwd_bwd_flops)
    fabric.barrier()
    speed_monitor = SpeedMonitorFabric(
        fabric,
        world_size=fabric.world_size,
        model_flops_fwd_bwd=fwd_bwd_flops,
        window_size=1,  # profile each iter - it's fast
    )
    print_mem_stats(fabric, model, speed_monitor, args)

    if args.compile:
        printer.debug("Running `torch.compile` on  model...", rank0_only=False)
        model = torch.compile(model)
    model = fabric.setup_module(model)
    fabric.print(model)
    printer.info(f"current memory usage with (sharded) model on device {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    optimizer = fabric.setup_optimizers(
        get_optimizer(
            model,
            lr=args.max_lr,
            weight_decay=args.weight_decay,
            betas=(args.beta1, args.beta2),
            foreach=args.adamw_foreach,
            use_paged_adamw=args.use_paged_adamw,
        )
    )
    printer.info(f"Peak memory usage after optimizer setup: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    if args.infinite_lr != -1:
        printer.info(
            f"Using Infinite LR Scheduler with max_lr={args.max_lr}, min_lr={args.min_lr}, warmup={args.warmup_period}, cooldown={args.lr_decay_period}, final_annealing={args.lr_final_annealing_period}"
        )
        scheduler = InfiniteLRScheduler(
            optimizer,
            max_lr=args.max_lr,
            constant_lr=args.infinite_lr,
            min_lr=args.min_lr,
            warmup_steps=args.warmup_period,
            cooldown_steps=args.lr_decay_period,
            annealing_steps=args.lr_final_annealing_period,
            max_steps=args.training_goal,
        )
        extra_checkpoint_steps = [args.training_goal - args.lr_final_annealing_period]
    else:
        scheduler = CosineDecayScheduler(
            optimizer,
            max_lr=args.max_lr,
            min_lr=args.min_lr,
            warmup_steps=args.warmup_period,
            decay_steps=args.lr_decay_period,
        )
        extra_checkpoint_steps = []
    printer.info("Using scheduler:", scheduler)

    state: State = {
        "model": model,
        "optimizer": optimizer,
        "hparams": dataclasses.asdict(args),
        "iter_num": 0,
        "step_count": 0,
        "epoch": 0,
    }

    resume_from_sample_idx = None
    resume_from_epoch = None
    if args.saved_checkpoint_path:
        load_optimizer_checkpoint(
            Path(args.saved_checkpoint_path),
            fabric,
            model,
            optimizer,
            fix_compile=args.compile,
        )

        metadata = torch.load(Path(args.saved_checkpoint_path) / "metadata.pt")
        state["iter_num"] = metadata["iter_num"] + 1
        state["step_count"] = metadata["step_count"]
        state["epoch"] = metadata["epoch"]
        state["hparams"] = metadata["hparams"]
        resume_from_sample_idx = state["step_count"] * args.batch_size
        resume_from_epoch = state.get("epoch") or 0
        speed_monitor.step = state["iter_num"] - 1
        printer.success(
            f"Resuming from step {state['step_count']} (sample idx={resume_from_sample_idx})",
            rank0_only=False,
        )
        printer.debug(state["hparams"])
    with main_process_first(
        fabric.local_rank, active=fabric.world_size > 1, infinite_barrier=True
    ):  # main process first to build caches if necessary
        train_dataloader, val_dataloader = get_dataloaders(
            data_dir=args.data_dir,
            block_size=args.block_size,
            batch_size=args.micro_batch_size,
            workers=args.workers,
            tokenizer_path=args.tokenizer_path,
            use_clipped_val=args.cross_tokenizer_val,
            val_batch_size=args.eval_micro_batch_size,
            resume_from_sample_idx=resume_from_sample_idx,
            resume_from_epoch=resume_from_epoch,
            decontaminated_packing=args.decontaminated_packing,
        )

    train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)

    fabric.barrier()
    fabric.seed_everything(args.seed + fabric.global_rank)
    printer.debug(
        f"Starting training: {fabric.global_rank}, seed: {args.seed +  fabric.global_rank}",
        rank0_only=False,
    )

    try:
        printer.info(f"peak memory usage before training {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

        train_time = time.perf_counter()
        train(
            fabric,
            args,
            state,
            train_dataloader,
            val_dataloader,
            scheduler,
            speed_monitor,
            extra_checkpoint_steps=extra_checkpoint_steps,
        )
        printer.success(f"Training time: {(time.perf_counter()-train_time):.2f}s")
        if not args.perf_benchmark:
            future = dlib_save_checkpoint_hf(
                fabric,
                state,
                args.out_dir,
                tags=["final"],
                state_dict_type=StateDictType.FULL_STATE_DICT,
            )
            if fabric.is_global_zero:
                # future.result() waits until checkpoint is saved, important else we exit before saving
                printer.success(f"Saved final checkpoint to {future.result()}")
        fabric.barrier()

    except KeyboardInterrupt:
        printer.error("Detected KeyboardInterrupt, stopping training...")


def train(
    fabric: L.Fabric,
    args: Args,
    state: State,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    lr_scheduler: LRScheduler,
    speed_monitor: SpeedMonitorFabric,
    extra_checkpoint_steps: list[int] = [],
):
    model = state["model"].train()
    optimizer = state["optimizer"]

    if val_dataloader is not None:
        do_and_log_eval(fabric, args, state, val_dataloader, speed_monitor)
        if args.only_val:
            exit(0)

    train_iter = iter(train_dataloader)

    # print bar only on rank0
    step_bar = range(state["step_count"], args.training_goal)
    if fabric.global_rank == 0:
        step_bar = tqdm(step_bar, desc="Such adaptation much wow...")

    try:
        from flash_attn.losses.cross_entropy import CrossEntropyLoss

        loss_fn = CrossEntropyLoss(ignore_index=-1)
        printer.success("Using Tri Dao flash-attn CrossEntropyLoss")
    except ImportError:
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)
        printer.warning("Could not import Tri Dao flash-attn CrossEntropyLoss, using torch.nn.CrossEntropyLoss")

    global_step_end_t = last_global_step_end_t = speed_monitor_end_t = time.perf_counter()
    for i in step_bar:
        iter_bar = range(args.gradient_accumulation_steps)
        if fabric.global_rank == 0:
            iter_bar = tqdm(iter_bar, desc=f"Step {i+1}...", leave=False)

        # iter until effective batch size is reached with gradient accumulation
        avg_data_fetch_t = 0
        avg_iter_t = 0
        for j in iter_bar:
            iter_start_t = time.perf_counter()
            state["iter_num"] += 1
            micro_batch = next(train_iter, None)
            if micro_batch is None:
                printer.info("Reached end of dataset, starting from beginning.")
                state["epoch"] += 1
                # Re-shuffle dataset w/ reproducible seed
                train_dataloader.dataset.training_order = (
                    train_dataloader.dataset.get_reproducible_shuffled_training_order_for_epoch(state["epoch"])
                )
                train_iter = iter(train_dataloader)
                micro_batch = next(train_iter)
            data_fetch_t = time.perf_counter()

            targets = micro_batch.pop("labels")
            do_optimizer_step = j == (args.gradient_accumulation_steps - 1)
            with fabric.no_backward_sync(model, enabled=args.gradient_accumulation_no_sync and not do_optimizer_step):
                logits = model(**micro_batch)["logits"]
                logits = logits.reshape(-1, logits.size(-1))
                targets = targets.reshape(-1)
                loss = loss_fn(logits, targets)
                fabric.backward(loss / args.gradient_accumulation_steps)
            iter_end_t = time.perf_counter()

            # Log performance stats
            speed_monitor.on_train_batch_end(
                args.micro_batch_size,
                iter_end_t - iter_start_t,
                iter_end_t - speed_monitor_end_t,
                # this assumes that device FLOPs are the same and that all devices have the same batch size
                tokens=micro_batch["input_ids"].numel(),
                compute=True,
                step_kwargs={
                    "trainer/optimizer_step": state["step_count"],
                    "trainer/iter": state["iter_num"],
                },
            )
            speed_monitor_end_t = time.perf_counter()
            avg_data_fetch_t += data_fetch_t - iter_start_t
            avg_iter_t += iter_end_t - iter_start_t
        ###########################
        ####### OPTIM STEP ########
        ###########################
        avg_data_fetch_t /= args.gradient_accumulation_steps
        avg_iter_t /= args.gradient_accumulation_steps
        opt_step_t0 = time.perf_counter()
        if args.grad_clip != -1:
            pre_clip_grad_norm = fabric.clip_gradients(
                model, optimizer, max_norm=args.grad_clip, error_if_nonfinite=True
            ).item()

        # Gradient & param tracking
        stat_tracking_elapsed_t = 0
        if args.model_profiling and state["step_count"] % args.model_profiling_interval == 0:
            sharded = (
                isinstance(fabric.strategy, _Sharded) and args.num_devices > 1 and args.fsdp_sharding_strategy != "NO_SHARD"
            )
            stat_tracking_elapsed_t = log_model_stats_to_wandb(
                model,
                log_weights=True,
                log_grads=True,
                sharded_weights=sharded,
                sharded_grads=sharded,
            )

        state["step_count"] += 1
        # determine and set the learning rate for this optimizer step
        lr = lr_scheduler.step(override_step=state["step_count"])
        optimizer.step()
        optimizer.zero_grad()
        last_global_step_end_t = global_step_end_t
        global_step_end_t = time.perf_counter()

        # Also log first opt step, do -1. Do after optimizer.step & zero_grad to log timings
        if (state["step_count"] - 1) % args.log_interval == 0:
            metrics = {
                "trainer/optimizer_step": state["step_count"],
                "trainer/iter": state["iter_num"],
                "trainer/tokens": state["step_count"] * args.batch_size * args.block_size,
                "trainer/samples": state["step_count"] * args.batch_size,
                "train/loss": loss.item(),
                "train/grad_norm": pre_clip_grad_norm,
                "trainer/lr": lr,
            }
            # all_devices_max_cuda_ram = fabric.all_reduce(
            #     torch.cuda.max_memory_allocated() / 1e9, reduce_op=torch.distributed.ReduceOp.MAX
            # )
            timings = {
                "iter_time": avg_iter_t,
                "data_fetch_time": avg_data_fetch_t,
                "global_step_time": global_step_end_t - last_global_step_end_t,
                "opt_step_time": (global_step_end_t - opt_step_t0) - stat_tracking_elapsed_t,
                "grad_tracking_time": stat_tracking_elapsed_t,
                "speed_monitor_time": speed_monitor_end_t - iter_end_t,
                "max_cuda_ram": f"{torch.cuda.max_memory_allocated() / 1e9:.2f} GB",
                # "all_devices_max_cuda_ram": f"{all_devices_max_cuda_ram:.2f} GB",
            }
            torch.cuda.reset_peak_memory_stats()
            printer.info(pretty_str_from_dict(metrics | timings, prefix="Step stats:"))
            fabric.log_dict(metrics)

        if val_dataloader is not None and (
            state["step_count"] % args.eval_interval == 0 or state["step_count"] in extra_checkpoint_steps
        ):
            do_and_log_eval(fabric, args, state, val_dataloader, speed_monitor)
            fabric.barrier()

        if state["step_count"] % args.save_interval == 0 or state["step_count"] in extra_checkpoint_steps:
            tags = ["extra"] if state["step_count"] in extra_checkpoint_steps else []
            dlib_save_checkpoint_hf(
                fabric,
                state,
                args.out_dir,
                state_dict_type=StateDictType.FULL_STATE_DICT,
                tags=tags,
            )


def do_model_modifications(
    fabric: L.Fabric,
    args: Args,
    model: MistralForCausalLM,
    device: torch.device,
    source_wte: torch.Tensor = None,
    source_lm_head: torch.Tensor = None,
):
    if args.deepfocus_init:
        deepfocus_init_(fabric, args, source_wte, source_lm_head, model)

    if args.wechsel_init:
        wechsel_init_(fabric, args, source_wte, source_lm_head, model)

    if args.mean_init:
        with fabric.strategy.precision.tensor_init_context(), device:
            printer.debug(
                model.model.embed_tokens.weight.device,
                model.lm_head.weight.device,
                source_wte.device,
                source_lm_head.device,
                fabric.device,
                device,
                rank0_only=False,
            )
            gen = torch.Generator(device=device).manual_seed(42)
            wte_mean, wte_std = source_wte.mean(dim=0), source_wte.std(dim=0)
            lm_head_mean, lm_head_std = source_lm_head.mean(dim=0), source_lm_head.std(dim=0)

            model.lm_head.weight.data = torch.stack(
                [torch.normal(lm_head_mean, lm_head_std, generator=gen) for _ in range(model.lm_head.weight.shape[0])]
            )
            printer.debug(model.lm_head.weight.data.shape)

            model.model.embed_tokens.weight.data = torch.stack(
                [torch.normal(wte_mean, wte_std, generator=gen) for _ in range(model.model.embed_tokens.weight.shape[0])]
            )
            printer.debug(model.model.embed_tokens.weight.data.shape)
    if args.random_init or args.zipf_init:
        with fabric.strategy.precision.tensor_init_context(), device:
            gen = torch.Generator(device=device).manual_seed(42)
            model.lm_head.weight.data = torch.randn_like(model.lm_head.weight) * 0.02
            model.model.embed_tokens.weight.data = torch.randn_like(model.model.embed_tokens.weight) * 0.02

    if args.zipf_init:
        # copy embeddings from original matrix but fix to new vocab length
        with fabric.strategy.precision.tensor_init_context(), device:
            if model.lm_head.weight.shape[0] <= source_lm_head.shape[0]:
                model.lm_head.weight.data = source_lm_head[: model.lm_head.weight.shape[0]]
                model.model.embed_tokens.weight.data = source_wte[: model.model.embed_tokens.weight.shape[0]]
            else:
                model.lm_head.weight.data[: source_lm_head.shape[0]] = source_lm_head
                model.model.embed_tokens.weight.data[: source_wte.shape[0]] = source_wte

    if args.smart_heuristic_init:
        from helpers.smart_heuristics import _xlmr_special_tokens, reinitialize_by_identity, reinitialize_by_script

        pretrained_tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        target_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
        new_vocab = target_tokenizer.get_vocab()
        old_vocab = pretrained_tokenizer.get_vocab()

        # do wte
        new_embeddings = model.model.embed_tokens.weight.clone().detach()
        old_embeddings = source_wte
        new_embeddings = reinitialize_by_script(old_vocab, old_embeddings, new_vocab, new_embeddings)
        new_embeddings = reinitialize_by_identity(
            old_vocab, old_embeddings, new_vocab, new_embeddings, tokens_to_ignore=_xlmr_special_tokens
        )[0]
        model.model.embed_tokens.weight.data = new_embeddings

        # do lm_head
        new_embeddings = model.lm_head.weight.clone().detach()
        old_embeddings = source_lm_head
        new_embeddings = reinitialize_by_script(old_vocab, old_embeddings, new_vocab, new_embeddings)
        new_embeddings = reinitialize_by_identity(
            old_vocab, old_embeddings, new_vocab, new_embeddings, tokens_to_ignore=_xlmr_special_tokens
        )[0]
        model.lm_head.weight.data = new_embeddings

    if args.train_only_embeddings:
        for n, p in model.named_parameters():
            if "embed_tokens" in n or "lm_head" in n:
                p.requires_grad = True
                printer.info(f"Training {n}")
            else:
                p.requires_grad = False
    if args.train_embeddings:
        model.model.embed_tokens.weight.requires_grad = True
        model.lm_head.weight.requires_grad = True


def do_and_log_eval(
    fabric: L.Fabric,
    args: Args,
    state: dict,
    val_dataloader: DataLoader,
    speed_monitor: SpeedMonitorFabric,
):
    state["model"].eval()
    t0 = time.perf_counter()
    val_metrics = validate(fabric, args, state["model"], val_dataloader)
    t1 = time.perf_counter() - t0
    speed_monitor.eval_end(t1)
    metrics = {
        "trainer/optimizer_step": state["step_count"],
        "trainer/iter": state["iter_num"],
        "val/loss": val_metrics["loss"].item(),
        "val/per_token_nll": val_metrics["per_token_nll"].item(),
        "val/per_doc_nll": val_metrics["per_doc_nll"].item(),
        "val/ppl": val_metrics["perplexity"].item(),
    }
    printer.info(pretty_str_from_dict(metrics | {"val/time": t1}, prefix="Eval Stats:"))
    fabric.log_dict(metrics)
    state["model"].train()


@torch.no_grad()  # @torch.inference_mode() leads to error with FSDP (not w/ DDP though...)
def validate(
    fabric: L.Fabric,
    args: Args,
    model: MistralForCausalLM,
    val_dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, float]:
    model.eval()
    val_iter = iter(val_dataloader)
    eval_iter_batch_size = args.eval_micro_batch_size * args.num_devices
    if args.cross_tokenizer_val and args.eval_micro_batch_size == 1 and args.compile:
        printer.info("Cross tokenizer val with torch.compile, manually padding to block size.")
        eval_iter_batch_size = args.micro_batch_size * args.num_devices

    max_iters_in_dataloader = len(val_dataloader)
    iters = args.eval_samples // eval_iter_batch_size if args.eval_samples != -1 else max_iters_in_dataloader
    iters = min(iters, max_iters_in_dataloader)

    num_non_pad_tokens = torch.tensor(0, device=fabric.device)
    losses = torch.zeros(iters, device=fabric.device)
    logprob_accumulator = torch.zeros(iters, device=fabric.device)

    for i in tqdm(range(iters), desc="Validating...", leave=False):
        if args.cross_tokenizer_val and args.eval_micro_batch_size == 1 and args.compile:
            stacked_ids = []
            stacked_targets = []
            for _ in range(args.micro_batch_size):
                micro_batch = next(val_iter)
                input_ids = micro_batch["input_ids"]
                targets = micro_batch["labels"]
                # Pad to block size for torch.compile
                assert input_ids.shape[1] <= args.block_size
                necessary_pad_tokens = args.block_size - input_ids.shape[1]
                assert necessary_pad_tokens >= 0
                input_ids = torch.nn.functional.pad(input_ids, (0, necessary_pad_tokens), value=3)
                targets = torch.nn.functional.pad(targets, (0, necessary_pad_tokens), value=-1)

                stacked_ids.append(input_ids)
                stacked_targets.append(targets)
            input_ids = torch.cat(stacked_ids, dim=0)
            targets = torch.cat(stacked_targets, dim=0)
        else:
            micro_batch = next(val_iter)
            input_ids = micro_batch["input_ids"]
            targets = micro_batch["labels"]
        logits = model(input_ids)["logits"]
        logits = logits.reshape(-1, logits.size(-1))
        targets = targets.reshape(-1)
        summed_loss = torch.nn.functional.cross_entropy(logits, targets, ignore_index=-1, reduction="sum")

        # Count num of non pad *labels* (since they count for loss). Assumes ignore idx == -1.
        non_pad_targets_in_batch = (targets != -1).sum()
        num_non_pad_tokens += non_pad_targets_in_batch
        losses[i] = summed_loss / non_pad_targets_in_batch  # equivalent to nn.cross_entropy w/ reduction="mean"
        logprob_accumulator[i] = summed_loss

    avg_loss = losses.mean()
    summed_corpus_nll = logprob_accumulator.sum()

    # Reduce across all processes
    avg_loss = fabric.all_reduce(avg_loss, reduce_op="mean")
    summed_corpus_nll = fabric.all_reduce(summed_corpus_nll, reduce_op="sum")
    num_non_pad_tokens = fabric.all_reduce(num_non_pad_tokens, reduce_op="sum")

    per_token_perplexity = torch.exp(avg_loss)
    per_token_nll = summed_corpus_nll / num_non_pad_tokens
    num_documents = iters * eval_iter_batch_size
    per_doc_nll = summed_corpus_nll / num_documents

    model.train()
    return {
        "loss": avg_loss,
        "perplexity": per_token_perplexity,
        "per_token_nll": per_token_nll,
        "per_doc_nll": per_doc_nll,
    }


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    setup(simple_parsing.parse(Args, add_config_path_arg=True, argument_generation_mode=""))
