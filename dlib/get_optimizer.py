import torch


def get_optimizer(
    model: torch.nn.Module,
    lr: float,
    weight_decay: float,
    betas: tuple[float, float],
    foreach=True,
    use_paged_adamw=False,
) -> torch.optim.Optimizer:
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    trainable_named_parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    assert len(trainable_parameters) == len(trainable_named_parameters)

    ### Do not include RMSNorm and embs for weight decay https://forums.fast.ai/t/is-weight-decay-applied-to-the-bias-term/73212/6
    no_decay = ["lm_head", "wte", "embed_tokens", "ln_f", "norm"]
    decay_params = [p for n, p in trainable_named_parameters if not any(nd in n for nd in no_decay)]
    no_decay_params = [p for n, p in trainable_named_parameters if any(nd in n for nd in no_decay)]
    trainable_params = []
    if len(decay_params) > 0:
        trainable_params.append({"params": decay_params, "weight_decay": weight_decay})
    if len(no_decay_params) > 0:
        trainable_params.append({"params": no_decay_params, "weight_decay": 0.0})
    # printer.info(f"no weight decay for: {[n for n, p in trainable_named_parameters if any(nd in n for nd in no_decay)]}")
    # printer.info(f"weight decay for: {[n for n, p in trainable_named_parameters if not any(nd in n for nd in no_decay)]}")

    if use_paged_adamw:
        from bitsandbytes.optim import PagedAdamW32bit

        optimizer = PagedAdamW32bit(
            trainable_parameters,
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            # foreach=foreach,
        )
    else:
        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            foreach=foreach,
        )
    return optimizer
