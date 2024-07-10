from typing import TYPE_CHECKING

import torch
from lightning import Fabric
from print_on_steroids import logger as printer
from transformers import AutoTokenizer, LlamaForCausalLM

if TYPE_CHECKING:
    from args import TrainingArgs as Args


def deepfocus_init_(
    fabric: Fabric,
    args: "Args",
    source_wte: torch.Tensor,
    source_lm_head: torch.Tensor,
    model: LlamaForCausalLM,
    keys: list[str] = ["wte", "lm_head"],
    device: torch.device = torch.device("cpu"),
):
    from deepfocus import FOCUS

    device = device or fabric.device
    printer.info("Using deepfocus to initialize embeddings")

    pretrained_tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    target_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    init_jobs = []
    if "wte" in keys:
        init_jobs.append(("wte", source_wte))
    if "lm_head" in keys:
        init_jobs.append(("lm_head", source_lm_head))
    with device:
        for key, source_embs in init_jobs:
            printer.info(f"FOCUS init for {key}")
            printer.info("src embs device", source_embs.device)
            target_embs = FOCUS(
                source_embeddings=source_embs.to(torch.float32),
                source_tokenizer=pretrained_tokenizer,
                target_tokenizer=target_tokenizer,
                language_identifier="de",
                auxiliary_embedding_mode=args.focus_auxiliary_mode,
                exact_match_all=args.focus_exact_match_all,
                match_symbols=args.focus_match_symbols,
                target_training_data_path="/scratch3/konstantin.dobler/oscar2023/de/train_reduced.txt",
                fasttext_model_dim=args.focus_fasttext_dim,
                fasttext_model_epochs=args.focus_fasttext_epochs,
                fasttext_model_min_count=args.focus_fasttext_min_count,
                fasttext_model_path=args.focus_fasttext_model_path,
                seed=args.seed,
                device=device,
            )
            if key == "wte":
                model.model.embed_tokens.weight.data = target_embs.to(
                    device=device, dtype=fabric.strategy.precision._desired_input_dtype
                )
            elif key == "lm_head":
                model.lm_head.weight.data = target_embs.to(device=device, dtype=fabric.strategy.precision._desired_input_dtype)
            print(target_embs.shape)


def wechsel_init_(fabric: Fabric, args: "Args", source_wte: torch.Tensor, source_lm_head: torch.Tensor, model):
    print("Using wechsel to initialize embeddings")
    import scipy
    from numpy import triu as np_triu

    scipy.linalg.triu = np_triu  # fix dependency bug for scipy 1.13.0
    import wechsel

    language = "de"
    if "/ar/" in str(args.data_dir):
        language = "ar"
    static_src_embs = wechsel.load_embeddings("en")
    static_tgt_embs = wechsel.load_embeddings(language)

    dict_full_name_lookup = {
        "sw": "swahili",
        "de": "german",
        "hi": "hindi",
        "vi": "vietnamese",
        "ar": "arabic",
        "lb": "luxembourgish",
        "ceb": "cebuano",
        "gd": "scottish gaelic",
        "ug": "uyghur",
        # from here on, the languages do not have pretrained fasttext word embeddings on https://fasttext.cc/docs/en/crawl-vectors.html
        "xh": "xhosa",
        "ha": "hausa",
        "sm": "samoan",
        "hmn": "hmong",
    }
    dictionary = dict_full_name_lookup[language]
    align_strat = "bilingual_dictionary"

    wechseler = wechsel.WECHSEL(
        static_src_embs,
        static_tgt_embs,
        bilingual_dictionary=dictionary,
        align_strategy=align_strat,
    )

    pretrained_tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    target_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    with fabric.strategy.precision.tensor_init_context(), fabric.device:
        for key, source_embs in [("wte", source_wte), ("lm_head", source_lm_head)]:
            fabric.print(f"Wechseling {key}")
            target_embeddings, info = wechseler.apply(
                pretrained_tokenizer,
                target_tokenizer,
                source_embs.cpu().to(torch.float32).numpy(),
                use_subword_info=align_strat is not None,
            )
            if key == "wte":
                model.model.embed_tokens.weight.data = torch.from_numpy(target_embeddings).to(
                    device=fabric.device, dtype=fabric.strategy.precision._desired_input_dtype
                )
            elif key == "lm_head":
                model.lm_head.weight.data = torch.from_numpy(target_embeddings).to(
                    device=fabric.device, dtype=fabric.strategy.precision._desired_input_dtype
                )
            print(target_embeddings.shape)
