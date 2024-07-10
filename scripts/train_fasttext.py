import multiprocessing
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import fasttext
from datasets.fingerprint import Hasher
from datasets.load import load_dataset
from print_on_steroids import logger
from simple_parsing import field, parse
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer

CACHE_DIR = (Path(os.getenv("XDG_CACHE_HOME", "~/.cache")) / "deepfocus").expanduser().resolve()
print(CACHE_DIR)


@dataclass
class Args:
    text_path: str
    target_tokenizer: str
    epochs: int
    dim: int
    min_count: int
    out: str = field(default="")
    processes: int = field(default=None)
    cache_tokenized_text: bool = field(default=True)
    limit_samples: int = field(default=None)


def train_fasttext(
    text_path: str,
    target_tokenizer: PreTrainedTokenizer,
    epochs,
    dim,
    min_count,
    processes=None,
    cache_tokenized_text=True,
):
    """Utility function to train a fasttext model on tokenized text data.
    Args:
        text_path (str): Path to a file containing text. This file will be used to train the fasttext model.
        target_tokenizer (PreTrainedTokenizer, optional): The tokenizer to to apply to the provided text file before training the fasttext model on the data.
        epochs (int, optional): The number of training epochs for the fasttext model.
        dim (int, optional): The embedding dimension for the fasttext model.
        processes (int, optional): The number of processes to use for parrallelization. Defaults to `multiprocessing.cpu_count()`.
        cache_tokenized_text(bool, optional): Whether to cache the tokenized text data. Defaults to False.

    Returns:
        A trained fasttext model.

    ------------
    Adapted from https://github.com/CPJKU/wechsel/blob/56ae305e5d7d20383cf891371ffeb7885763cdc5/wechsel/__init__.py#L128-L181.
    """
    processes = processes or multiprocessing.cpu_count()
    target_tokenizer_hash = Hasher().hash(target_tokenizer)
    # data_file_name = Path(text_path).stem
    data_file_suffix = Path(text_path).suffix

    text_path_sanitized = text_path.rstrip("/\\").replace("/", "_").replace("\\", "_")

    # print(target_tokenizer_hash, data_file_name, data_file_suffix)
    cache_file = CACHE_DIR / "data" / f"{text_path_sanitized}_tokenized_{target_tokenizer_hash}{data_file_suffix}"

    if cache_file.exists():
        logger.success(f"Tokenized text for {text_path} found at {cache_file}...")
    else:
        if cache_tokenized_text:
            logger.info(f"Tokenizing text in {text_path} and caching results in {cache_file}...")

        if text_path.endswith(".txt"):
            dataset = load_dataset("text", data_files=text_path, split="train")
        if text_path.endswith(".json") or text_path.endswith(".jsonl"):
            dataset = load_dataset("json", data_files=text_path, split="train")
        dataset = dataset.map(
            lambda sample: {"text": " ".join([token for token in target_tokenizer.tokenize(sample["text"])])},
            num_proc=processes,
        )
        if cache_tokenized_text:
            os.makedirs(str(cache_file.parent), exist_ok=True)

            with cache_file.open("w+", encoding="utf-8") as f:
                f.writelines((text + "\n" for text in tqdm(dataset["text"], desc="Writing data...")))
            logger.success(f"Tokenized target language training data for fasttext written to {cache_file}...")
        else:
            temp_file = tempfile.NamedTemporaryFile("w+", encoding="utf-8")
            for text in dataset["text"]:
                temp_file.write(text + "\n")
            cache_file = temp_file.name

    logger.info(f"Training fasttext model on {f'tokenized {text_path}' if cache_tokenized_text else cache_file}...")
    # We use CBOW instead of skipgram becasue CBOW is more closely aligned with Masked Language Modeling
    # minCount to filter out spurious tokens that will not get a good fasttext embedding
    return fasttext.train_unsupervised(
        str(cache_file),
        dim=dim,
        neg=10,
        model="cbow",
        epoch=epochs,
        thread=processes,
        minCount=min_count,
    )


def main():
    args = parse(Args)
    tokenizer = AutoTokenizer.from_pretrained(args.target_tokenizer)

    if args.out == "":
        target_tokenizer_hash = Hasher().hash(tokenizer)

        text_path_sanitized = Path(args.text_path).as_posix().replace("/", "_")

        model_path = Path(
            CACHE_DIR
            / "fasttext"
            / f"data_{text_path_sanitized}_tokenizer_{target_tokenizer_hash}_epochs_{args.epochs}_dim_{args.dim}_mincount_{args.min_count}.bin"
        )
    else:
        model_path = Path(args.out)

    fasttext_model = train_fasttext(
        args.text_path,
        tokenizer,
        args.epochs,
        args.dim,
        args.min_count,
        args.processes,
        args.cache_tokenized_text,
    )

    logger.success(f"Saving fasttext model to {model_path}.")
    os.makedirs(str(model_path.parent), exist_ok=True)
    fasttext_model.save_model(str(model_path))


if __name__ == "__main__":
    main()
