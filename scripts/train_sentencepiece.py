from dataclasses import dataclass

import jsonlines
import sentencepiece as spm
from simple_parsing import field, parse


@dataclass
class Args:
    file: str = field(default="./data/train.jsonl")
    """Expected format JSONL like: {"text": "Lorem ipsum...."}"""
    limit_samples: int = field(default=-1)
    """Limit the number of samples to train on. -1 means no limit."""

    model_name: str = field(default="sentencepiece_bpe")
    model_type: str = field(default="bpe")
    vocab_size: int = field(default=128 * 128 * 2)
    "Setting vocab size to multiple of 128 and 64 and 8 can give nice speedups when training on Tensor Cores."

    threads: int = field(default=16)
    character_coverage: float = field(default=0.9995)
    max_sentence_length: int = field(default=8192)
    train_extremely_large_corpus: bool = True
    allow_whitespace_only_pieces: bool = True
    split_digits: bool = True
    byte_fallback: bool = True


def main():
    args: Args = parse(Args)

    with jsonlines.open(args.file, mode="r") as reader:
        sentence_iter = (line["text"] for line in reader)

        extra_kwargs = {}
        if args.limit_samples != -1:
            extra_kwargs["input_sentence_size"] = args.limit_samples
            extra_kwargs["shuffle_input_sentence"] = True
        spm.SentencePieceTrainer.train(
            sentence_iterator=sentence_iter,
            model_prefix=args.model_name,
            model_type=args.model_type,
            vocab_size=args.vocab_size,
            num_threads=args.threads,
            character_coverage=args.character_coverage,
            max_sentence_length=args.max_sentence_length,
            allow_whitespace_only_pieces=args.allow_whitespace_only_pieces,
            split_digits=args.split_digits,
            byte_fallback=args.byte_fallback,
            train_extremely_large_corpus=args.train_extremely_large_corpus,
            pad_id=3,
            **extra_kwargs,
        )


if __name__ == "__main__":
    main()
