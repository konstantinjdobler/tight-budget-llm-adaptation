# Language Adaptation on a Tight Academic Compute Budget

This is the official repository for the paper "Language Adaptation on a Tight Academic Compute Budget: Tokenizer Swapping Works and Pure bfloat16 Is Enough".

We empirically analyze language adaptation of Mistral-7B-v0.1 to German and Arabic on "tight academic compute budgets".
The main findings include:

- We can do continued pretraining in pure bfloat16 rather than mixed precision. This is significant for settings with only a few GPUs, as mixed-precision training can be slow or impossible due to OOM. Pure bfloat16 is up to ~39% faster!
- Tokenizer swapping with a custom language-specific tokenizer works even on a tight compute budget (although it does not improve task performance in our experiments).
- Adapting to German did not actually improve German task performance but adapting to Arabic did. Languages that are already well-represented might not benefit from language adaptation!

For more details, view our paper on OpenReview: https://openreview.net/forum?id=VYfJaHeVod.

## Models

We provide the model checkpoints for the adapted Mistral-7B-v0.1 versions from the paper on Huggingface.

From our main experiments:

| Language | Tokenizer | Training Precision       | Huggingface link                                                                                                                            |
| -------- | --------- | ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------- |
| German   | German    | pure bfloat16            | [`konstantindobler/mistral7b-de-tokenizer-swap-pure-bf16`](https://huggingface.co/konstantindobler/mistral7b-de-tokenizer-swap-pure-bf16)   |
| German   | German    | mixed-precision bfloat16 | [`konstantindobler/mistral7b-de-tokenizer-swap-mixed-bf16`](https://huggingface.co/konstantindobler/mistral7b-de-tokenizer-swap-mixed-bf16) |
| German   | original  | pure bfloat16            | [`konstantindobler/mistral7b-de-pure-bf16`](https://huggingface.co/konstantindobler/mistral7b-de-pure-bf16)                                 |
| German   | original  | mixed-precision bfloat16 | [`konstantindobler/mistral7b-de-mixed-bf16`](https://huggingface.co/konstantindobler/mistral7b-de-mixed-bf16)                               |

From our hindsight experiments with improved training recipes:

| Language | Tokenizer | Training Precision                        | Huggingface link                                                                                                                                                                |
| -------- | --------- | ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| German   | German    | pure bfloat16                             | [`konstantindobler/mistral7b-de-tokenizer-swap-pure-bf16-v2`](https://huggingface.co/konstantindobler/mistral7b-de-tokenizer-swap-pure-bf16-v2)                                 |
| German   | German    | pure bfloat16 (anneal in mixed-precision) | [`konstantindobler/mistral7b-de-tokenizer-swap-pure-bf16-v2-anneal-ablation`](https://huggingface.co/konstantindobler/mistral7b-de-tokenizer-swap-pure-bf16-v2-anneal-ablation) |
| Arabic   | Arabic    | pure bfloat16                             | [`konstantindobler/mistral7b-ar-tokenizer-swap-pure-bf16`](https://huggingface.co/konstantindobler/mistral7b-ar-tokenizer-swap-pure-bf16)                                       |
| Arabic   | Arabic    | pure bfloat16 (anneal in mixed-precision) | [`konstantindobler/mistral7b-ar-tokenizer-swap-pure-bf16-anneal-ablation`](https://huggingface.co/konstantindobler/mistral7b-ar-tokenizer-swap-pure-bf16-anneal-ablation)       |

The German models all underperform Mistral-7B-v0.1 on German downstream tasks, however the [`konstantindobler/mistral7b-ar-tokenizer-swap-pure-bf16`](https://huggingface.co/konstantindobler/mistral7b-ar-tokenizer-swap-pure-bf16) checkpoint (as well as the anneal ablation) outperformed the base model and several baselines.

## Experiments

We perform our experiments using Docker images with pre-built dependencies for better reproducibility. We use two different images (one for our initial main experiments and another one with updated dependencies in our hindsight study).

### Environment

<details><summary>Main experiments</summary>

<p>

For our main experiments, we use the Docker image at `konstantinjdobler/tight-budget:main-experiments`. The dependency lockfile can be found at [`./helpers/main-exp-deps/conda-lock.yml`](./helpers/main-exp-deps/conda-lock.yml) with the corresponding Dockerfile at
[`./helpers/main-exp-deps/Dockerfile.condalock`](./helpers/main-exp-deps/Dockerfile.condalock). This uses `conda-lock` for exact pinning of package versions.

</details>
<p>

<details><summary>Hindsight Study</summary>

<p>

For the hindsight study experiments, we use the Docker image at `konstantinjdobler/tight-budget:hindsight`. The dependency lockfile can be found at [`./requirements.lock`](./requirements.lock) with the corresponding Dockerfile at [`./Dockerfile`](./Dockerfile). Instead of `conda-lock`, we used `rye` for exact pinning of package versions due to issues of `conda-lock` when using nightly `torch` versions.

</details>
<p>

### Data

We use data from OSCAR23.01 and CulturaX for our main experiments and hindsight study, respectively. We pre-tokenize the data into an optimized `np.memmap`-ed format. You can use the following commands to download and pre-tokenize our used datasets:

<details><summary>Main experiments</summary>

<p>

For our main experiments, we use OSCAR23.01. To download run:

```sh
bash scripts/run-docker.sh python dlib/data/data_download.py --lg de --dataset oscar2023 --max_train_size 10_000_000 --dev_size 50_000 --out_dir /my/data/dir --stream --disk_space --processes 8 --stream_shuffle_buffer_size 100_000
```

To pre-tokenize, run:

```sh
bash scripts/run-docker.sh python dlib/data/data_tokenization.py --source_dir /my/data/dir/ --tokenizer_path ./tokenizers/de/sp-bpe-de-32kauto --append_eos False --prepend_bos False --extra_val_clip_length 512 --conserve_disk_space
```

and also `--tokenizer_path mistralai/Mistral-7B-v0.1` for training with the original vocabulary. Note that `<eos>` tokens are added during the dataloading, not during pre-tokenization.

</details>
<p>

<details><summary>Hindsight Study</summary>

<p>

For our hindsight study experiments, we use data from CulturaX. To download run:

```sh
bash scripts/run-docker.sh python dlib/data/data_download.py --lg de --dataset culturax --max_train_size 10_000_000 --dev_size 50_000 --out_dir /my/data/dir --stream --disk_space --processes 16 --stream_shuffle_buffer_size 100_000
```

with `--lg de` and `--lg ar` for German and Arabic, respectively.

To pre-tokenize, run:

```sh
bash scripts/run-docker.sh python dlib/data/data_tokenization.py --source_dir /my/data/dir/ --tokenizer_path ./tokenizers/de/sp-bpe-de-32kauto --append_eos False --prepend_bos False --extra_val_clip_length 512 --conserve_disk_space
```

with `--tokenizer_path ./tokenizers/{de,ar}/sp-bpe-{de,ar}-32kauto` matching the language and also `--tokenizer_path mistralai/Mistral-7B-v0.1` for training with the original vocabulary. Note that `<bos>` tokens are added during the dataloading, not during pre-tokenization.

</details>
<p>

### Training

Our training script is [`./train.py`](./train.py). Since our main and hindsight experiments use different package versions and arguments, we need to use their respective Docker images and configs. In general, we adjusted _efficiency parameters_, such as micro-batch size, FSDP sharding level, and activation checkpointing depending on the GPU memory and number of devices we had available for each run.

In general, a training command looks like this:

```bash
bash scripts/run-docker.sh -i konstantinjdobler/tight-budget:main -g 0,2 python train.py \
 --config_path ./cfgs/main.yml --out_dir /path/to/ckptdir \
 -d /path/to/data --tokenizer {tokenizer} --precision {bf16-mixed,bf16-true} --mb {1,2,4,8} \
 --fsdp_sharding_strategy {FULL_SHARD,SHARD_GRAD_OP}  --activation_checkpointing {true,false} \
 --gradient_accumulation_no_sync {true,false} --use_paged_adamw {true,false}
```

In our launch script [`./scripts/run-docker.sh`](./scripts/run-docker.sh), we can select the Docker image via `-i konstantinjdobler/tight-budget:main` and the GPUs used for training via `-g 0,2` (this uses GPUs 0 and 2).

You will need to change the HuggingFace `repo_id` (in [`./dlib/checkpointing.py`](./dlib/checkpointing.py)) and W&B account name `WANDB_ENTITY` (in [`./train.py`](./train.py)) to match your accounts and adjust the mounts in the Docker launch script [`./scripts/run-docker.sh`](./scripts/run-docker.sh) to match your local machine.

You can run

```bash
bash run-docker.sh -i konstantinjdobler/tight-budget:main python train.py --help
```

for a list of all arguments. We describe the main ones here:

<details><summary>Training Arguments</summary>

<p>

For reproducing our main experiments, choose `--config_path ./cfgs/main.yml` and `-i konstantinjdobler/tight-budget:main`. For reproducing our hindsight experiments, choose `--config_path ./cfgs/hindsight.yml` and `-i konstantinjdobler/tight-budget:hindsight`.

Choose `--tokenizer` and `-d` based on the dataset as described in the data section.

The efficiency arguments `fsdp_sharding_strategy`, `--mb` (micro-batch size), `--activation_checkpointing`, `--gradient_accumulation_no_sync`, and `--use_paged_adamw` can be set according to number of available GPUs and GPU memory. In the commands to reproduce our trainings, we provide examples for four GPUs. In practice, we had to use many different numbers and types of GPUs according to availability and adjusted the efficiency arguments accordingly (benchmarked settings for 80GB GPUs can be found in Table 2 of the paper).

</details>
<p>

and list training commands to reproduce our experiments below:

<details><summary>Main Experiments</summary>

<p>

We adapt Mistral-7B to German using OSCAR23.01 with all combinations of `{original vocab, tokenizer swapping} x {pure bfloat16, mixed-precision bfloat16}`.

For tokenizer swapping, we use FOCUS embedding initialization. You can download the required fasttext embedding from [this source](https://huggingface.co/konstantindobler/fasttext-de-sentencepiece-bpe-32k).

Adapting Mistral-7B with tokenizer swapping and pure bfloat16:

```sh
bash scripts/run-docker.sh -i konstantinjdobler/tight-budget:main -g 0,1,2,3 python train.py --config_path ./cfgs/main.yml ./cfgs/deepfocus.yml --tokenizer ./tokenizers/de/sp-bpe-de-32kauto/ --focus_fasttext_model_path /path/to/fasttextmodel-de -d /path/to/OSCAR23.01-custom-de-tokenized --out_dir /path/to/ckptdir --precision bf16-true --fsdp_sharding_strategy SHARD_GRAD_OP --mb 1
```

Adapting Mistral-7B while keeping the original vocabulary and pure bfloat16:

```sh
bash scripts/run-docker.sh -i konstantinjdobler/tight-budget:main -g 0,1,2,3 python train.py --config_path ./cfgs/main.yml -d /path/to/OSCAR23.01-mistral-7b-tokenized --out_dir /path/to/ckptdir --precision bf16-true --fsdp_sharding_strategy SHARD_GRAD_OP --mb 1
```

Adapting Mistral-7B with tokenizer swapping and mixed-precision bfloat16:

```sh
bash scripts/run-docker.sh -i konstantinjdobler/tight-budget:main -g 0,1,2,3 python train.py --config_path ./cfgs/main.yml ./cfgs/deepfocus.yml --tokenizer ./tokenizers/de/sp-bpe-de-32kauto/ --focus_fasttext_model_path /path/to/fasttextmodel-de -d /path/to/OSCAR23.01-custom-de-tokenized --out_dir /path/to/ckptdir --precision bf16-mixed --fsdp_sharding_strategy FULL_SHARD --mb 8 --activation_checkpointing true --gradient_accumulation_no_sync false
```

Adapting Mistral-7B while keeping the original vocabulary and mixed-precision bfloat16:

```sh
bash scripts/run-docker.sh -i konstantinjdobler/tight-budget:main -g 0,1,2,3 python train.py --config_path ./cfgs/main.yml -d /path/to/OSCAR23.01-mistral-7b-tokenized --out_dir /path/to/ckptdir --precision bf16-mixed --fsdp_sharding_strategy FULL_SHARD --mb 8 --activation_checkpointing true --gradient_accumulation_no_sync false
```

</details>
<p>

<details><summary>Hindsight Experiments</summary>

<p>

In the hindsight study, we adapt Mistral-7B to German or Arabic using CulturaX with tokenizer swapping and pure bfloat16. Compared to the main experiments, we introduced several improvements, such as example packing without cross-document attention and use an "infinite" learning rate schedule.

For tokenizer swapping, we use FOCUS embedding intialization. You can download the required fasttext embedding from [this source](https://huggingface.co/konstantindobler/fasttext-de-sentencepiece-bpe-32k) for German (same as in the main experiments) and from [this source](https://huggingface.co/konstantindobler/fasttext-ar-sentencepiece-bpe-32k) for Arabic.

Adapting Mistral-7B to German with tokenizer swapping, pure bfloat16, and the improved training recipe:

```sh
bash scripts/run-docker.sh -i konstantinjdobler/tight-budget:hindsight -g 0,1,2,3 python train.py --config_path ./cfgs/hindsight.yml ./cfgs/deepfocus.yml --tokenizer ./tokenizers/de/sp-bpe-de-32kauto/ --focus_fasttext_model_path /path/to/fasttextmodel-de -d /path/to/CulturaX-custom-de-tokenized --out_dir /path/to/ckptdir --precision bf16-true --fsdp_sharding_strategy SHARD_GRAD_OP --mb 1
```

For Arabic, we first tune just the embeddings of the new tokenizer for a short period:

```sh
bash scripts/run-docker.sh -i konstantinjdobler/tight-budget:hindsight -g 0,1,2,3 python train.py --config_path ./cfgs/hindsight.yml ./cfgs/deepfocus.yml --tokenizer ./tokenizers/ar/sp-bpe-ar-32kauto/ --focus_fasttext_model_path /path/to/fasttextmodel-ar -d /path/to/CulturaX-custom-ar-tokenized --out_dir /path/to/ckptdir --precision bf16-true --mb 2 --train_only_embeddings --use_fsdp false --training_goal 100 --save_interval 120 --max_lr 4e-4 --infinite_lr 2e-4 --min_lr 1e-5 --warmup_period 10
```

and then proceed to start the regular continued pretraining starting from the tuned embeddings checkpoint:

```sh
bash scripts/run-docker.sh -i konstantinjdobler/tight-budget:hindsight -g 0,1,2,3 python train.py --config_path ./cfgs/hindsight.yml --tokenizer ./tokenizers/ar/sp-bpe-ar-32kauto/ -d /path/to/CulturaX-custom-ar-tokenized --out_dir /path/to/ckptdir --precision bf16-true --fsdp_sharding_strategy SHARD_GRAD_OP --mb 1 --model_path /path/to/tuned-embeddings-ckpt
```

We perform an additional ablation were use mixed-precision bfloat16 just for the final annealing phase of the learning rate schedule. We use the final checkpoint of the pure bfloat16 run before the annealing phase as starting point. For this we use:

```sh
bash scripts/run-docker.sh -i konstantinjdobler/tight-budget:hindsight -g 0,1,2,3 python train.py --config_path ./cfgs/hindsight.yml --tokenizer ./tokenizers/{language}/sp-bpe-{language}-32kauto/ -d /path/to/CulturaX-custom-{language}-tokenized --out_dir /path/to/ckptdir --precision bf16-mixed --fsdp_sharding_strategy FULL_SHARD --mb 8 --activation_checkpointing true --gradient_accumulation_no_sync false --saved_checkpoint_path /path/to/bf16-true-ckpt-before-annealing
```

</details>
<p>

### Evaluation

For German, we use [a fork of lm-eval](https://github.com/bjoernpl/lm-evaluation-harness-de/tree/mmlu_de).

Clone the repo and select the fork branch:

```bash
git clone -b mmlu_de git@github.com:bjoernpl/lm-evaluation-harness-de.git
```

install the dependencies and run the evaluations using:

```bash
python eval_de.py --model hf-causal --model_args "pretrained=/path/to/ckpt" --device cuda:0 --output_path /path/to/outdir/output.json --csv_path /path/to/outdir/results.csv
```

Run `mkdir -p /path/to/outdir/` if the directory does not yet exist.

For Arabic, we use `lighteval` and run evaluations on the OALL benchmark suite. Clone `lighteval`:

```bash
git clone git@github.com:huggingface/lighteval.git && cd lighteval && git checkout a98210fd3a2d1e8bface1c32b72ebd5017173a4c
```

install the necessary dependencies and run:

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch run_evals_accelerate.py --model_args "pretrained=/path/to/model" --tasks ./examples/tasks/OALL_tasks.txt --override_batch_size 8 --output_dir=/path/to/output --custom_tasks ./community_tasks/arabic_evals.py
```

### Performance Benchmark

To reproduce our performance benchmark, complete the setup steps and download and preprocess the German CulturaX dataset first. Then run:

```sh
cd scripts/perf_benchmarking && bash run_benchmarks.sh
```

## Citation

Please cite our work as:

```bibtex
@inproceedings{dobler2024language,
    title={Language Adaptation on a Tight Academic Compute Budget: Tokenizer Swapping Works and Pure bfloat16 Is Enough},
    author={Konstantin Dobler and Gerard de Melo},
    booktitle={2nd Workshop on Advancing Neural Network Training: Computational Efficiency, Scalability, and Resource Optimization (WANT@ICML 2024)},
    year={2024},
    url={https://openreview.net/forum?id=VYfJaHeVod}
}
```
