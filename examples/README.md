We provide diverse examples about fine-tuning LLMs.

Make sure to execute these commands in the `LLaMA-Factory` directory.

## Table of Contents

- [LoRA Fine-Tuning](#lora-fine-tuning)
- [QLoRA Fine-Tuning](#qlora-fine-tuning)
- [Full-Parameter Fine-Tuning](#full-parameter-fine-tuning)
- [Merging LoRA Adapters and Quantization](#merging-lora-adapters-and-quantization)
- [Inferring LoRA Fine-Tuned Models](#inferring-lora-fine-tuned-models)
- [Extras](#extras)

Use `CUDA_VISIBLE_DEVICES` (GPU) or `ASCEND_RT_VISIBLE_DEVICES` (NPU) to choose computing devices.

## Examples

### LoRA Fine-Tuning

#### (Continuous) Pre-Training

```bash
llamafactory-cli train examples/train_lora/llama3_lora_pretrain.yaml
```

#### Supervised Fine-Tuning

```bash
llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml
```

#### Multimodal Supervised Fine-Tuning

```bash
llamafactory-cli train examples/train_lora/llava1_5_lora_sft.yaml
```

#### Reward Modeling

```bash
llamafactory-cli train examples/train_lora/llama3_lora_reward.yaml
```

#### PPO Training

```bash
llamafactory-cli train examples/train_lora/llama3_lora_ppo.yaml
```

#### DPO/ORPO/SimPO Training

```bash
llamafactory-cli train examples/train_lora/llama3_lora_dpo.yaml
```

#### KTO Training

```bash
llamafactory-cli train examples/train_lora/llama3_lora_kto.yaml
```

#### Preprocess Dataset

It is useful for large dataset, use `tokenized_path` in config to load the preprocessed dataset.

```bash
llamafactory-cli train examples/train_lora/llama3_preprocess.yaml
```

#### Evaluating on MMLU/CMMLU/C-Eval Benchmarks

```bash
llamafactory-cli eval examples/train_lora/llama3_lora_eval.yaml
```

#### Batch Predicting and Computing BLEU and ROUGE Scores

```bash
llamafactory-cli train examples/train_lora/llama3_lora_predict.yaml
```

#### Supervised Fine-Tuning on Multiple Nodes

```bash
FORCE_TORCHRUN=1 NNODES=2 RANK=0 MASTER_ADDR=192.168.0.1 MASTER_PORT=29500 llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml
FORCE_TORCHRUN=1 NNODES=2 RANK=1 MASTER_ADDR=192.168.0.1 MASTER_PORT=29500 llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml
```

#### Supervised Fine-Tuning with DeepSpeed ZeRO-3 (Weight Sharding)

```bash
FORCE_TORCHRUN=1 llamafactory-cli train examples/train_lora/llama3_lora_sft_ds3.yaml
```

### QLoRA Fine-Tuning

#### Supervised Fine-Tuning with 4/8-bit Bitsandbytes Quantization (Recommended)

```bash
llamafactory-cli train examples/train_qlora/llama3_lora_sft_bitsandbytes.yaml
```

#### Supervised Fine-Tuning with 4/8-bit GPTQ Quantization

```bash
llamafactory-cli train examples/train_qlora/llama3_lora_sft_gptq.yaml
```

#### Supervised Fine-Tuning with 4-bit AWQ Quantization

```bash
llamafactory-cli train examples/train_qlora/llama3_lora_sft_awq.yaml
```

#### Supervised Fine-Tuning with 2-bit AQLM Quantization

```bash
llamafactory-cli train examples/train_qlora/llama3_lora_sft_aqlm.yaml
```

### Full-Parameter Fine-Tuning

#### Supervised Fine-Tuning on Single Node

```bash
FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/llama3_full_sft_ds3.yaml
```

#### Supervised Fine-Tuning on Multiple Nodes

```bash
FORCE_TORCHRUN=1 NNODES=2 RANK=0 MASTER_ADDR=192.168.0.1 MASTER_PORT=29500 llamafactory-cli train examples/train_full/llama3_full_sft_ds3.yaml
FORCE_TORCHRUN=1 NNODES=2 RANK=1 MASTER_ADDR=192.168.0.1 MASTER_PORT=29500 llamafactory-cli train examples/train_full/llama3_full_sft_ds3.yaml
```

#### Batch Predicting and Computing BLEU and ROUGE Scores

```bash
llamafactory-cli train examples/train_full/llama3_full_predict.yaml
```

### Merging LoRA Adapters and Quantization

#### Merge LoRA Adapters

Note: DO NOT use quantized model or `quantization_bit` when merging LoRA adapters.

```bash
llamafactory-cli export examples/merge_lora/llama3_lora_sft.yaml
```

#### Quantizing Model using AutoGPTQ

```bash
llamafactory-cli export examples/merge_lora/llama3_gptq.yaml
```

### Inferring LoRA Fine-Tuned Models

#### Use CLI

```bash
llamafactory-cli chat examples/inference/llama3_lora_sft.yaml
```

#### Use Web UI

```bash
llamafactory-cli webchat examples/inference/llama3_lora_sft.yaml
```

#### Launch OpenAI-style API

```bash
llamafactory-cli api examples/inference/llama3_lora_sft.yaml
```

### Extras

#### Full-Parameter Fine-Tuning using GaLore

```bash
llamafactory-cli train examples/extras/galore/llama3_full_sft.yaml
```

#### Full-Parameter Fine-Tuning using BAdam

```bash
llamafactory-cli train examples/extras/badam/llama3_full_sft.yaml
```

#### Full-Parameter Fine-Tuning using QBAdam

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/extras/qbadam/llama3_full_sft_v2_K1000_gc_16.yaml
```

#### LoRA+ Fine-Tuning

```bash
llamafactory-cli train examples/extras/loraplus/llama3_lora_sft.yaml
```

#### Mixture-of-Depths Fine-Tuning

```bash
llamafactory-cli train examples/extras/mod/llama3_full_sft.yaml
```

#### LLaMA-Pro Fine-Tuning

```bash
bash examples/extras/llama_pro/expand.sh
llamafactory-cli train examples/extras/llama_pro/llama3_freeze_sft.yaml
```

#### FSDP+QLoRA Fine-Tuning

```bash
bash examples/extras/fsdp_qlora/single_node.sh
```

#### PiSSA Fine-Tuning

```bash
llamafactory-cli train examples/extras/pissa/llama3_lora_sft.yaml
```

#### Delta Fine-Tuning
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/extras/delta/llama3_delta_sft_bitsandbytes.yaml

CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/extras/delta/llama3_delta_sft_bitsandbytes_K50.yaml

CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/extras/delta/llama3_delta_sft_bitsandbytes_K20.yaml

CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/extras/delta/llama3_delta_sft_bitsandbytes_K20_rank_64.yaml


CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/extras/delta/llama3_delta_simple_sparse_bnb_K50.yaml

CUDA_VISIBLE_DEVICES=3 llamafactory-cli train examples/extras/delta/llama3_delta_simple_sparse_bnb_K20.yaml


CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/extras/delta/llama3_delta_lion_like_v1_bnb_K20.yaml


CUDA_VISIBLE_DEVICES=1 llamafactory-cli train examples/extras/delta/llama3_delta_lion_like_v1_bnb_K20_pdb.yaml


CUDA_VISIBLE_DEVICES=2 llamafactory-cli train examples/extras/delta/qwen0.5_delta_lion_like_v1_bnb_K20_pdb.yaml


CUDA_VISIBLE_DEVICES=1 llamafactory-cli train examples/extras/delta/llama3_delta_lion_like_v2_bnb_K20.yaml


CUDA_VISIBLE_DEVICES=2 llamafactory-cli train examples/extras/delta/llama3_delta_lion_like_v2_bnb_K20_2e2.yaml


CUDA_VISIBLE_DEVICES=1 llamafactory-cli train examples/extras/delta_lion/llama3_delta_lion_K50_gc16_val1k_1e2.yaml

CUDA_VISIBLE_DEVICES=2 llamafactory-cli train examples/extras/delta_lion/llama3_delta_lion_l2_K50_gc16_val1k_1e2.yaml

CUDA_VISIBLE_DEVICES=3 llamafactory-cli train examples/extras/delta_lion/llama3_delta_lion_l2_K50_gc16_val1k_1e1.yaml x


CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/extras/delta_lion/llama3_delta_lion_K50_gc16_val1k_1e1.yaml x 



CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/extras/delta_lion/llama3_delta_lion_K50_gc16_val1k_1e3.yaml


CUDA_VISIBLE_DEVICES=3 llamafactory-cli train examples/extras/delta_lion/llama3_delta_lion_K50_gc16_val1k_1e3.yaml

CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/extras/delta_lion/llama3_delta_lion_l2_K50_gc16_val1k_1e3.yaml

cd /home/ubuntu/date/mq_tst/temp_2/LLaMA-Factory-Badam/
conda activate newpeft


nohup CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/extras/delta_lion/llama3_delta_lion_l2_K50_gc16_val1k_1e3.yaml > /home/yourusername/logs/training.log 2>&1 &
echo $! > /home/yourusername/logs/training.pid








nohup CUDA_VISIBLE_DEVICES=0 llamafactory-cli train examples/extras/delta_only/llama3_delta_only_K50_gc16_val1k_nohup.yaml > /home/yourusername/logs/training.log 2>&1 &
echo $! > /home/yourusername/logs/training.pid