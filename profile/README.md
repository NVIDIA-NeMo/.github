<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

## NVIDIA NeMo Framework Overview

NeMo Framework is NVIDIA's GPU accelerated, end-to-end training framework for large language models (LLMs), multi-modal models and speech models. It enables seamless scaling of training (both pretraining and post-training) workloads from single GPU to thousand-node clusters for both 🤗Hugging Face/PyTorch and Megatron models. This GitHub organization includes a suite of libraries and recipe collections to help users train models from end to end. 

NeMo Framework is also a part of the NVIDIA NeMo software suite for managing the AI agent lifecycle.

## Latest 📣 announcements and 🗣️ discussions 
### 🐳 NeMo AutoModel
- [10/6/2025][Enabling PyTorch Native Pipeline Parallelism for 🤗 Hugging Face Transformer Models](https://github.com/NVIDIA-NeMo/Automodel/discussions/589)
- [9/22/2025][Fine-tune Hugging Face Models Instantly with Day-0 Support with NVIDIA NeMo AutoModel](https://github.com/NVIDIA-NeMo/Automodel/discussions/477)
- [9/18/2025][🚀 NeMo Framework Now Supports Google Gemma 3n: Efficient Multimodal Fine-tuning Made Simple](https://github.com/NVIDIA-NeMo/Automodel/discussions/494)

### 🔬 NeMo RL
- [10/1/2025][On-policy Distillation](https://github.com/NVIDIA-NeMo/RL/discussions/1445)
- [9/27/2025][FP8 Quantization in NeMo RL](https://github.com/NVIDIA-NeMo/RL/discussions/1216)
- [8/15/2025][NeMo-RL: Journey of Optimizing Weight Transfer in Large MoE Models by 10x](https://github.com/NVIDIA-NeMo/RL/discussions/1189)

### 💬 NeMo Speech
- [8/1/2025][Guide to Fine-tune Nvidia NeMo models with Granary Data](https://github.com/NVIDIA-NeMo/NeMo/discussions/14758)

More to come and stay tuned!

## Getting Started

||Installation|Checkpoint Conversion HF<>Megatron|LLM example recipes and scripts|VLM example recipes and scripts|
|-|-|-|-|-|
|1 ～ 1,000 GPUs|[NeMo Automodel](https://github.com/NVIDIA-NeMo/Automodel?tab=readme-ov-file#getting-started), [NeMo RL](https://github.com/NVIDIA-NeMo/RL?tab=readme-ov-file#prerequisites)|No Need|[Pre-training](https://github.com/NVIDIA-NeMo/Automodel?tab=readme-ov-file#llm-pre-training), [SFT](https://github.com/NVIDIA-NeMo/Automodel?tab=readme-ov-file#llm-supervised-fine-tuning-sft), [LoRA](https://github.com/NVIDIA-NeMo/Automodel?tab=readme-ov-file#llm-parameter-efficient-fine-tuning-peft), [DPO](https://github.com/NVIDIA-NeMo/RL/blob/main/examples/run_dpo.py), [GRPO](https://github.com/NVIDIA-NeMo/RL/blob/main/examples/run_grpo_math.py)|[SFT](https://github.com/NVIDIA-NeMo/Automodel?tab=readme-ov-file#vlm-supervised-fine-tuning-sft), [LoRA](https://github.com/NVIDIA-NeMo/Automodel?tab=readme-ov-file#vlm-parameter-efficient-fine-tuning-peft), [GRPO](https://github.com/NVIDIA-NeMo/RL/blob/main/examples/run_vlm_grpo.py)
|Over 1,000 GPUs|[NeMo Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge?tab=readme-ov-file#-installation), [NeMo-RL](https://github.com/NVIDIA-NeMo/RL?tab=readme-ov-file#prerequisites)|[Conversion](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/conversion/README.md)|[Pretrain, SFT, and LoRA](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/recipes/llama/llama3.py), [DPO](https://github.com/NVIDIA-NeMo/RL/blob/main/examples/run_dpo.py) with [megatron_cfg](https://github.com/NVIDIA-NeMo/RL/blob/fa379fffbc9c5580301fa748dbba269c7d90f883/examples/configs/dpo.yaml#L99), [GRPO](https://github.com/NVIDIA-NeMo/RL/blob/main/examples/run_grpo_math.py) with [megatron_cfg](https://github.com/NVIDIA-NeMo/RL/blob/fa379fffbc9c5580301fa748dbba269c7d90f883/examples/configs/grpo_math_1B_megatron.yaml#L79)|[SFT, LoRA](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/recipes/qwen_vl/qwen25_vl.py), [GRPO megatron config](https://github.com/NVIDIA-NeMo/RL/blob/main/examples/configs/vlm_grpo_3B_megatron.yaml)|

## Repo organization under NeMo Framework

### Summary of key functionalities and container strategy of each repo

Visit the individual repos to find out more 🔍, raise :bug:, contribute ✍️ and participate in discussion forums 🗣️!
<p></p>

|Repo|Key Functionality & Documentation Link|Training Loop|Training Backends|Infernece Backends|Model Coverage|Container|
|-|-|-|-|-|-|-|
|[NeMo Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge)|[Pretraining, LoRA, SFT](https://docs.nvidia.com/nemo/megatron-bridge/latest/)|PyT native loop|Megatron-core|NA|LLM & VLM|NeMo Framework Container
|[NeMo AutoModel](https://github.com/NVIDIA-NeMo/Automodel)|[Pretraining, LoRA, SFT](https://docs.nvidia.com/nemo/automodel/latest/index.html)|PyT native loop|PyTorch|NA|LLM, VLM, Omni, VFM|NeMo AutoModel Container|
|[Previous NeMo ->will repurpose to focus on Speech](https://github.com/NVIDIA-NeMo/NeMo)|[Pretraining,SFT](https://docs.nvidia.com/nemo-framework/user-guide/latest/speech_ai/index.html)|PyTorch Lightning Loop|Megatron-core & PyTorch|RIVA|Speech|NA|
|[NeMo RL](https://github.com/NVIDIA-NeMo/RL)|[SFT, RL](https://docs.nvidia.com/nemo/rl/latest/index.html)|PyT native loop|Megatron-core & PyTorch|vLLM|LLM, VLM|NeMo RL container|
|[NeMo Gym](https://github.com/NVIDIA-NeMo/Gym)|[RL Environment, integrate with RL Framework](https://docs.nvidia.com/nemo/gym/latest/index.html)|NA|NA|NA|NA|NeMo RL Container (WIP)|
|[NeMo Aligner (deprecated)](https://github.com/NVIDIA/NeMo-Aligner)|SFT, RL|PyT Lightning Loop|Megatron-core|TRTLLM|LLM|NA
|[NeMo Curator](https://github.com/NVIDIA-NeMo/Curator)|[Data curation](https://docs.nvidia.com/nemo/curator/latest/)|NA|NA|NA|Agnostic|NeMo Curator Container|
|[NeMo Evaluator](https://github.com/NVIDIA-NeMo/Evaluator)|[Model evaluation](https://docs.nvidia.com/nemo/evaluator/latest/)|NA|NA||Agnostic|NeMo Framework Container|
|[NeMo Export-Deploy](https://github.com/NVIDIA-NeMo/Export-Deploy)|[Export to Production](https://docs.nvidia.com/nemo/export-deploy/latest/index.html)|NA|NA|vLLM, TRT, TRTLLM, ONNX|Agnostic|NeMo Framework Container|
|[NeMo Run](https://github.com/NVIDIA-NeMo/Run)|[Experiment launcher](https://docs.nvidia.com/nemo/run/latest/)|NA|NA|NA|Agnostic|NeMo Framework Container|
|[NeMo Guardrails](https://github.com/NVIDIA-NeMo/Guardrails)|[Guardrail model response](https://docs.nvidia.com/nemo/guardrails/latest/)|NA|NA|NA||NA|
|[NeMo Skills](https://github.com/NVIDIA-NeMo/Skills)|[Reference pipeline for SDG & Eval](https://nvidia.github.io/NeMo-Skills/)|NA|NA|NA|Agnostic|NA|
|[NeMo Emerging Optimizers](https://github.com/NVIDIA-NeMo/Emerging-Optimizers)|[Collection of Optimizers](https://docs.nvidia.com/nemo/emerging-optimizers/0.1.0/index.html)|NA|Agnostic|NA|NA|NA|
|[NeMo DFM]([https://github.com/NVIDIA-NeMo/Emerging-Optimizers](https://github.com/NVIDIA-NeMo/DFM/tree/main))|[Diffusion foundation model training]([https://docs.nvidia.com/nemo-framework/user-guide/latest/vlms/index.html](https://github.com/NVIDIA-NeMo/DFM/tree/main/docs))|PyT native loop|Megatron-core and PyTorch|NA|DiT|NA|
|[NeMotron](https://github.com/NVIDIA-NeMo/Nemotron)|Developer asset hub for nemotron models|NA|NA|NA|Nemotron models|NA|
|[NeMo Data Designer](https://github.com/NVIDIA-NeMo/DataDesigner)|Synthetic data generation toolkit|NA|NA|NA|NA|NA|

<div align="center">
  Table 1. NeMo Framework Repos
</div>
<p></p>

### Diagram Ilustration of Repos under NeMo Framework (WIP)

  ![image](/RepoDiagram.png)
  
<div align="center">
  Figure 1. NeMo Framework Repo Overview
</div>
<p></p>

### Some background motivations and historical contexts
The NeMo GitHub Org and its repo collections are created to address the following problems
* **Need for composability**: The [Previous NeMo](https://github.com/NVIDIA/NeMo) is monolithic and encompasses too many things, making it hard for users to find what they need. Container size is also an issue. Breaking down the Monolithic repo into a series of functional-focused repos to facilitate code discovery.
* **Need for customizability**: The [Previous NeMo](https://github.com/NVIDIA/NeMo) uses PyTorch Lighting as the default trainer loop, which provides some out of the box functionality but making it hard to customize. [NeMo Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge), [NeMo AutoModel](https://github.com/NVIDIA-NeMo/Automodel), and [NeMo RL](https://github.com/NVIDIA-NeMo/RL) have adopted pytorch native custom loop to improve flexibility and ease of use for developers. 

<!--
## Contribution & Support

- Follow [Contribution Guidelines](../CONTRIBUTING.md)
- Report issues via GitHub Discussions
- Enterprise support available through NVIDIA AI Enterprise
-->

## License

Apache 2.0 licensed with third-party attributions documented in each repository.
