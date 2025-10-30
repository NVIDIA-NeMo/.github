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
- [10/1/2025][On-policy Distillation (Qwen3-style) (Qwen3-style)](https://github.com/NVIDIA-NeMo/RL/discussions/1254)
- [9/27/2025][FP8 Quantization in NeMo RL](https://github.com/NVIDIA-NeMo/RL/discussions/1216)
- [8/15/2025][NeMo-RL: Journey of Optimizing Weight Transfer in Large MoE Models by 10x](https://github.com/NVIDIA-NeMo/RL/discussions/1189)
- [7/31/2025][NeMo-RL V0.3: Scalable and Performant Post-training with Nemo-RL via Megatron-Core](https://github.com/NVIDIA-NeMo/RL/discussions/1161)
- [5/15/2025][Reinforcement Learning with NVIDIA NeMo-RL: Reproducing a DeepScaleR Recipe Using GRPO ](https://github.com/NVIDIA-NeMo/RL/discussions/1188)

### 💬 NeMo Speech
- [8/1/2025][Guide to Fine-tune Nvidia NeMo models with Granary Data](https://github.com/NVIDIA-NeMo/NeMo/discussions/14758)

More to come and stay tuned!

## Repo organization under NeMo Framework
  ![image](/RepoDiagram.png)
  
<div align="center">
  Figure 1. NeMo Framework Repo Overview
</div>
<p></p>

## Summary of key functionalities and container strategy of each repo

Visit the individual repos to find out more 🔍, raise :bug:, contribute ✍️ and participate in discussion forums 🗣️!
<p></p>

|Repo|Summary|Training Loop|Training Backends|Infernece Backends|Model Coverage|Container|
|-|-|-|-|-|-|-|
|[NeMo Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge)|Pretraining, LoRA, SFT|PyT native loop|Megatron-core|NA|LLM & VLM|NeMo Framework Container
|[NeMo AutoModel](https://github.com/NVIDIA-NeMo/Automodel)|Pretraining, LoRA, SFT|PyT native loop|PyTorch|NA|LLM, VLM, Omni, VFM|NeMo AutoModel Container|
|[NeMo 1.x & 2.x (with Lightning)->will repurpose to focus on Speech](https://github.com/NVIDIA-NeMo/NeMo)|Pretraining,SFT|PyTorch Lightning Loop|Megatron-core & PyTorch|RIVA|Speech|NA|
|[NeMo RL](https://github.com/NVIDIA-NeMo/RL)|SFT, RL|PyT native loop|Megatron-core & PyTorch|vLLM|LLM, VLM|NeMo RL container|
|NeMo Gym (WIP)|RL Environment, integrate with RL Framework|NA|NA|NA|NA|NeMo RL Container|
|[NeMo Aligner (deprecated)](https://github.com/NVIDIA/NeMo-Aligner)|SFT, RL|PyT Lightning Loop|Megatron-core|TRTLLM|LLM|NA
|[NeMo Curator](https://github.com/NVIDIA-NeMo/Curator)|Data curation|NA|NA|NA|Agnostic|NeMo Curator Container|
|[NeMo Evaluator](https://github.com/NVIDIA-NeMo/Evaluator)|Model evaluation|NA|NA||Agnostic|NeMo Framework Container|
|[NeMo Export-Deploy](https://github.com/NVIDIA-NeMo/Export-Deploy)|Export to Production|NA|NA|vLLM, TRT, TRTLLM, ONNX|Agnostic|NeMo Framework Container|
|[NeMo Run](https://github.com/NVIDIA-NeMo/Run)|Experiment launcher|NA|NA|NA|Agnostic|NeMo Framework Container|
|[NeMo Guardrails](https://github.com/NVIDIA-NeMo/Guardrails)|Guardrail model response|NA|NA|NA||NA|
|[NeMo Skills](https://github.com/NVIDIA-NeMo/Skills)|Reference pipeline for SDG & Eval|NA|NA|NA|Agnostic|NA|
|[NeMo Emerging Optimizers](https://github.com/NVIDIA-NeMo/Emerging-Optimizers)|Collection of Optimizers|NA|Agnostic|NA|NA|NA|
|NeMo DFM (WIP)|Diffusion foundation model training|PyT native loop|Megatron-core and PyTorch|PyTorch|VFM, Diffusion|TBD|
|[NeMotron](https://github.com/NVIDIA-NeMo/Nemotron)|Developer asset hub for nemotron models|NA|NA|NA|Nemotron models|NA|
|NeMo Data-designer (WIP)|Synthetic data generation toolkit|NA|NA|NA|NA|NA|


<div align="center">
  Table 1. NeMo Framework Repos
</div>
<p></p>

### Some background contexts and motivations
The NeMo GitHub Org and its repo collections are created to address the following problems
* **Need for composability**: The [Previous NeMo](https://github.com/NVIDIA/NeMo) is monolithic and encompasses too many things, making it hard for users to find what they need. Container size is also an issue. Breaking down the Monolithic repo into a series of functional-focused repos to facilitate code discovery.
* **Need for customizability**: The [Previous NeMo](https://github.com/NVIDIA/NeMo) uses PyTorch Lighting as the default trainer loop, which provides some out of the box functionality but making it hard to customize. [NeMo Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge), [NeMo AutoModel](https://github.com/NVIDIA-NeMo/Automodel), and [NeMo RL](https://github.com/NVIDIA-NeMo/RL) have adopted pytorch native custom loop to improve flexibility and ease of use for developers. 

## Documentation

To learn more about NVIDIA NeMo Framework and all of its component libraries, please refer to the [NeMo Framework User Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/overview.html), which includes quick start guide, tutorials, model-specific recipes, best practice guides and performance benchmarks.  

<!--
## Contribution & Support

- Follow [Contribution Guidelines](../CONTRIBUTING.md)
- Report issues via GitHub Discussions
- Enterprise support available through NVIDIA AI Enterprise
-->

## License

Apache 2.0 licensed with third-party attributions documented in each repository.
