<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Framework

NeMo Framework is NVIDIA's GPU accelerated, end-to-end training framework for large language models (LLMs), multi-modal models and speech models. It enables seamless scaling of training (both pretraining and post-training) workloads from single GPU to thousand-node clusters for both 🤗Hugging Face/PyTorch and Megatron models. This GitHub organization includes a suite of libraries and recipe collections to help users train models from end to end. 

NeMo Framework is also a part of the NVIDIA NeMo software suite for managing the AI agent lifecycle.

  ![image](/RepoDiagram.png)
  
<div align="center">
  Figure 1. NeMo Framework Repo Overview
</div>
<p></p>

Visit the individual repos to find out more 🔍, raise :bug:, contribute ✍️ and participate in discussion forums 🗣️!
<p></p>

|Repo|Summary|Training Loop|Training Backends|Infernece Backends|Model Coverage|
|-|-|-|-|-|-|
|[NeMo Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge)|Pretraining, LoRA, SFT|PyT native loop|Megatron-core|NA|LLM & VLM|
|[NeMo AutoModel](https://github.com/NVIDIA-NeMo/Automodel)|Pretraining, LoRA, SFT|PyT native loop|PyTorch DTensor|NA|LLM, VLM, Omni, VFM|
|[NeMo 1.x & 2.x (with Lightning)->will repurpose to focus on Speech](https://github.com/NVIDIA-NeMo/NeMo)|Pretraining,SFT|PyTorch Lightning Loop|PyTorch|RIVA|Speech|
|[NeMo RL](https://github.com/NVIDIA-NeMo/RL)|SFT, RL|PyT native loop|Megatron-core, PyT DTensor|vLLM|LLM, VLM|
|[NeMo Aligner (deprecated)](https://github.com/NVIDIA/NeMo-Aligner)|SFT, RL|PyT Lightning Loop|Megatron-core|TRTLLM|LLM|
|[NeMo Curator](https://github.com/NVIDIA-NeMo/Curator)|Data curation|NA|NA|NA|Agnostic|
|[NeMo Eval](https://github.com/NVIDIA-NeMo/Eval)|Model evaluation|NA|NA||Agnostic|
[NeMo Export-Deploy](https://github.com/NVIDIA-NeMo/Export-Deploy)|Export to Production|NA|NA|vLLM, TRT, TRTLLM, ONNX|Agnostic|
[NeMo Run](https://github.com/NVIDIA-NeMo/Run)|Experiment launcher|NA|NA|NA|Agnostic
[NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails) (to be added to the Github Org)|Guardrail model response|NA|NA|NA||
|[NeMo Skills](https://github.com/NVIDIA/NeMo-Skills) (to be added to the Github Org)|Reference pipeline for SDG & Eval|NA|NA|NA|Agnostic|
|NeMo VFM|Video foundation model training|PyT native loop|Megatron-core and PyTorch|PyTorch|VFM, Diffusion|


<div align="center">
  Table 1. NeMo Framework Repos
</div>
<p></p>

📢 Also take a look at [our blogs](https://nvidia-nemo.github.io/blog/) for the latest exciting things that we are working on!

## Some background contexts and motivations
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

<!-- discussions-index-start -->
## Discussions

### Recent Discussions

### Discussions by Repo
<!-- discussions-index-end -->

## License

Apache 2.0 licensed with third-party attributions documented in each repository.
