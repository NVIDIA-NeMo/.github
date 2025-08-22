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
* [NeMo Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) (Lightning-free, Megatron-core backend trainng)
* [NeMo AutoModel](https://github.com/NVIDIA-NeMo/Automodel) (Lightning-free, PyTorch backend training)
* [NeMo RL](https://github.com/NVIDIA-NeMo/RL) (Lightning-free, with both PyTorch and Megatron-core backends)
* [NeMo Curator](https://github.com/NVIDIA-NeMo/Curator)
* [NeMo Eval](https://github.com/NVIDIA-NeMo/Eval)
* [NeMo Export-Deploy](https://github.com/NVIDIA-NeMo/Export-Deploy)
* [NeMo Run](https://github.com/NVIDIA-NeMo/Run)
* [Previous NeMo (with Lightning)](https://github.com/NVIDIA/NeMo) (This is the previous NeMo 1.x/2.x repo with Lightning that will be added to the GitHub Org and repurposed to focus on Speech)
* [NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails) (to be added to the Github Org)
* [NeMo Speech](https://github.com/NVIDIA-NeMo) (to be added to the Github Org)
* [NeMo Skills](https://github.com/NVIDIA/NeMo-Skills) (to be added to the Github Org)
* NeMo VFM (coming up - Lightning-free, both Megatron-core and PyTorch backends)
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

## License

Apache 2.0 licensed with third-party attributions documented in each repository.
