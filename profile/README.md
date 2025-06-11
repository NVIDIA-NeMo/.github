<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NVIDIA NeMo Framework

NeMo Framework is NVIDIA's GPU accelerated, end-to-end training framework for large language models (LLMs), multi-modal models and speech models. It enables seamless scaling of training (both pretraining and post-training) workloads from single GPU to thousand-node clusters for both :hugs:Hugging Face and Megatron models. 

This GitHub organization hosts repositories for NeMo's core components and integrations, including:

**[Nemo RL](https://github.com/NVIDIA-NeMo/rl)**

- State of the art post-training techniques such as GRPO, DPO, SFT etc.Distributed inference runtime with Rust-based orchestration
- Seamless integration with :hugs:Hugging Face for users to try out a wide range of models
- High performance implementation with Megatron Core with various parallelisms for large models and long context lengths.

**[Nemo Curator](https://github.com/NVIDIA-NeMo/curator)**

- Fast and scalable dataset preparation and curation for both pretraining and post-training pipelines.
- Significant time savings by leveraging GPUs with Dask and RAPIDS.
- Customizable and modular interface, simplifying pipeline expansion and accelerating model convergence through the preparation of high-quality tokens.




## Documentation

To learn more about NVIDIA NeMo Framework and all of its various component libraries, please refer to the [NeMo Framework User Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/overview.html), which includes quick start guide, tutorials, model-specific recipes, best practice guides and performance benchmarks.  

## Contribution & Support

- Follow [Contribution Guidelines](../CONTRIBUTING.md)
- Report issues via GitHub Discussions
- Enterprise support available through NVIDIA AI Enterprise

## License

Apache 2.0 licensed with third-party attributions documented in each repository.
