---
date:
  created: 2025-09-05
categories:
    - automodel
authors:
    - hemil_desai
---

# Challenges in Enabling PyTorch Native Pipeline Parallelism for Hugging Face Transformer Models

<!--
nemo_discussion: {
  "repo": "https://github.com/NVIDIA-NeMo/Automodel",
  "authors": ["hemildesai"]
}
-->
 
## Introduction

As large language models (LLMs) continue to grow in scale - from billions to hundreds of billions of parameters - training these models efficiently across multiple GPU nodes has become increasingly challenging. While data parallelism works well for smaller models, larger models often exceed the memory capacity of a single GPU or a single node, necessitating more sophisticated parallelization strategies.

Pipeline parallelism is one such strategy that addresses this challenge by splitting a model's layers across different devices and processing them in a pipelined fashion. Each device processes a different stage of the model, enabling training of models that wouldn't fit on a single device, while maintaining high GPU utilization through overlapped computation. You can read more about pipeline parallelism in this [PyTorch guide](https://docs.pytorch.org/docs/stable/distributed.pipelining.html) or in the [Megatron-LM paper](https://arxiv.org/pdf/1909.08053).

[NeMo Automodel](https://github.com/NVIDIA-NeMo/Automodel) is a GPU-accelerated PyTorch library for training LLMs. We recently added support for PyTorch native pipeline parallelism via:
1. `AutoPipeline` for any Hugging Face Transformer language model, including popular LLMs in the [AutoModelForCausalLM](https://huggingface.co/transformers/v3.5.1/model_doc/auto.html#automodelforcausallm) category such as **Llama**, **Qwen**, **Mistral**, **Gemma**, with support for vision language models and additional architectures coming soon.
2. A `functional` API for custom models, or for users seeking more granular control. The `functional` API offers modular building blocks that can be adapted to any PyTorch model architecture—making pipeline parallelism accessible across the entire ecosystem.

This article will focus on `AutoPipeline`, and users can refer to the guide [here](https://github.com/NVIDIA-NeMo/Automodel/blob/main/docs/guides/pipelining.md) for more details on the `functional` API.

<!-- more -->

While we drew inspiration from [TorchTitan](https://github.com/pytorch/torchtitan/blob/main/torchtitan/distributed/pipeline_parallel.py) during the development of our pipelining component, enabling automatic pipeline parallelism for Hugging Face models presented a unique set of challenges. In this article, we explore those challenges and share the solutions we implemented in `AutoPipeline` to make pipeline parallelism both robust and user-friendly


## How AutoPipeline Works: High-Level Process

To give you a high-level overview, when you call `AutoPipeline(...).build(model, loss_fn)`, here's what happens under the hood:

1. **Model Analysis**: Detect model structure (has `.model` attribute, number of layers, rotary embeddings, etc.)
2. **Stage Calculation**: Determine virtual stages based on `layers_per_stage` and validate against pipeline size
3. **Module Assignment**: Generate or use provided module names for each pipeline stage (e.g., which layers go to which stage)
4. **Model Splitting**: Deep copy the model (on meta device) for each stage, then remove unneeded modules (keeping only assigned modules per stage)
5. **Stage Creation**: Wrap each model chunk in a `PipelineStage` with proper stage indexing and device placement
6. **Parallelization**: Apply additional parallelization (DP/TP/FSDP) if a `parallelize_fn` is provided
7. **Schedule Building**: Create the pipeline schedule (1f1b, interleaved, etc.) with the stages and loss function

The result is a complete pipeline-parallel setup with automatic handling of all the challenges described in this article.

## Challenge 1: Module Assignment - Understanding Model Structure

When implementing pipeline parallelism, one of the first challenges is determining how to split the model across pipeline stages. This isn't simply a matter of dividing layers equally - certain components need special treatment based on how Hugging Face models are structured.

Let's examine a typical Hugging Face causal language model structure using Qwen3 as an example:

```python
# From transformers/models/qwen3/modeling_qwen3.py

class Qwen3ForCausalLM(Qwen3PreTrainedModel, GenerationMixin):
    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3Model(config)      # Inner model wrapper
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)  # Outside model.model

class Qwen3Model(Qwen3PreTrainedModel):
    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)  # Inside model.model
        self.layers = nn.ModuleList([
            Qwen3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)  # Inside model.model
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)  # Shared utility inside model.model
```

This creates a hierarchical structure where:

- **Top level**: `Qwen3ForCausalLM` contains `model` (inner model) and `lm_head` (output projection)
- **Inner model**: `model.model` contains `embed_tokens`, `layers`, `norm`, and `rotary_emb`

When splitting this model across pipeline stages, different components have different placement requirements:

1. **Input Embeddings (`model.embed_tokens`)**: Must be in the **first stage only** - converts token IDs to embeddings
2. **Transformer Layers (`model.layers`)**: Distributed across **multiple stages** - the core computation
3. **Final Normalization (`model.norm`)**: Must be in the **last or second last stage** - applies final layer normalization
4. **Language Modeling Head (`lm_head`)**: Must be in the **last stage only** - projects to vocabulary logits
5. **Rotary Embeddings (`model.rotary_emb`)**: Must be in **all stages** - shared position encoding utility

These placement constraints become even more pronounced for vision language models and other complex model architectures.

Our `generate_hf_model_fqn_per_model_part` function in `functional.py` handles this complexity automatically for most cases:

```python
# From nemo_automodel/components/distributed/pipelining/functional.py
def generate_hf_model_fqn_per_model_part(
    num_stages: int,
    num_layers: int,
    include_embeddings: bool = True,
    include_lm_head: bool = True,
    include_rotary_emb: bool = True,
    fqn_prefix: str = "model.",  # "model." for nested HF models
) -> list[list[str]]:

    for stage_idx in range(num_stages):
        stage_modules = []

        # First stage: add embeddings if requested
        if stage_idx == 0 and include_embeddings:
            stage_modules.append(f"{fqn_prefix}embed_tokens")  # model.embed_tokens

        # Add transformer layers for this stage
        for _ in range(stage_layer_count):
            stage_modules.append(f"{fqn_prefix}layers.{current_layer}")  # model.layers.X
            current_layer += 1

        # Last stage: add norm and lm_head if requested
        if stage_idx == num_stages - 1:
            stage_modules.append(f"{fqn_prefix}norm")  # model.norm (inside model.model)
            if include_lm_head:
                stage_modules.append("lm_head")  # lm_head (outside model.model, no prefix!)

        if include_rotary_emb:
            # Always include rotary_emb in all stages (shared utility)
            stage_modules.append(f"{fqn_prefix}rotary_emb")  # model.rotary_emb
```

This implementation demonstrates several key insights:

1. **Hierarchical Naming**: The `fqn_prefix="model."` parameter accounts for HuggingFace's nested structure where most components are inside `model.model`

    ```mermaid
    graph TD
      A[Qwen3ForCausalLM] --> B[model: Qwen3Model]
      A --> C[lm_head]
      B --> D[model.embed_tokens]
      B --> E[model.layers]
      B --> F[model.norm]
      B --> G[model.rotary_emb]
    ```

2. **Mixed Hierarchy Handling**: Notice that `lm_head` has no prefix because it lives at the top level (`Qwen3ForCausalLM.lm_head`), while `norm` uses the prefix because it's inside the inner model (`Qwen3ForCausalLM.model.norm`)

    ```mermaid
    graph TD
      A[Qwen3ForCausalLM]
      A --> C[lm_head]
      A --> B[model]
      B --> N[model.norm]
    ```

3. **Shared Component Replication**: The `rotary_emb` is added to **every stage** because position embeddings are needed by all transformer layers

4. **Smart Distribution**: The function automatically calculates how many layers per stage, handling remainder layers by distributing them to the first few stages

To illustrate how this works in practice, consider a 32-layer Qwen3 model split across 4 stages:

```python
[
    # Stage 0: Input processing + first 8 layers + shared utilities
    ["model.embed_tokens", "model.layers.0", ..., "model.layers.7", "model.rotary_emb"],

    # Stage 1: Middle layers + shared utilities
    ["model.layers.8", ..., "model.layers.15", "model.rotary_emb"],

    # Stage 2: Middle layers + shared utilities
    ["model.layers.16", ..., "model.layers.23", "model.rotary_emb"],

    # Stage 3: Final layers + output processing + shared utilities
    ["model.layers.24", ..., "model.layers.31", "model.norm", "lm_head", "model.rotary_emb"]
]
```

This intelligent assignment ensures that each stage has exactly what it needs, while avoiding duplication of unique components like embeddings and the language modeling head. It can also serve as a reference for automatically splitting any custom models for your own use case.


## Challenge 2: nn.ModuleList vs nn.ModuleDict: The Indexing Problem

A subtle but critical issue in pipeline parallelism involves how PyTorch's `nn.ModuleList` and `nn.ModuleDict` behave when models are split across stages. This seemingly minor implementation detail can cause significant problems with checkpointing and state management.

Most Hugging Face models use `nn.ModuleList` to store transformer layers:

```python
# Standard HuggingFace pattern
class TransformerModel(nn.Module):
    def __init__(self):
        self.layers = nn.ModuleList([
            TransformerLayer() for _ in range(32)  # layers 0-31
        ])
```

The problem arises when we split this model across pipeline stages. Each stage gets a subset of the layers, but `nn.ModuleList` automatically re-indexes its contents starting from 0.

```python
# After splitting across 4 stages:
# Stage 0 gets: layers[0:8]   -> Re-indexed as layers[0:8]   ✓ Correct
# Stage 1 gets: layers[8:16]  -> Re-indexed as layers[0:8]   ✗ Wrong!
# Stage 2 gets: layers[16:24] -> Re-indexed as layers[0:8]   ✗ Wrong!
# Stage 3 gets: layers[24:32] -> Re-indexed as layers[0:8]   ✗ Wrong!
```

This seemingly innocent re-indexing creates a disaster scenario for checkpointing:

```python
# During training, model saves checkpoint with state_dict keys:
{
  "stage_0.layers.0.weight": tensor(...),  # Actually layer 0
  "stage_0.layers.1.weight": tensor(...),  # Actually layer 1
  ...
  "stage_1.layers.0.weight": tensor(...),  # Actually layer 8, but saved as 0!
  "stage_1.layers.1.weight": tensor(...),  # Actually layer 9, but saved as 1!
  ...
}

# During loading, this creates total confusion:
# - Stage 1's "layer 0" weights get loaded where layer 8 weights should go
# - Original layer 8-15 weights are completely lost
# - Model convergence is destroyed
```

Fortunately, AutoPipeline solves this by converting `nn.ModuleList` to `nn.ModuleDict` with explicit layer naming:

```python
elif isinstance(module, nn.ModuleList):
    indices_to_keep = {int(idx) for idx in layers_to_keep if idx.isdigit()}
    new_layers = nn.ModuleDict(
        {str(i): layer for i, layer in enumerate(module) if i in indices_to_keep}
    )
    setattr(parent_module, name, new_layers)

# After conversion and splitting:
# Stage 0: {"0": layer_0, "1": layer_1, ..., "7": layer_7}
# Stage 1: {"8": layer_8, "9": layer_9, ..., "15": layer_15}
# Stage 2: {"16": layer_16, "17": layer_17, ..., "23": layer_23}
# Stage 3: {"24": layer_24, "25": layer_25, ..., "31": layer_31}
```

With this approach, checkpoint saving and loading work correctly across all pipeline stages, maintaining the original layer identities throughout the training process.

## Challenge 3: Forward Method Patching: Handling Missing Modules

Another complex challenge in pipeline parallelism is ensuring that forward methods work correctly when modules are distributed across different pipeline stages. Standard Hugging Face forward methods assume all components are available locally, but in pipeline parallelism, this assumption breaks down.

To understand the issue, consider a standard Hugging Face model forward method:

```python
# Standard HuggingFace forward method
def forward(self, input_ids, attention_mask=None, **kwargs):
    # This assumes embed_tokens exists on this stage
    inputs_embeds = self.embed_tokens(input_ids)  # ← Fails on stages 1,2,3!

    hidden_states = inputs_embeds

    # This assumes layers exists on this stage
    for layer in self.layers:  # ← Fails if no layers on this stage!
        hidden_states = layer(hidden_states, **kwargs)

    # This assumes norm exists on this stage
    hidden_states = self.norm(hidden_states)  # ← Fails on stages 0,1,2!

    ...

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )

```

**Problem 1**: When we split the model across stages:
- Stage 0 has `embed_tokens`, but stages 1-3 don't
- Stage 3 has `norm` and `lm_head`, but stages 0-2 don't
- Calling `self.embed_tokens(input_ids)` on stage 1 results in `AttributeError: 'NoneType' object has no attribute '__call__'`

**Problem 2**: PyTorch's Pipeline Parallelism API expects each stage to return a single tensor output, which can be passed to the next stage or used by the loss function in the final stage. However, Hugging Face models typically produce customized outputs, which are not directly compatible with this requirement.

To address these fundamental incompatibilities, AutoPipeline solves this by replacing the standard forward methods with pipeline-aware versions that handle missing modules and outputs gracefully. The actual implementation can be found in [`hf_utils.py`](https://github.com/NVIDIA-NeMo/Automodel/blob/main/nemo_automodel/components/distributed/pipelining/hf_utils.py). AutoPipeline automatically applies these patches based on model type.

Let's examine how this transformation works in practice.

**Before Patching (Fails)**:
```python
# Problem 1: Missing modules
# Stage 1 trying to run standard forward method
hidden_states = self.embed_tokens(input_ids)  # ← AttributeError!
# embed_tokens is None on this stage

# Problem 2: Complex return types
def forward(self, input_ids, **kwargs):
    # ... forward computation ...
    return CausalLMOutputWithPast(  # ← Pipeline expects simple tensor!
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )
```

**After Patching (Works)**:
```python
# Problem 1 Solution: Intelligent module detection
if hasattr(self, "embed_tokens") and self.embed_tokens is not None:
    hidden_states = self.embed_tokens(input_ids)  # First stage
else:
    hidden_states = input_ids  # Middle/last stage - already embeddings

# Problem 2 Solution: Pipeline-aware return types
def pipeline_forward_causal_lm(self, input_ids, **kwargs):
    # Get outputs from the inner model
    outputs = self.model(input_ids=input_ids, **kwargs)
    hidden_states = outputs if isinstance(outputs, torch.Tensor) else outputs.last_hidden_state

    # Apply language modeling head if it exists on this stage
    if hasattr(self, "lm_head") and self.lm_head is not None:
        logits = self.lm_head(hidden_states)
        return logits  # ← Simple tensor for pipeline
    else:
        return hidden_states  # ← Pass hidden states to next stage
```

This comprehensive patching approach solves both the missing module problem and the output compatibility issue, allowing Hugging Face models to work seamlessly with PyTorch's pipeline parallelism API.

While this solution is effective, it does introduce some maintenance considerations. First, we need to keep the patched forward methods in sync whenever `transformers` version is upgraded, otherwise it can cause unexpected errors. Second, not all language models may have the same `forward` method skeleton, which can result in incorrectly patched methods leading to subtle issues.


## Challenge 4: Gradient Scaling

A subtle but critical challenge in pipeline parallelism is ensuring correct gradient scaling when combining multiple parallelism strategies. This issue emerges during real training scenarios and can impact model convergence.

The problem became apparent during convergence testing, where we discovered that pipeline parallel training with mixed parallelism (PP + DP) resulted in different gradient norms compared to training with data parallelism alone. This occurred because, when pipeline parallelism was combined with data parallelism, gradients were incorrectly scaled by default—leading to different gradient norm curves.

According to [PyTorch's pipeline parallelism documentation](https://docs.pytorch.org/docs/stable/distributed.pipelining.html):

> "Gradients are scaled by num_microbatches depending on the scale_grads argument, defaulting to True. This setting should match the configuration of your loss_fn, which may either average losses (scale_grads=True) or sum losses (scale_grads=False)."

However, our training recipes use per-token loss calculation, which required a different approach. As a result, we had to disable automatic scaling in the pipeline schedule (`scale_grads=False`) and handle gradient normalization manually in the training loop, ensuring proper scaling across all parallelism dimensions. This approach gives us precise control over gradient scaling, while maintaining compatibility with our per-token loss calculation.

Specifically, we scale gradients in pipeline parallelism by dividing by a factor of `num_label_tokens_in_batch / dp_group_size`. The `/ dp_group_size` is needed because FSDP averages the gradients across the data parallel ranks during reduction. ([ref](https://github.com/pytorch/pytorch/blob/main/torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py#L697-L698)).

The result is identical loss curves and gradient norm patterns across all parallelism configurations, ensuring that pipeline parallelism maintains correctness.


## Verified HF models supported out of the box

After solving these challenges, many Hugging Face models that previously ran into GPU OOMs now train cleanly with AutoPipeline. Below is a summary of the models we successfully fine-tuned out of the box:

| Model family | Sizes verified | PP sizes used |
| --- | --- | --- |
| Llama (2, 3/3.1/3.3) | 65B, 70B | 4, 8 |
| CodeLlama | 70B | 4, 8 |
| Qwen (1.5, 2, 2.5, Math) | 72B | 4, 8 |
| Mixtral (MoE) | 8x7B (46.7B Total 12.9B Active), 8x22B (141B Total 39B Active) | 4, 8 (8x22B: 8 only) |
| Mistral | Large (123B) | 8 |
| GLM | 32B, 4.5 Air (106B Total 12B Active) | 4, 8 |
| Llama 70B finetunes (Hermes, H2OGPT, ChatQA, Tulu, Nemotron, etc.) | 70B | 4, 8 |

Note: This table summarizes models with at least one finished run. Many additional fine-tuned variants also ran successfully; the table groups them by family for brevity.

## Conclusion

If you are training any HuggingFace Transformer model - Llama, Qwen, Mistral, Gemma, or any other, `AutoPipeline` provides the tools needed to scale your training across multiple GPUs efficiently and correctly.

If you are training custom models and prefer more granular control, the `functional` API provides modular building blocks that can be adapted to any PyTorch model architecture, ensuring that the benefits of pipeline parallelism are accessible across the entire ecosystem.

**Ready to get started?** Check out an example recipe with pipeline parallelism [here](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/llama3_1/llama3_1_8b_hellaswag_pp.yaml) and more [documentation](https://docs.nvidia.com/nemo/automodel/latest/guides/pipelining.html) For questions, issues, or contributions, visit our [GitHub repository](https://github.com/NVIDIA-NeMo/Automodel).

## Contributors

This work wouldn't have been possible without the incredible contributions from our team.

Special thanks to [Huiying Li](https://github.com/HuiyingLi), [Adil Asif](https://github.com/adil-a) and [Alexandros Koumparoulis](https://github.com/akoumpa) for their help adding pipelining support into Automodel - including checkpointing support, recipe integration, convergence sweeps, etc.

Additionally, a huge shoutout to [Wenwen Gao](https://github.com/snowmanwwg), [Bernard Nguyen](https://github.com/bernardwin), and [Jennifer Gerhold](https://github.com/jgerh) for their invaluable guidance on the content — from shaping the narrative to ensuring technical accuracy and clarity.
