# NeMo Framework: Product Walkthrough

> **Audience:** Technical writing team (all levels)  
> **Goal:** Understand what each product is, who it's for, and where it fits  
> **Not covered here:** Deep implementation details, parallelism internals, benchmark methodology

---

## What Is NeMo Framework?

NeMo Framework is NVIDIA's open-source platform for building generative AI models. It covers the **entire model lifecycle** — from preparing training data, to training models, to making them safer and smarter, to evaluating quality, to deploying them in production.

Think of it as a **toolbox with specialized tools for each stage**, not a single monolithic product. Each tool is a separate project (repo) with its own docs, and they're designed to work together.

**The lifecycle in plain terms:**

```
Prepare Data → Train the Model → Align / Improve → Evaluate Quality → Deploy to Production
```

![NeMo Framework Pipeline](assets/diagram-00-nemo-framework-pipeline.png)

---

## Quick Reference

| # | Product | One-Line Summary | Stage | Docs |
|---|---------|-----------------|-------|------|
| 1 | [AutoModel](#1-automodel) | Fine-tune AI models with minimal setup | Training | [docs](https://docs.nvidia.com/nemo/oss/automodel/latest/) |
| 2 | [Curator](#2-curator--video-curator) | Clean and filter training data at scale | Data | [docs](https://docs.nvidia.com/nemo/oss/curator/latest/) |
| 3 | [Customizer](#3-customizer) | Fine-tune models via API (managed service) | Training | Product docs |
| 4 | [Data Designer](#4-data-designer) | Generate synthetic training data | Data | [docs](https://nvidia-nemo.github.io/DataDesigner/latest/) |
| 5 | [Evaluator](#5-evaluator) | Benchmark model quality across 100+ tests | Evaluation | [docs](https://docs.nvidia.com/nemo/oss/evaluator/latest/) |
| 6 | [Gym](#6-gym) | Build practice environments for RL training | Alignment | [docs](https://docs.nvidia.com/nemo/oss/gym/latest/) |
| 7 | [MCORE](#7-mcore-megatron-core) | Low-level engine for large-scale training | Training (engine) | [docs](https://docs.nvidia.com/Megatron-Core/) |
| 8 | [Megatron-Bridge](#8-megatron-bridge) | Train at massive scale (1,000+ GPUs) | Training | [docs](https://docs.nvidia.com/nemo/oss/megatron-bridge/latest/) |
| 9 | [nvFSDP](#9-nvfsdp) | Memory-efficient training technique inside AutoModel | Training (component) | [docs](https://docs.nvidia.com/nemo/oss/automodel/latest/) |
| 10 | [RL](#10-rl) | Improve models using reinforcement learning | Alignment | [docs](https://docs.nvidia.com/nemo/oss/rl/latest/) |
| 11 | [Toolkit (Speech)](#11-toolkit-speech) | Train speech recognition and text-to-speech models | Training | [docs](https://docs.nvidia.com/nemo-framework/user-guide/latest/speech_ai/index.html) |

---

## Key Terms

A short glossary for terms that come up repeatedly across products.

| Term | Plain-English Meaning |
|------|----------------------|
| **Fine-tuning** | Teaching an existing AI model new skills using your own data |
| **SFT (Supervised Fine-Tuning)** | Fine-tuning by showing the model examples of correct answers |
| **LoRA** | A memory-efficient way to fine-tune — updates a small slice of the model instead of the whole thing |
| **RLHF / RL** | Reinforcement Learning — improving a model by giving it feedback (rewards) on its outputs |
| **DPO / GRPO** | Specific RL techniques for aligning model behavior with human preferences |
| **Pretraining** | Training a model from scratch on a large dataset (expensive, rare) |
| **Inference** | Running a trained model to get predictions / answers |
| **Hugging Face (HF)** | The most popular open platform for sharing AI models — NeMo works with HF models directly |
| **NGC** | NVIDIA's container registry — pre-built Docker images optimized for NVIDIA hardware |
| **NIM** | NVIDIA Inference Microservices — a way to deploy models as ready-to-use API endpoints |
| **Checkpoint** | A saved snapshot of a model's learned knowledge (like a save file in a video game) |

---

## 1. AutoModel

![NeMo AutoModel](assets/diagram-03-automodel.png)

**Repo:** [NVIDIA-NeMo/Automodel](https://github.com/NVIDIA-NeMo/Automodel) | **Docs:** [docs.nvidia.com/nemo/oss/automodel](https://docs.nvidia.com/nemo/oss/automodel/latest/)

### What Is It?

AutoModel is the **easiest way to fine-tune AI models on NVIDIA GPUs**. You pick a model from Hugging Face, provide your data, and AutoModel handles the distributed training — whether you have 1 GPU or hundreds.

### How Does It Work?

1. You choose a model from Hugging Face (Llama, Qwen, DeepSeek, Gemma, Mistral, etc.)
2. You write a short YAML config file describing your training job
3. You run one command — AutoModel figures out how to split the work across your GPUs

The key idea: **the same script works on any scale**. You don't rewrite code when you go from a laptop to a cluster.

### Use Cases

- Fine-tune a language model on company-specific data (support tickets, legal docs, medical records)
- Train a vision-language model to understand your domain's images
- Run LoRA (lightweight) fine-tuning when GPU memory is limited
- Pretrain a model from scratch on a custom dataset

### Where It's Positioned

AutoModel is the **recommended starting point** for most training tasks. It works natively with Hugging Face models — no format conversion needed. If you outgrow it (need 1,000+ GPUs), you move to Megatron-Bridge.

### Writer Notes

- Very actively developed — new model support added weekly
- Uses `uv` as its package manager (not just pip)
- Checkpoints from AutoModel plug directly into NeMo RL for alignment

---

## 2. Curator / Video Curator

![NeMo Curator](assets/diagram-01-curator.png)

**Repo:** [NVIDIA-NeMo/Curator](https://github.com/NVIDIA-NeMo/Curator) | **Docs:** [docs.nvidia.com/nemo/oss/curator](https://docs.nvidia.com/nemo/oss/curator/latest/)

### What Is It?

Curator is a **data cleaning and filtering toolkit** for AI training data. "Garbage in, garbage out" — Curator helps you go in with quality data across text, images, video, and audio.

### How Does It Work?

Curator runs GPU-accelerated pipelines that process raw data through stages:

1. **Ingest** — Load data from files, web crawls, or storage
2. **Filter & Classify** — Remove low-quality, duplicate, or unsafe content
3. **Deduplicate** — Find and remove near-identical content (even fuzzy matches)
4. **Output** — Clean, curated dataset ready for training

It works on four data types:

| Data Type | What Curator Does |
|-----------|-------------------|
| **Text** | Filters by quality, detects language, removes duplicates, classifies content |
| **Image** | Scores visual quality, detects inappropriate content, removes duplicates |
| **Video** | Detects scene changes, extracts clips, filters by motion/quality, deduplicates |
| **Audio** | Transcribes speech, checks transcription quality, filters by accuracy |

**Video Curator** is not a separate product — it's the video pipeline within the same Curator library.

### Use Cases

- Clean a web-crawled text dataset before training a language model
- Filter a large image collection for training a vision model
- Process raw video footage into training clips for a video generation model
- Prepare audio datasets for speech recognition training

### Where It's Positioned

Curator sits at the **very beginning of the pipeline** — before any training happens. Its output feeds into AutoModel, Megatron-Bridge, or any other training tool. It's especially valuable at scale (terabytes of data) where GPU acceleration matters most.

### Writer Notes

- Docs are organized by modality: text, image, video, audio — each has its own getting-started guide
- Video Curator is part of the same `nemo-curator` pip package
- Powered by NVIDIA RAPIDS (GPU data processing libraries) + Ray (distributed computing)

---

## 3. Data Designer

![NeMo Data Designer](assets/diagram-02-data-designer.png)

**Repo:** [NVIDIA-NeMo/DataDesigner](https://github.com/NVIDIA-NeMo/DataDesigner) | **Docs:** [nvidia-nemo.github.io/DataDesigner](https://nvidia-nemo.github.io/DataDesigner/latest/)

### What Is It?

Data Designer is a **synthetic data generation** tool. When you don't have enough real training data — or need data with specific properties — Data Designer creates it for you using a combination of statistical rules and LLM generation.

### How Does It Work?

You define a "schema" for the data you want:

1. **Define columns** — Some are generated by statistical rules (e.g., "pick a random product category"), others are generated by an LLM (e.g., "write a customer review for this category")
2. **Set dependencies** — One column's output can feed into another's prompt (so a review matches its category)
3. **Validate** — Built-in validators check that generated data meets your quality rules
4. **Preview → Generate** — Test with a small sample, then scale up

### Use Cases

- Generate training data for a chatbot when you have few real conversations
- Create diverse test datasets for model evaluation
- Build domain-specific instruction-following datasets
- Augment limited real-world data with synthetic examples

### Where It's Positioned

Data Designer sits alongside Curator in the **data preparation stage** — but they solve different problems. Curator *cleans existing data*; Data Designer *creates new data*. They're complementary.

### Writer Notes

- Docs are hosted on GitHub Pages (`nvidia-nemo.github.io`), not `docs.nvidia.com`
- Supports NVIDIA, OpenAI, and OpenRouter as LLM providers
- Has a CLI for configuration: `data-designer config`

---

## 4. Evaluator

![NeMo Evaluator](assets/diagram-07-evaluator.png)

**Repo:** [NVIDIA-NeMo/Evaluator](https://github.com/NVIDIA-NeMo/Evaluator) | **Docs:** [docs.nvidia.com/nemo/oss/evaluator](https://docs.nvidia.com/nemo/oss/evaluator/latest/)

### What Is It?

Evaluator is a **benchmarking platform** for testing how well AI models perform. It runs your model against 100+ standardized tests (benchmarks) and gives you reproducible scores.

### How Does It Work?

1. You point Evaluator at your model (any model that exposes a chat/completion API)
2. You pick which benchmarks to run (math, code, safety, general knowledge, etc.)
3. Evaluator pulls the right test container, runs the benchmark, and reports results
4. All configurations and seeds are saved so results are reproducible

It has two parts:
- **Launcher** (`nemo-evaluator-launcher`) — the CLI you interact with; handles setup and orchestration
- **Engine** (`nemo-evaluator`) — runs the actual benchmark; most users don't touch this directly

### Use Cases

- Compare your fine-tuned model against the base model to measure improvement
- Run a standard benchmark suite before releasing a model
- Test for safety issues (jailbreak vulnerability, bias, hallucination)
- Evaluate code generation, math reasoning, or multilingual capabilities

### Where It's Positioned

Evaluator sits **after training and alignment** — it answers "how good is this model?" It works with any model that has an OpenAI-compatible API, so it's not locked to NeMo-trained models.

### Writer Notes

- Users primarily interact with the launcher CLI, not the engine
- Each benchmark runs in its own Docker container from NGC
- Supports running on local machines, SLURM clusters, and cloud backends

---

## 5. Gym

![NeMo Gym](assets/diagram-06-nemo-gym.png)

**Repo:** [NVIDIA-NeMo/Gym](https://github.com/NVIDIA-NeMo/Gym) | **Docs:** [docs.nvidia.com/nemo/oss/gym](https://docs.nvidia.com/nemo/oss/gym/latest/)

### What Is It?

Gym provides **practice environments** where AI models learn through trial and error (reinforcement learning). Think of it like a training simulator — the model takes actions, the environment gives feedback, and the model improves.

### How Does It Work?

Gym comes with pre-built **environments** ("resource servers") across different skill domains:

| Domain | Example Environments | What the Model Practices |
|--------|---------------------|-------------------------|
| **Agent tasks** | Calendar scheduling, web search, workplace assistant | Multi-step tool use |
| **Coding** | Competitive programming, software engineering | Writing and debugging code |
| **Math** | Math problem solving with verification | Mathematical reasoning |
| **Knowledge** | Multiple-choice Q&A | Factual knowledge (like MMLU) |
| **Instruction following** | Format constraints, structured outputs | Following precise instructions |

Each environment includes a dataset of problems, a way to verify answers, and integration with NeMo RL for training.

### Use Cases

- Train a model to use tools (search, calendar, APIs) through practice
- Improve a model's math or coding skills with verifiable feedback
- Build a custom environment for your domain-specific RL training
- Collect training data (rollouts) for use with NeMo RL

### Where It's Positioned

Gym is a **companion to NeMo RL**. RL provides the training algorithms; Gym provides the practice environments. Together they handle the "alignment" stage of the pipeline.

### Writer Notes

- Early-stage project — APIs are still evolving
- No GPU required for Gym itself (only for the model doing inference)
- Each environment has its own README, config, and dataset

---

## 6. MCORE (Megatron-Core)

**Repo:** [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM) (inside `megatron/core/`) | **Docs:** [docs.nvidia.com/Megatron-Core](https://docs.nvidia.com/Megatron-Core/)

### What Is It?

Megatron-Core (MCORE) is the **engine under the hood** of several NeMo products. It provides the low-level building blocks for training very large models across many GPUs efficiently. Most users never interact with MCORE directly — they use it through Megatron-Bridge or NeMo RL.

### How Does It Work?

When you train a model that's too large to fit on a single GPU, MCORE splits the work across many GPUs using various strategies:

- **Split the model** across GPUs (different layers on different GPUs)
- **Split the data** across GPUs (each GPU processes different examples)
- **Split individual layers** across GPUs (for very wide layers)

It also provides optimized model architectures, efficient checkpointing (saving/loading model snapshots), and mixed-precision training to use less memory.

### Use Cases

- You're building a **custom training framework** and need high-performance distributed training primitives
- You're a **framework developer** integrating with Megatron-based systems

Most end users should use Megatron-Bridge or AutoModel instead.

### Where It's Positioned

MCORE is a **dependency**, not a standalone product for most users:

```
Users interact with:  AutoModel  ←or→  Megatron-Bridge  ←or→  NeMo RL
                           ↓                  ↓                   ↓
Under the hood:        PyTorch            MCORE               MCORE
                       (DTensor)       (Megatron-Core)     (Megatron-Core)
```

### Writer Notes

- Lives in the **NVIDIA/Megatron-LM** repo (different GitHub org than most NeMo repos)
- 15,000+ GitHub stars — one of the most popular NVIDIA open-source projects
- MCORE docs should be written for framework developers, not end users

---

## 7. Megatron-Bridge

![NeMo Megatron-Bridge](assets/diagram-04-megatron-bridge.png)

**Repo:** [NVIDIA-NeMo/Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) | **Docs:** [docs.nvidia.com/nemo/oss/megatron-bridge](https://docs.nvidia.com/nemo/oss/megatron-bridge/latest/)

### What Is It?

Megatron-Bridge is a **training library for extreme scale** — designed for training runs on 1,000+ GPUs. It connects Hugging Face models with MCORE's powerful distributed training engine.

### How Does It Work?

The "bridge" metaphor is literal — it bridges two worlds:

- **Hugging Face world:** Where models are shared, easy to use, and widely adopted
- **Megatron world:** Where training is maximally optimized for NVIDIA hardware at massive scale

Megatron-Bridge converts models between these formats (in both directions) and provides a training loop on top. You can:

1. Start from a Hugging Face model
2. Convert it to Megatron format (automatically, using the `AutoBridge` API)
3. Train at massive scale with MCORE
4. Convert back to Hugging Face format for sharing or deployment

### Use Cases

- Pretraining or fine-tuning at 1,000+ GPU scale where maximum throughput matters
- Organizations that need Megatron-level performance but want Hugging Face compatibility
- Converting between Hugging Face and Megatron checkpoint formats

### Where It's Positioned

Megatron-Bridge is the **heavy-duty training option** — complementary to AutoModel:

| | AutoModel | Megatron-Bridge |
|---|---|---|
| **Best for** | Getting started, rapid iteration | Maximum scale and throughput |
| **GPU sweet spot** | 1 to ~1,000 GPUs | 1,000+ GPUs |
| **Model source** | Hugging Face (native) | Hugging Face (via conversion) |
| **Underlying engine** | PyTorch (DTensor) | Megatron-Core |

### Writer Notes

- Refactored from the original NeMo training stack
- Has been adopted by several community projects (VeRL, SkyRL, Slime)
- Recipes for popular models live in `src/megatron/bridge/recipes/`

---

## 9. nvFSDP

**Location:** Inside [AutoModel](https://github.com/NVIDIA-NeMo/Automodel) | **Docs:** [docs.nvidia.com/nemo/oss/automodel](https://docs.nvidia.com/nemo/oss/automodel/latest/)

### What Is It?

nvFSDP is **not a standalone product** — it's a component inside AutoModel that handles **memory-efficient distributed training**. It automatically splits a model's memory footprint across GPUs so you can train models that are too large to fit on a single GPU.

### How Does It Work?

In simple terms: instead of every GPU holding a full copy of the model, nvFSDP shards (splits) the model's weights, gradients, and optimizer state across all GPUs. Each GPU holds only a fraction, and they coordinate during training.

This is NVIDIA's optimized version of PyTorch's FSDP2 (Fully Sharded Data Parallel) technology.

### Use Cases

- Training large models on limited GPU memory
- Users don't interact with nvFSDP directly — they benefit from it automatically when using AutoModel

### Where It's Positioned

nvFSDP is an **implementation detail** of AutoModel. Users configure it through AutoModel's YAML settings. Writers may encounter the term in code or internal docs, but externally it's usually referred to as "FSDP2."

### Writer Notes

- Not a separate repo, pip package, or product — it's part of AutoModel's internals
- The name "nvFSDP" may appear in internal docs; externally prefer "FSDP2" or "FSDP2 Strategy"
- NeMo RL's DTensor backend also uses this under the hood (via AutoModel)

---

## 10. RL

![NeMo RL](assets/diagram-05-nemo-rl.png)

**Repo:** [NVIDIA-NeMo/RL](https://github.com/NVIDIA-NeMo/RL) | **Docs:** [docs.nvidia.com/nemo/oss/rl](https://docs.nvidia.com/nemo/oss/rl/latest/)

### What Is It?

NeMo RL is a **post-training library** that makes AI models better through reinforcement learning — the model generates outputs, gets feedback on quality, and learns to produce better results over time.

### How Does It Work?

The core loop:

1. **Generate** — The model produces responses to prompts
2. **Score** — A reward system evaluates the responses (correct math? followed instructions? safe output?)
3. **Train** — The model updates its behavior based on the scores
4. Repeat

NeMo RL supports several training techniques:

| Technique | What It Does |
|-----------|-------------|
| **GRPO** | Reinforcement learning using group-relative scoring — the main RL method |
| **DPO** | Learns from pairs of "preferred vs. rejected" responses |
| **SFT** | Standard supervised fine-tuning (also available here for convenience) |
| **Distillation** | A smaller "student" model learns from a larger "teacher" model |
| **Reward Modeling** | Trains a model to predict human preferences (used as the scorer) |

### Use Cases

- Improve a model's math and reasoning abilities using verified solutions
- Align a model with human preferences (helpful, harmless, honest)
- Train a model to use tools through multi-turn interaction
- Distill a large model's capabilities into a smaller, cheaper model

### Where It's Positioned

RL covers the **alignment stage** — after initial training (AutoModel / Megatron-Bridge) and before evaluation. It works with:

- **NeMo Gym** for practice environments and reward signals
- **AutoModel** or **Megatron-Bridge** as its training engine (user picks via config)
- **Evaluator** to measure improvement after alignment

### Writer Notes

- Most-starred repo in the NVIDIA-NeMo org (1,300+ stars)
- Used to train NVIDIA's Nemotron-3-Nano-30B model
- Training backend (AutoModel vs. Megatron) is selected automatically based on the YAML config
- Uses Ray for distributed infrastructure

---

## 11. Toolkit (Speech)

**Repo:** [NVIDIA-NeMo/NeMo](https://github.com/NVIDIA-NeMo/NeMo) | **Docs:** [docs.nvidia.com/nemo-framework (Speech AI)](https://docs.nvidia.com/nemo-framework/user-guide/latest/speech_ai/index.html)

### What Is It?

The NeMo Toolkit is the **speech AI** component of the framework, providing tools for training automatic speech recognition (ASR) and text-to-speech (TTS) models.

### How Does It Work?

- **ASR (Speech-to-Text):** Train models that convert spoken audio into text — supports multiple languages
- **TTS (Text-to-Speech):** Train models that generate natural-sounding speech from text
- Comes with a large collection of **pretrained models** that can be fine-tuned for specific domains, accents, or languages

### Use Cases

- Build a custom speech recognition system for a specific industry (medical, legal, call center)
- Create a text-to-speech voice for a specific language or brand
- Fine-tune an existing speech model on your organization's vocabulary

### Where It's Positioned

This is the **original NeMo repository**. Historically, it contained everything (LLM, speech, vision). Over time, LLM training was split into AutoModel and Megatron-Bridge. This repo now **primarily houses the speech workloads**.

NeMo Curator's audio pipeline can prepare data that feeds into Toolkit (Speech) for training.

### Writer Notes

- The repo is named "NeMo" which can be confusing — "NeMo" (the repo) now means "speech," while "NeMo Framework" means the entire ecosystem
- Docs live under the NeMo Framework User Guide, not a standalone site
- This repo is large and historically complex — speech docs should be clearly scoped

---

## How the Products Connect

![NeMo Framework Interoperability Map](assets/diagram-13-interoperability-map.png)

Here's how data and models flow between the products:

| Step | What Happens | Products Involved |
|------|-------------|-------------------|
| 1. **Prepare data** | Clean real data or generate synthetic data | Curator, Data Designer |
| 2. **Train** | Fine-tune or pretrain a model | AutoModel (standard) or Megatron-Bridge (large scale) |
| 3. **Align** | Improve the model with RL or preference learning | RL + Gym |
| 4. **Evaluate** | Benchmark quality across standardized tests | Evaluator |
| 5. **Deploy** | Export and serve the model in production | (Export-Deploy, Guardrails — not covered in this deck) |

**Alternatively:** Use **Customizer** for steps 2–3 via API instead of running open-source code.

**Under the hood:** MCORE powers Megatron-Bridge and RL's Megatron backend. nvFSDP powers AutoModel and RL's DTensor backend. **NeMo Run** (not covered in this deck) is an experiment launcher that works across all products.

---

## Comparison Cheat Sheet

### "I want to train a model" — Which product?

| Situation | Use This |
|-----------|----------|
| Fine-tune a Hugging Face model on my data | **AutoModel** |
| Train at massive scale (1,000+ GPUs) | **Megatron-Bridge** |
| Fine-tune via API without managing infrastructure | **Customizer** |
| Train a speech recognition or TTS model | **Toolkit (Speech)** |

### "I want to improve a trained model" — Which product?

| Situation | Use This |
|-----------|----------|
| Align with human preferences (DPO, GRPO) | **RL** |
| Train a model to use tools via practice | **RL + Gym** |
| Distill a large model into a smaller one | **RL** (on-policy distillation) |

### "I need training data" — Which product?

| Situation | Use This |
|-----------|----------|
| Clean / filter / deduplicate existing data | **Curator** |
| Generate synthetic data from scratch | **Data Designer** |

---

## Documentation Landscape

Not all products are documented in the same place:

| Docs Host | Products |
|-----------|----------|
| `docs.nvidia.com/nemo/oss/...` | AutoModel, Megatron-Bridge, RL, Gym, Evaluator, Curator |
| `docs.nvidia.com/Megatron-Core/` | MCORE |
| `nvidia-nemo.github.io/...` | Data Designer, Skills |
| `docs.nvidia.com/nemo-framework/user-guide/...` | Toolkit (Speech) |

### Open-Source vs. Managed

| Type | Products |
|------|----------|
| **Open-source (GitHub)** | AutoModel, Curator, Data Designer, Evaluator, Gym, MCORE, Megatron-Bridge, RL, Toolkit |
| **Closed-source (microservice)** | Customizer |
| **Internal component** (not a standalone product) | nvFSDP |
