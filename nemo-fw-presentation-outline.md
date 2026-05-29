# NeMo Framework: Technical Writer Presentation Outline

> **Audience:** Technical writing team  
> **Goal:** Introduce what NeMo Framework is, walk through each repo, and highlight what writers need to know  
> **Estimated time:** 45–60 min  
> **Diagrams:** See the `assets/` folder for per-product architecture diagrams

---

## 1. What Is NeMo Framework?

- **One-liner:** An open-source collection of NVIDIA libraries that covers every stage of the generative AI model lifecycle — from data curation, to training, alignment, evaluation, and deployment.
- Supports LLMs, VLMs, Speech, and Diffusion models.
- Scales from a single GPU on a workstation to 10,000+ GPU nodes on SLURM/Kubernetes clusters.
- Day-0 Hugging Face support: users can train virtually any model on the HF Hub without format conversion.
- All repos live under the **[NVIDIA-NeMo](https://github.com/NVIDIA-NeMo)** GitHub org. Apache 2.0 licensed.

### Key Talking Points

- NeMo Framework is **not** a single repo — it is an **ecosystem of ~15 focused repositories**.
- Each repo has its own docs site, container, and release cadence.
- Optimized NGC containers are published for the core repos (AutoModel, Megatron-Bridge, RL, Curator, Evaluator, Export-Deploy).

---

## 2. The Pipeline at a Glance

![NeMo Framework Pipeline](assets/diagram-00-nemo-framework-pipeline.png)

```
Data ──▶ Training ──▶ Alignment ──▶ Evaluation ──▶ Deployment
```

| Stage | Primary Repos |
|-------|---------------|
| **Data** | Curator, Data Designer, Skills |
| **Training** | AutoModel, Megatron-Bridge, NeMo Speech, DFM, Emerging-Optimizers |
| **Alignment** | NeMo RL, NeMo Gym |
| **Evaluation** | Evaluator, Skills |
| **Deployment** | Export-Deploy, Guardrails |
| **Infrastructure** | NeMo Run |
| **Models/Recipes** | Nemotron |

---

## 3. Data Stage

### 3a. NeMo Curator

![NeMo Curator](assets/diagram-01-curator.png)

- **Repo:** [NVIDIA-NeMo/Curator](https://github.com/NVIDIA-NeMo/Curator) — 1,394 stars
- **What it does:** GPU-accelerated data curation at scale for training better AI models.
- **Modalities:** Text, Image, Video, Audio.
- **Highlights for writers:**
  - **Text:** 30+ heuristic filters, fuzzy/exact/semantic deduplication (MinHash LSH), language detection, quality classification.
  - **Image:** CLIP embeddings, aesthetic filtering, NSFW detection, deduplication.
  - **Video:** Scene detection (TransNetV2), clip extraction, motion/aesthetic filtering, GPU H.264 encoding, Cosmos-Embed1 embeddings.
  - **Audio:** ASR transcription, WER filtering, quality assessment.
  - Powered by **NVIDIA RAPIDS** (cuDF, cuML, cuGraph) + Ray for multi-node scaling.
  - Proven results: 16x faster fuzzy dedup on 8 TB dataset; 40% lower TCO vs CPU.
- **Docs:** [docs.nvidia.com/nemo-oss/curator](https://docs.nvidia.com/nemo-oss/curator/latest/)
- **Container:** [NGC NeMo Curator](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo-curator)

### 3b. NeMo Data Designer

![NeMo Data Designer](assets/diagram-02-data-designer.png)

- **Repo:** [NVIDIA-NeMo/DataDesigner](https://github.com/NVIDIA-NeMo/DataDesigner) — 698 stars
- **What it does:** Generate high-quality **synthetic datasets** from scratch or from seed data.
- **Highlights for writers:**
  - Statistical samplers for controlled distributions (category, numeric, etc.).
  - LLM-powered text generation columns with dependency-aware field generation.
  - Built-in validators (Python, SQL, custom) and LLM-as-a-judge scoring.
  - Preview mode for fast iteration before full-scale generation.
  - Supports NVIDIA Build API, OpenAI, OpenRouter, and custom providers.
  - CLI for model/provider configuration (`data-designer config`).
  - Collects anonymized telemetry on model usage (opt-out available).
- **Docs:** [nvidia-nemo.github.io/DataDesigner](https://nvidia-nemo.github.io/DataDesigner/latest/)

### 3c. NeMo Skills (Data Side)

![NeMo Skills](assets/diagram-10-skills.png)

- **Repo:** [NVIDIA-NeMo/Skills](https://github.com/NVIDIA-NeMo/Skills) — 816 stars
- **What it does:** Synthetic data generation pipelines for math, code, and science datasets.
- **Highlights for writers:**
  - End-to-end SDG pipelines: generate → filter → train → evaluate.
  - Released major open datasets: OpenMathInstruct-2 (14M pairs), OpenMathReasoning (3.2M CoT solutions), OpenScienceReasoning-2.
  - Flexible LLM inference: API providers, local servers, large-scale SLURM jobs.
  - Host models with TensorRT-LLM, vLLM, SGLang, or Megatron.
- **Docs:** [nvidia-nemo.github.io/Skills](https://nvidia-nemo.github.io/Skills/)

---

## 4. Training Stage

### 4a. NeMo AutoModel (Primary — up to ~1K GPUs)

- **Repo:** [NVIDIA-NeMo/Automodel](https://github.com/NVIDIA-NeMo/Automodel) — 288 stars
- **What it does:** PyTorch DTensor-native SPMD training library for LLMs and VLMs with out-of-the-box Hugging Face support.
- **Highlights for writers:**
  - **SPMD philosophy:** Same script runs on 1 GPU or 1000+ — parallelism is configuration, not code rewrites.
  - **YAML-driven recipes** with CLI overrides for any field.
  - **Supported tasks:** Pretraining, SFT, LoRA (PEFT), Knowledge Distillation.
  - **Model coverage (LLM):** Llama 3.x, Qwen 2.5/3, DeepSeek V3/V3.2, Gemma 2/3, Mistral/Mixtral, Phi 2/3/4, GPT-OSS, Nemotron, Moonlight, Baichuan, Seed, GLM, MiniMax, Step — essentially *any* HF causal LM.
  - **Model coverage (VLM):** Gemma 3 VL, Gemma 3n VL, Qwen2.5 VL, Qwen3 VL 235B, Kimi K2.5 VL.
  - **Parallelism:** FSDP2, Tensor Parallel, Context Parallel, Sequence Parallel, Pipeline Parallel (3D parallelism).
  - **Performance features:** FP8 via torchao, sequence packing, distributed checkpointing (SafeTensors).
  - **Performance numbers:** DeepSeek V3 671B → 250 TFLOPs/GPU on 256 GPUs; GPT-OSS 20B → 279 TFLOPs/GPU.
  - Actively developed — new model support weekly (MiniMax-M2, DeepSeek V3.2, Step 3.5-flash in Feb 2026).
  - Install: `pip install nemo-automodel` or `uv sync`.
  - **Launch options:** `torchrun`, `automodel` CLI (interactive + SLURM), Kubernetes (coming).
- **Docs:** [docs.nvidia.com/nemo-oss/automodel](https://docs.nvidia.com/nemo-oss/automodel/latest/)
- **Container:** [NGC NeMo AutoModel](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo-automodel)

### 4b. NeMo Megatron-Bridge (Scale — 1K+ GPUs)

- **Repo:** [NVIDIA-NeMo/Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) — 423 stars
- **What it does:** Training library that bridges Hugging Face and [Megatron-Core](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core) for maximum throughput at extreme scale.
- **Highlights for writers:**
  - **Core capability:** Bidirectional checkpoint conversion between HF and Megatron formats — online, parallelism-aware, memory-efficient.
  - **`AutoBridge` API:** Auto-detect model architecture, convert, and materialize Megatron models in a few lines of Python.
  - **Supported parallelisms:** TP, PP, VPP, CP, EP, ETP — 6D parallelism for near-linear scaling to thousands of nodes.
  - **Training features:** Pretraining, SFT, LoRA/DoRA (PEFT), FP8/BF16/FP4 mixed precision.
  - **Model coverage:** Llama 2–3.3, Qwen 2–3 (incl. MoE and VL), DeepSeek V2/V3, Gemma/Gemma 3 VL, Nemotron-H, Nemotron Nano v2/VL, GPT-OSS, GLM-4.5, Mistral/Ministral, Moonlight, OlMoE.
  - **PyTorch-native training loop** — refactored from the legacy NeMo training stack for greater flexibility.
  - Community adoptions: VeRL, Slime, SkyRL, Mind Lab (trained trillion-parameter GRPO LoRA on 64 H800s).
- **Docs:** [docs.nvidia.com/nemo-oss/megatron-bridge](https://docs.nvidia.com/nemo-oss/megatron-bridge/latest/)
- **Container:** [NGC NeMo Framework](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo)

### 4c. NeMo Speech

- **Repo:** [NVIDIA-NeMo/NeMo](https://github.com/NVIDIA-NeMo/NeMo)
- **What it does:** Pretraining and SFT for speech AI models (ASR, TTS) using Megatron-Core.
- **Highlights for writers:**
  - Covers automatic speech recognition (ASR) and text-to-speech (TTS).
  - Built on Megatron-Core backend.
  - Part of the original NeMo monorepo, now housing the speech-specific workloads.
- **Docs:** [docs.nvidia.com/nemo-framework (Speech AI)](https://docs.nvidia.com/nemo-framework/user-guide/latest/speech_ai/index.html)

### 4d. NeMo DFM (Diffusion Foundation Models)

- **Repo:** [NVIDIA-NeMo/DFM](https://github.com/NVIDIA-NeMo/DFM) — 29 stars
- **What it does:** Training and inference for diffusion models (video, image, text generation).
- **Highlights for writers:**
  - **Dual-path architecture:** Megatron Bridge path (max scalability) and AutoModel path (easy experimentation).
  - **Supported models:** DiT (Diffusion Transformers), WAN 2.1 (World Action Networks for video).
  - Features: Flow Matching, EDM samplers, sequence packing, distributed checkpointing.
  - YAML-driven recipes, `uv run` for reproducible environments.
- **Docs:** [github.com/NVIDIA-NeMo/DFM/docs](https://github.com/NVIDIA-NeMo/DFM/tree/main/docs)

### 4e. Emerging-Optimizers

- **Repo:** [NVIDIA-NeMo/Emerging-Optimizers](https://github.com/NVIDIA-NeMo/Emerging-Optimizers)
- **What it does:** Collection of cutting-edge optimizers (e.g., Muon, Dion) for use across training libraries.
- **Docs:** [docs.nvidia.com/nemo-oss/emerging-optimizers](https://docs.nvidia.com/nemo-oss/emerging-optimizers/latest/index.html)

---

## 5. Alignment Stage

### 5a. NeMo RL

- **Repo:** [NVIDIA-NeMo/RL](https://github.com/NVIDIA-NeMo/RL) — 1,306 stars (most-starred in the org)
- **What it does:** Scalable post-training library for reinforcement learning on LLMs and VLMs.
- **Highlights for writers:**
  - **Algorithms:** GRPO, GSPO, DAPO, DPO, SFT (w/ LoRA), Reward Modeling (RM), On-policy Distillation.
  - **Multi-turn RL:** Tool use, games, multi-step environments.
  - **Async RL:** Asynchronous rollouts + replay buffers for fully async GRPO.
  - **Two training backends:**
    - **DTensor** (PyTorch-native FSDP2, TP, CP, SP, PP) — via NeMo AutoModel.
    - **Megatron-Core** (6D parallelism) — via Megatron-Bridge.
  - **Two generation backends:** vLLM and Megatron Inference.
  - **End-to-end FP8** training + FP8 vLLM generation.
  - **VLM support:** SFT and GRPO for vision-language models.
  - Ray-based infrastructure for resource management and worker isolation.
  - Used to train [Nemotron-3-Nano-30B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8).
  - Latest release: v0.5.0 (Jan 2026) with LoRA support for DTensor and Megatron backends.
  - Install: `uv venv && uv run python examples/run_grpo.py`
- **Docs:** [docs.nvidia.com/nemo-oss/rl](https://docs.nvidia.com/nemo-oss/rl/latest/)
- **Container:** [NGC NeMo RL](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo-rl)

### 5b. NeMo Gym

- **Repo:** [NVIDIA-NeMo/Gym](https://github.com/NVIDIA-NeMo/Gym) — 637 stars
- **What it does:** Build RL training environments for LLMs — provides infrastructure to develop environments, scale rollout collection, and integrate with training frameworks.
- **Highlights for writers:**
  - Scaffolding for multi-step, multi-turn, and user-modeling RL scenarios.
  - Growing collection of **resource servers** (training environments):
    - **Agent:** Calendar scheduling, Google Search, Workplace Assistant, Math Advanced Calculations.
    - **Coding:** Competitive coding (Code Gen), Mini SWE Agent (SWE-bench style).
    - **Knowledge:** MCQA (MMLU/GPQA/HLE style).
    - **Math:** Math with Judge (OpenMathReasoning dataset).
    - **Instruction following:** IFEval/IFBench style + Structured Outputs.
  - Each resource server ships with datasets, configs, tests, and a README.
  - Integrates with NeMo RL and other training frameworks.
  - Responses API-based agent architecture.
  - Early development — APIs evolving.
- **Docs:** [docs.nvidia.com/nemo-oss/gym](https://docs.nvidia.com/nemo-oss/gym/latest/index.html)

---

## 6. Evaluation Stage

### 6a. NeMo Evaluator

- **Repo:** [NVIDIA-NeMo/Evaluator](https://github.com/NVIDIA-NeMo/Evaluator) — 195 stars
- **What it does:** Open-source platform for robust, reproducible, and scalable LLM evaluation across 100+ benchmarks.
- **Highlights for writers:**
  - **Two components:**
    - `nemo-evaluator` — core engine that manages harness ↔ model interaction.
    - `nemo-evaluator-launcher` — CLI/orchestration layer that handles config, environment selection, and container launch.
  - **18 evaluation harnesses** with pre-built NGC containers:
    - Language: lm-evaluation-harness (MMLU, GSM8K, ARC, BBH, IFEval, etc.), simple-evals (MATH-500, AIME, HumanEval).
    - Code: bigcode-evaluation-harness (MBPP, HumanEval+), livecodebench, compute-eval (CUDA), scicode.
    - Safety: garak (vulnerability testing), safety-harness (Aegis v2, WildGuard).
    - VLM: vlmevalkit (MMMU, ChartQA, MathVista, OCRBench).
    - Specialized: BFCL (function calling), MT-Bench, TAU2-Bench, RULER (long-context), CoDec (contamination detection), MTEB (embeddings).
  - Works with any **OpenAI-compatible endpoint** — hosted (build.nvidia.com, NIM) or self-hosted (vLLM, TRT-LLM).
  - **Reproducibility by default:** All configs, seeds, and software provenance captured automatically.
  - **Scale anywhere:** Local machine, SLURM, Lepton AI, cloud-native backends.
  - Install: `pip install nemo-evaluator-launcher`
- **Docs:** [docs.nvidia.com/nemo-oss/evaluator](https://docs.nvidia.com/nemo-oss/evaluator/latest/)

### 6b. NeMo Skills (Evaluation Side)

- Skills also provides evaluation pipelines across math (AIME 24/25, HMMT), code (SWE-bench, LiveCodeBench), science (HLE, SciCode, GPQA), instruction following (IFBench, IFEval), long-context (RULER), VLM (MMMU-Pro), and more.
- Easy to parallelize evaluations across SLURM jobs and self-host LLM judges.

---

## 7. Deployment Stage

### 7a. NeMo Export-Deploy

- **Repo:** [NVIDIA-NeMo/Export-Deploy](https://github.com/NVIDIA-NeMo/Export-Deploy) — 27 stars
- **What it does:** Export NeMo and HF models to optimized inference backends and deploy for efficient serving.
- **Highlights for writers:**
  - **Export targets:** TensorRT-LLM, vLLM, ONNX, TensorRT.
  - **Deployment options:** NVIDIA Triton Inference Server (PyTriton) and Ray Serve.
  - **Model support:** NeMo LLMs, NeMo Multimodal, Hugging Face models, NIM Embedding, NIM Reranking.
  - **Precision:** BF16, FP8, INT8 (PTQ, QAT), FP4 (coming soon).
  - **Multi-GPU / Multi-instance** deployment support.
  - Serves as the bridge from training to production inference.
  - Install: `pip install nemo-export-deploy` (lightweight) or use NeMo Framework container for full features.
- **Docs:** [docs.nvidia.com/nemo-oss/export-deploy](https://docs.nvidia.com/nemo-oss/export-deploy/latest/)
- **Container:** Included in [NGC NeMo Framework](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo)

### 7b. NeMo Guardrails

- **Repo:** [NVIDIA-NeMo/Guardrails](https://github.com/NVIDIA-NeMo/Guardrails) — 5,635 stars (highest in the org!)
- **What it does:** Open-source toolkit for adding programmable guardrails to LLM-based conversational applications.
- **Highlights for writers:**
  - **5 types of rails:** Input, Dialog, Retrieval, Execution, Output.
  - **Colang language** (Python-like DSL) for defining dialog flows and rails — two versions: 1.0 and 2.0.
  - **Built-in guardrails library:** Jailbreak detection, fact-checking, hallucination detection, content moderation (Aegis, ActiveFence), sensitive data masking.
  - **Use cases:** RAG fact-checking, domain assistants, LLM endpoint safety, LangChain integration.
  - Works with OpenAI, Llama, Falcon, Vicuna, Mosaic, and more.
  - CLI: `nemoguardrails chat`, `nemoguardrails server`, `nemoguardrails evaluate`.
  - OpenAI-compatible server endpoint at `/v1/chat/completions`.
  - Published in EMNLP 2023 — academic paper available.
  - Latest version: 0.20.0.
- **Docs:** [docs.nvidia.com/nemo-oss/guardrails](https://docs.nvidia.com/nemo-oss/guardrails)

---

## 8. Infrastructure & Tooling

### 8a. NeMo Run

- **Repo:** [NVIDIA-NeMo/Run](https://github.com/NVIDIA-NeMo/Run) — 216 stars
- **What it does:** Configure, launch, and manage ML experiments across computing environments.
- **Highlights for writers:**
  - **Three core responsibilities:** Configuration, Execution, Management.
  - **Pythonic:** Everything configured in Python — no need for multi-tool workflows.
  - **Executors:** LocalExecutor, SlurmExecutor, SkypilotExecutor — set up once, scale easily.
  - **Modular:** Decouple task from executor; reuse environment configs across tasks.
  - Built on Fiddle (Google), TorchX, Skypilot, XManager.
  - Pre-release — API subject to change before v1.0.
- **Docs:** [docs.nvidia.com/nemo-oss/run](https://docs.nvidia.com/nemo-oss/run/latest/)

### 8b. Nemotron (Models & Recipes)

- **Repo:** [NVIDIA-NeMo/Nemotron](https://github.com/NVIDIA-NeMo/Nemotron)
- **What it does:** Recipes and scripts for NVIDIA's Nemotron model family.
- Contains training recipes, configuration files, and documentation for reproducing Nemotron models.

---

## 9. How the Repos Connect — Interoperability Map

![NeMo Framework Interoperability Map](assets/diagram-13-interoperability-map.png)

| From | To | How |
|------|----|-----|
| **Curator** → Training | Curated datasets feed into AutoModel / Megatron-Bridge / RL |
| **Data Designer** → Training | Synthetic data generation → SFT / RLHF datasets |
| **Skills** → Training / Eval | SDG pipelines + evaluation benchmarks |
| **AutoModel** ↔ **HF Hub** | Day-0 support — no checkpoint conversion needed |
| **AutoModel** → **NeMo RL** | Checkpoints used directly as starting points for DPO/GRPO |
| **Megatron-Bridge** ↔ **HF Hub** | Bidirectional checkpoint conversion via `AutoBridge` |
| **Megatron-Bridge** → **NeMo RL** | Megatron-Core training backend for RL at scale |
| **NeMo Gym** → **NeMo RL** | RL environments provide training data + reward signals |
| **Training** → **Evaluator** | Evaluate trained models against 100+ benchmarks |
| **Training** → **Export-Deploy** | Export to TRT-LLM / vLLM / ONNX for production |
| **Export-Deploy** → **Guardrails** | Add safety rails on top of deployed models |
| **NeMo Run** → All | Experiment launcher for any library across local/SLURM/K8s |

---

## 10. NGC Containers Quick Reference

| Container | Key Libraries Included |
|-----------|----------------------|
| [NeMo Framework](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo) | Megatron-Bridge, Export-Deploy, Evaluator, Run |
| [NeMo AutoModel](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo-automodel) | AutoModel |
| [NeMo RL](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo-rl) | NeMo RL (+ vLLM, Megatron) |
| [NeMo Curator](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo-curator) | Curator + RAPIDS |

---

## 11. Key Themes for Technical Writers

1. **Two training paths story:** AutoModel (PyTorch-native, easy HF, up to ~1K GPUs) vs. Megatron-Bridge (Megatron-Core, 6D parallelism, 1K+ GPUs). Several repos (DFM, RL) support both backends — consistent messaging is important.
2. **Day-0 HF support:** A major selling point. Both AutoModel and Megatron-Bridge work with HF models — but the mechanism differs (DTensor native vs. `AutoBridge` conversion).
3. **YAML-driven recipes:** AutoModel, Megatron-Bridge, RL, and DFM all follow a YAML config + CLI override pattern. Docs should be consistent in how they describe this.
4. **`uv` as the package manager:** Several repos have adopted `uv` for reproducible environments. Writers should document `uv venv`, `uv sync`, `uv run` patterns.
5. **Scale spectrum:** The framework spans from `pip install` on a laptop to 10K+ GPU SLURM clusters. Docs should provide clear paths for each scale.
6. **Multi-repo cross-references:** Many workflows span 2+ repos (e.g., Curator → AutoModel → RL → Evaluator → Export-Deploy). Cross-linking and journey-based docs are critical.
7. **Active development:** Several repos (AutoModel, RL, Gym) are under heavy active development with weekly model additions. Docs need processes for rapid updates.

---

## 12. Suggested Discussion / Q&A Topics

- How should we handle cross-repo documentation? (Unified getting-started guide vs. per-repo docs)
- What is the versioning/release cadence across repos?
- Which repos are highest priority for documentation investment?
- How do we keep model support tables current?
- Should we have a shared glossary across all NeMo docs sites?
- Container documentation: one page per container or integrated into each repo's docs?

---

## Appendix: Repo Summary Table

| Repo | Stage | Stars | One-Liner | Docs |
|------|-------|-------|-----------|------|
| [Curator](https://github.com/NVIDIA-NeMo/Curator) | Data | 1,394 | GPU-accelerated data curation (text, image, video, audio) | [link](https://docs.nvidia.com/nemo-oss/curator/latest/) |
| [Data Designer](https://github.com/NVIDIA-NeMo/DataDesigner) | Data | 698 | Synthetic data generation from scratch or seed data | [link](https://nvidia-nemo.github.io/DataDesigner/latest/) |
| [Skills](https://github.com/NVIDIA-NeMo/Skills) | Data + Eval | 816 | SDG pipelines + evaluation for math, code, science | [link](https://nvidia-nemo.github.io/Skills/) |
| [AutoModel](https://github.com/NVIDIA-NeMo/Automodel) | Training | 288 | PyTorch DTensor-native training with HF support | [link](https://docs.nvidia.com/nemo-oss/automodel/latest/) |
| [Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) | Training | 423 | Megatron-Core training with bidirectional HF conversion | [link](https://docs.nvidia.com/nemo-oss/megatron-bridge/latest/) |
| [NeMo (Speech)](https://github.com/NVIDIA-NeMo/NeMo) | Training | — | Speech AI (ASR, TTS) on Megatron-Core | [link](https://docs.nvidia.com/nemo-framework/user-guide/latest/speech_ai/index.html) |
| [DFM](https://github.com/NVIDIA-NeMo/DFM) | Training | 29 | Diffusion model training (video, image) | [link](https://github.com/NVIDIA-NeMo/DFM/tree/main/docs) |
| [Emerging-Optimizers](https://github.com/NVIDIA-NeMo/Emerging-Optimizers) | Training | — | Collection of cutting-edge optimizers | [link](https://docs.nvidia.com/nemo-oss/emerging-optimizers/latest/) |
| [NeMo RL](https://github.com/NVIDIA-NeMo/RL) | Alignment | 1,306 | Scalable post-training (GRPO, DPO, SFT, distillation) | [link](https://docs.nvidia.com/nemo-oss/rl/latest/) |
| [Gym](https://github.com/NVIDIA-NeMo/Gym) | Alignment | 637 | RL environments for LLM training | [link](https://docs.nvidia.com/nemo-oss/gym/latest/) |
| [Evaluator](https://github.com/NVIDIA-NeMo/Evaluator) | Evaluation | 195 | 100+ benchmarks across 18 harnesses | [link](https://docs.nvidia.com/nemo-oss/evaluator/latest/) |
| [Export-Deploy](https://github.com/NVIDIA-NeMo/Export-Deploy) | Deployment | 27 | Export to TRT-LLM/vLLM/ONNX + Triton serving | [link](https://docs.nvidia.com/nemo-oss/export-deploy/latest/) |
| [Guardrails](https://github.com/NVIDIA-NeMo/Guardrails) | Deployment | 5,635 | Programmable safety rails with Colang DSL | [link](https://docs.nvidia.com/nemo-oss/guardrails) |
| [Run](https://github.com/NVIDIA-NeMo/Run) | Infra | 216 | Experiment launcher (local, SLURM, K8s) | [link](https://docs.nvidia.com/nemo-oss/run/latest/) |
| [Nemotron](https://github.com/NVIDIA-NeMo/Nemotron) | Recipes | — | Nemotron model family recipes | [link](https://github.com/NVIDIA-NeMo/Nemotron#readme) |
