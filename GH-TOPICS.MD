## GitHub Topics Strategy

### Topic Categories

| Category | Topics | Purpose |
|----------|--------|---------|
| **Pipeline Stage** | `stage-data`, `stage-training`, `stage-alignment`, `stage-evaluation`, `stage-deployment`, `stage-safety` | Filter by workflow stage |
| **Model Type** | `model-llm`, `model-vlm`, `model-speech`, `model-diffusion`, `model-omni` | Filter by model modality |
| **Training Method** | `method-pretraining`, `method-sft`, `method-lora`, `method-rl`, `method-dpo`, `method-grpo` | Filter by training technique |
| **Backend** | `backend-megatron`, `backend-pytorch`, `backend-vllm`, `backend-tensorrt` | Filter by infrastructure |
| **Meta** | `nvidia-nemo` | All repos in the framework |

<!-- ### Topic Assignments

| Repo | Pipeline Stage | Model Type | Training Method | Backend |
|------|----------------|------------|-----------------|---------|
| **Automodel** | `stage-training` | `model-llm`, `model-vlm`, `model-omni` | `method-pretraining`, `method-sft`, `method-lora` | `backend-pytorch` |
| **Megatron-Bridge** | `stage-training` | `model-llm`, `model-vlm` | `method-pretraining`, `method-sft`, `method-lora` | `backend-megatron` |
| **NeMo (Speech)** | `stage-training` | `model-speech` | `method-pretraining`, `method-sft` | `backend-megatron`, `backend-pytorch` |
| **RL** | `stage-alignment` | `model-llm`, `model-vlm` | `method-sft`, `method-rl`, `method-dpo`, `method-grpo` | `backend-megatron`, `backend-pytorch`, `backend-vllm` |
| **Gym** | `stage-alignment` | `model-llm`, `model-vlm` | `method-rl` | — |
| **DFM** | `stage-training` | `model-diffusion` | `method-pretraining` | `backend-megatron`, `backend-pytorch` |
| **Curator** | `stage-data` | — | — | — |
| **DataDesigner** | `stage-data` | — | — | — |
| **Skills** | `stage-data`, `stage-evaluation` | `model-llm` | — | — |
| **Evaluator** | `stage-evaluation` | — | — | — |
| **Export-Deploy** | `stage-deployment` | — | — | `backend-vllm`, `backend-tensorrt` |
| **Guardrails** | `stage-safety` | `model-llm` | — | — |
| **Run** | `stage-training` | — | — | — |
| **Emerging-Optimizers** | `stage-training` | — | — | — |
| **Nemotron** | — | `model-llm` | — | — |

*All repos also get `nvidia-nemo` for org-wide filtering.* -->

### Example Filter Links

Once topics are applied, we you add convenience links to your project's README and docs:

```markdown
**Browse by pipeline stage:**
[Data](https://github.com/orgs/NVIDIA-NeMo/repositories?q=topic:stage-data) ·
[Training](https://github.com/orgs/NVIDIA-NeMo/repositories?q=topic:stage-training) ·
[Alignment](https://github.com/orgs/NVIDIA-NeMo/repositories?q=topic:stage-alignment) ·
[Evaluation](https://github.com/orgs/NVIDIA-NeMo/repositories?q=topic:stage-evaluation) ·
[Deployment](https://github.com/orgs/NVIDIA-NeMo/repositories?q=topic:stage-deployment)

**Browse by model type:**
[LLM](https://github.com/orgs/NVIDIA-NeMo/repositories?q=topic:model-llm) ·
[VLM](https://github.com/orgs/NVIDIA-NeMo/repositories?q=topic:model-vlm) ·
[Speech](https://github.com/orgs/NVIDIA-NeMo/repositories?q=topic:model-speech) ·
[Diffusion](https://github.com/orgs/NVIDIA-NeMo/repositories?q=topic:model-diffusion)
```