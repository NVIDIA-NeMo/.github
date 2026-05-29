# NeMo OSS taxonomy

Canonical vocabulary for the Fern hub (`docs.nvidia.com/nemo`), org README, and `components/repos.ts`. When copy disagrees, this file wins.

## Top-level map

```
NVIDIA NeMo (commercial suite — OSS + microservices + NIM + services)
└── NeMo OSS (GitHub org NVIDIA-NeMo + this hub)
    ├── NeMo Framework — model lifecycle (libraries + optional NGC bundle)
    ├── NeMo Platform — agent integration product (CLI, SDK, Studio)
    └── 22 catalog repos (libraries, integration, reference, infrastructure)
```

## Terms

| Term | Meaning |
| --- | --- |
| **NVIDIA NeMo** | Full software suite. Includes commercial products not listed on this hub. |
| **NeMo OSS** | Public open source in [NVIDIA-NeMo](https://github.com/NVIDIA-NeMo) and documentation on **docs.nvidia.com/nemo**. Discovery layer — not a single product. |
| **NeMo Framework** | Named **model-lifecycle** stack: composable libraries from data through deployment. **Not one codebase.** |
| **NeMo Framework container** | NGC image `nvcr.io/nvidia/nemo:<tag>`. Bundles Megatron-Bridge, Evaluator, Export-Deploy, Run, and NeMo Speech. |
| **NeMo Platform** | [nemo-platform](https://github.com/NVIDIA-NeMo/nemo-platform) — CLI, SDK, and Studio for **agent** evaluate / secure / tune / deploy. Composes libraries; not a pipeline stage. |
| **Library** | A focused repo with its own docs and release cadence (Curator, AutoModel, RL, …). |
| **NeMo Speech** | The [NeMo](https://github.com/NVIDIA-NeMo/NeMo) repo — speech AI only. Do not use “NeMo” alone for the whole ecosystem. |

## Repo `kind` (catalog metadata)

| Kind | Role | Examples |
| --- | --- | --- |
| `library` | Default product repo for a lifecycle stage | Curator, AutoModel, RL, Evaluator |
| `integration` | Composes multiple libraries into one product surface | NeMo Platform |
| `reference` | Recipes, cookbooks, reference pipelines | Skills, Nemotron |
| `infrastructure` | Shared CI or meta repos | FW-CI-templates |

Every repo still has one **stage** (org README columns). **Kind** clarifies role when stage alone is misleading (for example Platform listed under Inference for discoverability).

## Lifecycle stages

Data · Pretraining · RL · Inference · E2E — same columns as [profile/README.md](../profile/README.md).

## Hub page roles

| Page | Job |
| --- | --- |
| **Concepts** | Glossary |
| **Ecosystem** | Positioning and choices (Framework vs Platform, commercial boundary) |
| **Architecture** | Structure — pipeline, backends, containers, Platform overlay |
| **Libraries** | Inventory from `repos.ts` |

## Out of scope for this hub

Customizer, NIM, and other commercial NeMo microservices — link from Ecosystem only; do not duplicate product docs.
