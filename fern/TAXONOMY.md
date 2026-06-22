# NeMo Open Source Software taxonomy

Canonical vocabulary for the staged Fern hub (`docs.nvidia.com/nemo/oss`), org README, and `components/repos.ts`. When copy disagrees, this file wins.

## Top-level map

```
NVIDIA NeMo (commercial suite — open source + microservices + NIM + services)
└── NeMo Open Source Software (GitHub org NVIDIA-NeMo + this hub)
    ├── NeMo Framework — model lifecycle (libraries + optional NGC bundle)
    ├── NeMo Platform — agent integration product (CLI, SDK, Studio)
    └── 22 catalog repos (libraries, integration, reference, infrastructure)
```

## Terms

| Term | Meaning |
| --- | --- |
| **NVIDIA NeMo** | Full software suite spanning open source libraries, commercial products, NIM, microservices, and services. |
| **NeMo Open Source Software** | Public open source in [NVIDIA-NeMo](https://github.com/NVIDIA-NeMo), staged hub orientation, library documentation, and NGC containers. Entry point for choosing a stack, stage, library, or container. |
| **NeMo Framework** | Named **model-lifecycle** stack: composable libraries from data through deployment, each with its own source and docs. |
| **NeMo Framework container** | NGC image `nvcr.io/nvidia/nemo:<tag>`. Bundles Megatron-Bridge, Evaluator, Export-Deploy, Run, and NeMo Speech. |
| **NeMo Platform** | [nemo-platform](https://github.com/NVIDIA-NeMo/nemo-platform) — CLI, SDK, and Studio for **agent** evaluate / secure / tune / deploy. Composes libraries into an agent integration experience. |
| **Library** | A focused repo with its own docs and release cadence (Curator, AutoModel, RL, …). |
| **NeMo Speech** | The [NeMo](https://github.com/NVIDIA-NeMo/NeMo) repo — use this wording for speech AI in user-facing copy. |

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
| **Concepts** | Core mental models and relationships. Use concept pages for explanatory topics, not term lookup. |
| **Ecosystem** | Positioning and choices (Framework vs Platform, commercial boundary) |
| **Architecture** | Structure — pipeline, backends, containers, Platform overlay |
| **Libraries** | Inventory from `repos.ts` |
| **Task map** | Task-first routing from user intent to library, runtime path, and owning docs |
| **Runtime chooser** | Setup-path decision guide for containers, pip/source installs, and Platform setup |
| **Glossary** | Lookup-oriented definitions for terms, acronyms, and product names |
| **External learning** | Curated third-party blogs, videos, and partner examples with freshness caveats |

## Concepts section

Concepts is a directory, not a glossary. Keep pages focused on stable relationships that help readers reason across repos:

| Concept page | Job |
| --- | --- |
| **Framework and Platform** | Distinguish model-lifecycle work from integrated agent workflows |
| **Lifecycle stages** | Explain Data, Pretraining, RL, Inference, and E2E as workflow stages |
| **Repository catalog model** | Explain stage, kind, and tags |
| **Training backends and checkpoints** | Explain AutoModel, Megatron-Bridge, and checkpoint flow at a decision level |
| **Containers and installs** | Explain Framework container, standalone containers, and library installs |
| **Documentation surfaces** | Explain what the hub, library docs, repos, release notes, and glossary own |

## Broader suite references

Customizer, NIM, and other commercial NeMo microservices have their own product documentation. Mention them only when needed to explain the broader suite; do not use commercial microservice docs as open source setup destinations.
