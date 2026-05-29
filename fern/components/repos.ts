/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Canonical list of NVIDIA-NeMo GitHub organization repositories.
 * https://github.com/orgs/NVIDIA-NeMo/repositories
 *
 * Taxonomy (three layers):
 * - `stage` — README lifecycle column (Data · Pretraining · RL · Inference · E2E). Drives catalog filters.
 * - `kind` — repo role (library · integration · reference · infrastructure). See fern/TAXONOMY.md.
 * - `tags` — Search facets (modality, technique, role). Use for cross-cutting discovery in the search box.
 *
 * GitHub topic strategy (see GH-TOPICS.MD) can map `stage-*` topics to these stages when applied on repos.
 */

/** NeMo Speech docs — use /latest/ when published; /nightly/ is current. */
export const NEMO_SPEECH_DOCS_URL = "https://docs.nvidia.com/nemo/speech/nightly/";

/** Lifecycle stage — matches profile/README.md "Libraries by stage" columns. */
export type RepoStage = "data" | "pretraining" | "rl" | "inference" | "e2e";

export type RepoStatus = "active" | "archived";

/** Catalog role — see fern/TAXONOMY.md */
export type RepoKind = "library" | "integration" | "reference" | "infrastructure";

export interface NemoRepo {
  /** GitHub repo name (e.g. Automodel) */
  name: string;
  description: string;
  /** Primary lifecycle stage (README columns) */
  stage: RepoStage;
  /** Catalog role when stage alone is misleading */
  kind: RepoKind;
  githubUrl: string;
  docsUrl?: string;
  containerUrl?: string;
  status?: RepoStatus;
  /** Search facets — modality, technique, or cross-cutting role (e.g. evaluation, deployment) */
  tags?: string[];
}

/** Filter tabs — same order and labels as the org README. */
export const REPO_STAGES: { id: RepoStage | "all"; label: string }[] = [
  { id: "all", label: "All" },
  { id: "data", label: "Data" },
  { id: "pretraining", label: "Pretraining" },
  { id: "rl", label: "RL" },
  { id: "inference", label: "Inference" },
  { id: "e2e", label: "E2E" },
];

export const REPO_KINDS: { id: RepoKind; label: string }[] = [
  { id: "library", label: "Library" },
  { id: "integration", label: "Integration" },
  { id: "reference", label: "Reference" },
  { id: "infrastructure", label: "Infrastructure" },
];

/** 22 open-source libraries in the NVIDIA-NeMo org (excludes the .github meta repo). */
export const NEMO_REPOS: NemoRepo[] = [
  // Data
  {
    name: "Curator",
    description: "Scalable data preprocessing and curation for text, image, video, and audio.",
    stage: "data",
    kind: "library",
    githubUrl: "https://github.com/NVIDIA-NeMo/Curator",
    docsUrl: "https://docs.nvidia.com/nemo/curator/latest/",
    containerUrl: "https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo-curator",
    tags: ["multimodal", "curation"],
  },
  {
    name: "DataDesigner",
    description: "Generate high-quality synthetic data from scratch or from seed data.",
    stage: "data",
    kind: "library",
    githubUrl: "https://github.com/NVIDIA-NeMo/DataDesigner",
    docsUrl: "https://nvidia-nemo.github.io/DataDesigner/latest/",
    tags: ["synthetic-data", "mcp"],
  },
  {
    name: "DataDesignerPlugins",
    description: "Plugins extending NeMo Data Designer workflows.",
    stage: "data",
    kind: "library",
    githubUrl: "https://github.com/NVIDIA-NeMo/DataDesignerPlugins",
    tags: ["synthetic-data", "plugins"],
  },
  {
    name: "Anonymizer",
    description: "Detect and protect PII through context-aware replacement and rewriting.",
    stage: "data",
    kind: "library",
    githubUrl: "https://github.com/NVIDIA-NeMo/Anonymizer",
    tags: ["pii", "privacy"],
  },
  {
    name: "Safe-Synthesizer",
    description: "Create private, safe versions of sensitive tabular datasets.",
    stage: "data",
    kind: "library",
    githubUrl: "https://github.com/NVIDIA-NeMo/Safe-Synthesizer",
    docsUrl:
      "https://docs.nvidia.com/nemo/microservices/latest/generate-private-synthetic-data/",
    tags: ["privacy", "tabular"],
  },
  {
    name: "SDG-PGMs",
    description: "Build probabilistic graphical models (PGMs) for synthetic data generation.",
    stage: "data",
    kind: "library",
    githubUrl: "https://github.com/NVIDIA-NeMo/SDG-PGMs",
    tags: ["synthetic-data", "pgm"],
  },
  // Pretraining
  {
    name: "Automodel",
    description: "PyTorch distributed training for LLMs/VLMs with day-0 Hugging Face support.",
    stage: "pretraining",
    kind: "library",
    githubUrl: "https://github.com/NVIDIA-NeMo/Automodel",
    docsUrl: "https://docs.nvidia.com/nemo/automodel/latest/",
    containerUrl: "https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo-automodel",
    tags: ["llm", "vlm", "huggingface", "pytorch"],
  },
  {
    name: "Megatron-Bridge",
    description: "Megatron-based training with bidirectional Hugging Face checkpoint conversion.",
    stage: "pretraining",
    kind: "library",
    githubUrl: "https://github.com/NVIDIA-NeMo/Megatron-Bridge",
    docsUrl: "https://docs.nvidia.com/nemo/megatron-bridge/latest/",
    containerUrl: "https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo",
    tags: ["llm", "vlm", "megatron"],
  },
  {
    name: "NeMo Speech",
    description: "Speech AI (ASR, TTS) training and inference — the NeMo GitHub repo.",
    stage: "pretraining",
    kind: "library",
    githubUrl: "https://github.com/NVIDIA-NeMo/NeMo",
    docsUrl: NEMO_SPEECH_DOCS_URL,
    tags: ["speech", "asr", "tts"],
  },
  {
    name: "Emerging-Optimizers",
    description: "Collection of cutting-edge optimizers for large-scale training.",
    stage: "pretraining",
    kind: "library",
    githubUrl: "https://github.com/NVIDIA-NeMo/Emerging-Optimizers",
    docsUrl: "https://docs.nvidia.com/nemo/emerging-optimizers/latest/index.html",
    tags: ["optimizers"],
  },
  {
    name: "DFM",
    description: "Large-scale diffusion model training and inference (archived).",
    stage: "pretraining",
    kind: "library",
    githubUrl: "https://github.com/NVIDIA-NeMo/DFM",
    docsUrl: "https://github.com/NVIDIA-NeMo/DFM/tree/main/docs",
    status: "archived",
    tags: ["diffusion"],
  },
  // RL
  {
    name: "RL",
    description: "Scalable post-training — SFT, DPO, GRPO, distillation, and reinforcement learning.",
    stage: "rl",
    kind: "library",
    githubUrl: "https://github.com/NVIDIA-NeMo/RL",
    docsUrl: "https://docs.nvidia.com/nemo/rl/latest/",
    containerUrl: "https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo-rl",
    tags: ["dpo", "grpo", "alignment", "agents"],
  },
  {
    name: "Gym",
    description: "RL environments and benchmarks to evaluate and improve models and agents.",
    stage: "rl",
    kind: "library",
    githubUrl: "https://github.com/NVIDIA-NeMo/Gym",
    docsUrl: "https://docs.nvidia.com/nemo/gym/latest/index.html",
    tags: ["environments", "agents"],
  },
  {
    name: "ProRL-Agent-Server",
    description: "Rollout-as-a-service for multi-turn agent RL (pairs with NeMo RL and Gym).",
    stage: "rl",
    kind: "library",
    githubUrl: "https://github.com/NVIDIA-NeMo/ProRL-Agent-Server",
    docsUrl: "https://github.com/NVIDIA-NeMo/ProRL-Agent-Server#readme",
    tags: ["agents", "rollout"],
  },
  // Inference
  {
    name: "Evaluator",
    description: "Scalable, reproducible evaluation across 100+ benchmarks and harnesses.",
    stage: "inference",
    kind: "library",
    githubUrl: "https://github.com/NVIDIA-NeMo/Evaluator",
    docsUrl: "https://docs.nvidia.com/nemo/evaluator/latest/",
    containerUrl: "https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo",
    tags: ["evaluation", "benchmarks"],
  },
  {
    name: "Export-Deploy",
    description: "Export NeMo and Hugging Face models to TRT-LLM, vLLM, ONNX, and serving stacks.",
    stage: "inference",
    kind: "library",
    githubUrl: "https://github.com/NVIDIA-NeMo/Export-Deploy",
    docsUrl: "https://docs.nvidia.com/nemo/export-deploy/latest/",
    containerUrl: "https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo",
    tags: ["deployment", "serving", "vllm"],
  },
  {
    name: "Guardrails",
    description: "Programmable guardrails for LLM-based conversational systems (Colang).",
    stage: "inference",
    kind: "library",
    githubUrl: "https://github.com/NVIDIA-NeMo/Guardrails",
    docsUrl: "https://docs.nvidia.com/nemo/guardrails/latest/",
    tags: ["safety", "agents"],
  },
  {
    name: "NeMo Platform",
    description:
      "CLI, SDK, and web UI to evaluate, harden, tune, and deploy production agents using NeMo libraries.",
    stage: "inference",
    kind: "integration",
    githubUrl: "https://github.com/NVIDIA-NeMo/nemo-platform",
    docsUrl: "https://nvidia-nemo.github.io/nemo-platform/main/",
    tags: ["agents", "platform", "deployment"],
  },
  // E2E
  {
    name: "Skills",
    description: "Reference pipelines for synthetic data generation and evaluation (math, code, science).",
    stage: "e2e",
    kind: "reference",
    githubUrl: "https://github.com/NVIDIA-NeMo/Skills",
    docsUrl: "https://nvidia-nemo.github.io/Skills/",
    tags: ["sdg", "evaluation", "pipelines"],
  },
  {
    name: "Nemotron",
    description: "Developer asset hub — recipes, cookbooks, datasets, and Nemotron reference examples.",
    stage: "e2e",
    kind: "reference",
    githubUrl: "https://github.com/NVIDIA-NeMo/Nemotron",
    docsUrl: "https://github.com/NVIDIA-NeMo/Nemotron#readme",
    tags: ["nemotron", "recipes"],
  },
  {
    name: "Run",
    description: "Configure, launch, and manage ML experiments (local, SLURM, Kubernetes).",
    stage: "e2e",
    kind: "library",
    githubUrl: "https://github.com/NVIDIA-NeMo/Run",
    docsUrl: "https://docs.nvidia.com/nemo/run/latest/",
    containerUrl: "https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo",
    tags: ["orchestration", "experiments"],
  },
  {
    name: "FW-CI-templates",
    description: "CI/CD workflow templates shared across NeMo open-source libraries.",
    stage: "e2e",
    kind: "infrastructure",
    githubUrl: "https://github.com/NVIDIA-NeMo/FW-CI-templates",
    tags: ["ci", "github-actions"],
  },
];

export function stageLabel(stage: RepoStage): string {
  return REPO_STAGES.find((s) => s.id === stage)?.label ?? stage;
}

export function kindLabel(kind: RepoKind): string {
  return REPO_KINDS.find((k) => k.id === kind)?.label ?? kind;
}
