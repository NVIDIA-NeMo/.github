/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Canonical list of NVIDIA-NeMo GitHub organization repositories.
 * https://github.com/orgs/NVIDIA-NeMo/repositories
 */

export type RepoCategory =
  | "data"
  | "training"
  | "alignment"
  | "evaluation"
  | "deployment"
  | "infrastructure";

export type RepoStatus = "active" | "archived";

export interface NemoRepo {
  /** GitHub repo name (e.g. Automodel) */
  name: string;
  description: string;
  category: RepoCategory;
  githubUrl: string;
  docsUrl?: string;
  containerUrl?: string;
  status?: RepoStatus;
  /** Extra facets for search (e.g. speech, agents) */
  tags?: string[];
}

export const REPO_CATEGORIES: { id: RepoCategory | "all"; label: string }[] = [
  { id: "all", label: "All" },
  { id: "data", label: "Data" },
  { id: "training", label: "Training" },
  { id: "alignment", label: "Alignment & agents" },
  { id: "evaluation", label: "Evaluation" },
  { id: "deployment", label: "Deployment & safety" },
  { id: "infrastructure", label: "Infrastructure" },
];

/** 23 repositories as listed on the org page (including .github). */
export const NEMO_REPOS: NemoRepo[] = [
  // Data
  {
    name: "Curator",
    description: "Scalable data preprocessing and curation for text, image, video, and audio.",
    category: "data",
    githubUrl: "https://github.com/NVIDIA-NeMo/Curator",
    docsUrl: "https://docs.nvidia.com/nemo/curator/latest/",
    containerUrl: "https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo-curator",
    tags: ["multimodal", "deduplication"],
  },
  {
    name: "DataDesigner",
    description: "Generate high-quality synthetic data from scratch or from seed data.",
    category: "data",
    githubUrl: "https://github.com/NVIDIA-NeMo/DataDesigner",
    docsUrl: "https://nvidia-nemo.github.io/DataDesigner/latest/",
    tags: ["synthetic-data", "mcp"],
  },
  {
    name: "DataDesignerPlugins",
    description: "Plugins extending NeMo Data Designer workflows.",
    category: "data",
    githubUrl: "https://github.com/NVIDIA-NeMo/DataDesignerPlugins",
    tags: ["synthetic-data", "plugins"],
  },
  {
    name: "Skills",
    description: "Reference pipelines for synthetic data generation and evaluation (math, code, science).",
    category: "data",
    githubUrl: "https://github.com/NVIDIA-NeMo/Skills",
    docsUrl: "https://nvidia-nemo.github.io/Skills/",
    tags: ["evaluation", "sdg"],
  },
  {
    name: "Safe-Synthesizer",
    description: "Create private, safe versions of sensitive tabular datasets.",
    category: "data",
    githubUrl: "https://github.com/NVIDIA-NeMo/Safe-Synthesizer",
    docsUrl:
      "https://docs.nvidia.com/nemo/microservices/latest/generate-private-synthetic-data/",
    tags: ["privacy", "tabular"],
  },
  {
    name: "Anonymizer",
    description: "Detect and protect PII through context-aware replacement and rewriting.",
    category: "data",
    githubUrl: "https://github.com/NVIDIA-NeMo/Anonymizer",
    tags: ["pii", "privacy"],
  },
  {
    name: "SDG-PGMs",
    description: "Build probabilistic graphical models (PGMs) for synthetic data generation.",
    category: "data",
    githubUrl: "https://github.com/NVIDIA-NeMo/SDG-PGMs",
    tags: ["synthetic-data", "pgm"],
  },
  // Training
  {
    name: "Automodel",
    description: "PyTorch distributed training for LLMs/VLMs with day-0 Hugging Face support.",
    category: "training",
    githubUrl: "https://github.com/NVIDIA-NeMo/Automodel",
    docsUrl: "https://docs.nvidia.com/nemo/automodel/latest/",
    containerUrl: "https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo-automodel",
    tags: ["llm", "vlm", "huggingface"],
  },
  {
    name: "Megatron-Bridge",
    description: "Megatron-based training with bidirectional Hugging Face checkpoint conversion.",
    category: "training",
    githubUrl: "https://github.com/NVIDIA-NeMo/Megatron-Bridge",
    docsUrl: "https://docs.nvidia.com/nemo/megatron-bridge/latest/",
    containerUrl: "https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo",
    tags: ["llm", "vlm", "megatron"],
  },
  {
    name: "NeMo",
    description: "Speech AI (ASR, TTS) and legacy NeMo toolkit; org focus shifting to modular libs.",
    category: "training",
    githubUrl: "https://github.com/NVIDIA-NeMo/NeMo",
    docsUrl: "https://docs.nvidia.com/nemo-framework/user-guide/latest/speech_ai/index.html",
    tags: ["speech", "asr", "tts"],
  },
  {
    name: "Nemotron",
    description: "Developer asset hub — recipes, cookbooks, datasets, and Nemotron reference examples.",
    category: "training",
    githubUrl: "https://github.com/NVIDIA-NeMo/Nemotron",
    docsUrl: "https://github.com/NVIDIA-NeMo/Nemotron#readme",
    tags: ["nemotron", "recipes"],
  },
  {
    name: "Emerging-Optimizers",
    description: "Collection of cutting-edge optimizers for large-scale training.",
    category: "training",
    githubUrl: "https://github.com/NVIDIA-NeMo/Emerging-Optimizers",
    docsUrl: "https://docs.nvidia.com/nemo/emerging-optimizers/latest/index.html",
  },
  {
    name: "DFM",
    description: "Large-scale diffusion model training and inference (archived).",
    category: "training",
    githubUrl: "https://github.com/NVIDIA-NeMo/DFM",
    docsUrl: "https://github.com/NVIDIA-NeMo/DFM/tree/main/docs",
    status: "archived",
    tags: ["diffusion"],
  },
  // Alignment & agents
  {
    name: "RL",
    description: "Scalable post-training — SFT, DPO, GRPO, distillation, and reinforcement learning.",
    category: "alignment",
    githubUrl: "https://github.com/NVIDIA-NeMo/RL",
    docsUrl: "https://docs.nvidia.com/nemo/rl/latest/",
    containerUrl: "https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo-rl",
    tags: ["dpo", "grpo", "rlhf"],
  },
  {
    name: "Gym",
    description: "RL environments and benchmarks to evaluate and improve models and agents.",
    category: "alignment",
    githubUrl: "https://github.com/NVIDIA-NeMo/Gym",
    docsUrl: "https://docs.nvidia.com/nemo/gym/latest/index.html",
    tags: ["environments", "agents"],
  },
  {
    name: "ProRL-Agent-Server",
    description: "Rollout-as-a-service for multi-turn agent RL (pairs with NeMo RL and Gym).",
    category: "alignment",
    githubUrl: "https://github.com/NVIDIA-NeMo/ProRL-Agent-Server",
    docsUrl: "https://github.com/NVIDIA-NeMo/ProRL-Agent-Server#readme",
    tags: ["agents", "rollout", "openhands"],
  },
  // Evaluation
  {
    name: "Evaluator",
    description: "Scalable, reproducible evaluation across 100+ benchmarks and harnesses.",
    category: "evaluation",
    githubUrl: "https://github.com/NVIDIA-NeMo/Evaluator",
    docsUrl: "https://docs.nvidia.com/nemo/evaluator/latest/",
    containerUrl: "https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo",
    tags: ["benchmarks"],
  },
  // Deployment & safety
  {
    name: "Export-Deploy",
    description: "Export NeMo and Hugging Face models to TRT-LLM, vLLM, ONNX, and serving stacks.",
    category: "deployment",
    githubUrl: "https://github.com/NVIDIA-NeMo/Export-Deploy",
    docsUrl: "https://docs.nvidia.com/nemo/export-deploy/latest/",
    containerUrl: "https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo",
    tags: ["inference", "triton"],
  },
  {
    name: "Guardrails",
    description: "Programmable guardrails for LLM-based conversational systems (Colang).",
    category: "deployment",
    githubUrl: "https://github.com/NVIDIA-NeMo/Guardrails",
    docsUrl: "https://docs.nvidia.com/nemo/guardrails/latest/",
    tags: ["safety", "agents"],
  },
  {
    name: "nemo-platform",
    description: "Platform to ship agents that are faster, more accurate, and safer.",
    category: "deployment",
    githubUrl: "https://github.com/NVIDIA-NeMo/nemo-platform",
    tags: ["agents", "platform"],
  },
  // Infrastructure
  {
    name: "Run",
    description: "Configure, launch, and manage ML experiments (local, SLURM, Kubernetes).",
    category: "infrastructure",
    githubUrl: "https://github.com/NVIDIA-NeMo/Run",
    docsUrl: "https://docs.nvidia.com/nemo/run/latest/",
    containerUrl: "https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo",
    tags: ["experiments", "orchestration"],
  },
  {
    name: "FW-CI-templates",
    description: "CI/CD workflow templates shared across NeMo Framework libraries.",
    category: "infrastructure",
    githubUrl: "https://github.com/NVIDIA-NeMo/FW-CI-templates",
    tags: ["ci", "github-actions"],
  },
  {
    name: ".github",
    description: "Organization profile, this documentation hub, and shared org settings.",
    category: "infrastructure",
    githubUrl: "https://github.com/NVIDIA-NeMo/.github",
    docsUrl: "https://docs.nvidia.com/nemo",
    tags: ["org", "hub"],
  },
];

export function categoryLabel(category: RepoCategory): string {
  return REPO_CATEGORIES.find((c) => c.id === category)?.label ?? category;
}
