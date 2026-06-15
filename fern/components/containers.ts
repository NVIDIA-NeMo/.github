/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Canonical NGC containers for NeMo Open Source Software — distinct from the 22-repo library catalog.
 * Update `latestTag` and `FRAMEWORK_RECENT_RELEASES` when a new Framework container ships.
 */

import type { RepoStage } from "./repos";

export type ContainerKind = "multi-library" | "standalone";

export interface NemoContainer {
  /** Display name on NGC */
  name: string;
  /** Full image without tag, e.g. nvcr.io/nvidia/nemo */
  image: string;
  ngcUrl: string;
  description: string;
  kind: ContainerKind;
  /** Primary lifecycle stages this container supports */
  stages: RepoStage[];
  latestTag?: string;
  docsUrl?: string;
  /** Bundled libraries (multi-library Framework container only) */
  bundledLibraries?: string[];
  tags?: string[];
}

export const CONTAINER_KINDS: { id: ContainerKind | "all"; label: string }[] = [
  { id: "all", label: "All" },
  { id: "multi-library", label: "Multi-library" },
  { id: "standalone", label: "Standalone" },
];

export const CONTAINER_STAGES: { id: RepoStage | "all"; label: string }[] = [
  { id: "all", label: "All stages" },
  { id: "data", label: "Data" },
  { id: "pretraining", label: "Pretraining" },
  { id: "rl", label: "RL" },
  { id: "inference", label: "Inference" },
  { id: "e2e", label: "E2E" },
];

/** Recent NeMo Framework container tags — keep last three. */
export interface FrameworkRelease {
  tag: string;
  summary: string;
}

export const FRAMEWORK_RECENT_RELEASES: FrameworkRelease[] = [
  {
    tag: "26.02",
    summary:
      "Updated Megatron-Bridge, Evaluator, Export-Deploy, Run, and Speech. See Megatron-Bridge software component versions for pinned packages.",
  },
  {
    tag: "25.11.01",
    summary: "Patch release. See Megatron-Bridge and Export-Deploy release notes.",
  },
  {
    tag: "25.11",
    summary:
      "Updates across Megatron-Bridge, Evaluator, Export-Deploy, Run, and Speech. Evaluator adds direct nemo-evaluator use, Logprob benchmarks, and multi-node Ray evaluations via NeMo Run.",
  },
];

export const SOFTWARE_VERSIONS_URL =
  "https://docs.nvidia.com/nemo/megatron-bridge/latest/releases/software-versions.html";

export const NGC_NEMO_TEAM_URL =
  "https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/containers";

/** Published NeMo Open Source Software containers on NGC (standalone + multi-library stack). */
export const NEMO_CONTAINERS: NemoContainer[] = [
  {
    name: "NeMo Framework",
    image: "nvcr.io/nvidia/nemo",
    ngcUrl: "https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo",
    description:
      "Primary multi-library training stack — Megatron-Bridge, Evaluator, Export-Deploy, Run, and Speech in one image.",
    kind: "multi-library",
    stages: ["pretraining", "rl", "inference", "e2e"],
    latestTag: "26.02",
    docsUrl: "https://docs.nvidia.com/nemo/megatron-bridge/latest/",
    bundledLibraries: ["Megatron-Bridge", "Evaluator", "Export-Deploy", "Run", "NeMo Speech"],
    tags: ["llm", "vlm", "speech", "megatron"],
  },
  {
    name: "NeMo AutoModel",
    image: "nvcr.io/nvidia/nemo-automodel",
    ngcUrl: "https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo-automodel",
    description: "PyTorch-native distributed training for LLMs and VLMs with Hugging Face support.",
    kind: "standalone",
    stages: ["pretraining"],
    docsUrl: "https://docs.nvidia.com/nemo/automodel/latest/",
    tags: ["llm", "vlm", "huggingface", "pytorch"],
  },
  {
    name: "NeMo RL",
    image: "nvcr.io/nvidia/nemo-rl",
    ngcUrl: "https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo-rl",
    description: "Alignment and reinforcement learning — SFT, DPO, GRPO, and distillation.",
    kind: "standalone",
    stages: ["rl"],
    docsUrl: "https://docs.nvidia.com/nemo/rl/latest/",
    tags: ["dpo", "grpo", "alignment"],
  },
  {
    name: "NeMo Curator",
    image: "nvcr.io/nvidia/nemo-curator",
    ngcUrl: "https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo-curator",
    description: "Data preprocessing and curation for text, image, video, and audio at scale.",
    kind: "standalone",
    stages: ["data"],
    docsUrl: "https://docs.nvidia.com/nemo/curator/latest/",
    tags: ["curation", "multimodal"],
  },
];

export function containerKindLabel(kind: ContainerKind): string {
  return kind === "multi-library" ? "Multi-library" : "Standalone";
}

export function containerStageLabels(stages: RepoStage[]): string {
  return stages.map((s) => CONTAINER_STAGES.find((x) => x.id === s)?.label ?? s).join(" · ");
}
