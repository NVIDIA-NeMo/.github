"use client";

/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
import { useMemo, useState } from "react";
import {
  NEMO_REPOS,
  REPO_STAGES,
  stageLabel,
  type NemoRepo,
  type RepoStage,
} from "./repos";

const ACCENT = "#76B900";
const BORDER = "var(--border-default, #dddddd)";
const MUTED = "var(--grayscale-a11, #666666)";

function matchesQuery(repo: NemoRepo, query: string): boolean {
  const q = query.trim().toLowerCase();
  if (!q) return true;
  const haystack = [
    repo.name,
    repo.description,
    repo.stage,
    stageLabel(repo.stage),
    ...(repo.tags ?? []),
  ]
    .join(" ")
    .toLowerCase();
  return haystack.includes(q);
}

function RepoCard({ repo }: { repo: NemoRepo }) {
  const primaryHref = repo.docsUrl ?? repo.githubUrl;
  return (
    <article
      className="nemo-repo-card"
      style={{
        border: `1px solid ${BORDER}`,
        borderRadius: "8px",
        padding: "1rem 1.125rem",
        display: "flex",
        flexDirection: "column",
        gap: "0.5rem",
        background: "var(--background-default, #fff)",
        minHeight: "100%",
      }}
    >
      <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: "0.5rem" }}>
        <h3 style={{ margin: 0, fontSize: "1.05rem", lineHeight: 1.3 }}>
          <a href={primaryHref} style={{ color: "inherit", textDecoration: "none" }}>
            {repo.name}
          </a>
        </h3>
        {repo.status === "archived" ? (
          <span
            style={{
              fontSize: "0.7rem",
              fontWeight: 600,
              padding: "2px 8px",
              borderRadius: "4px",
              background: "#f0f0f0",
              color: MUTED,
              whiteSpace: "nowrap",
            }}
          >
            Archived
          </span>
        ) : null}
      </div>
      <p style={{ margin: 0, fontSize: "0.875rem", lineHeight: 1.45, color: MUTED, flex: 1 }}>
        {repo.description}
      </p>
      <div style={{ display: "flex", flexWrap: "wrap", gap: "0.35rem", alignItems: "center" }}>
        <span
          style={{
            fontSize: "0.7rem",
            fontWeight: 600,
            padding: "2px 8px",
            borderRadius: "4px",
            background: "rgba(118, 185, 0, 0.12)",
            color: "#3d5c00",
          }}
        >
          {stageLabel(repo.stage)}
        </span>
        {(repo.tags ?? []).slice(0, 3).map((tag) => (
          <span
            key={tag}
            style={{
              fontSize: "0.65rem",
              padding: "2px 6px",
              borderRadius: "4px",
              border: `1px solid ${BORDER}`,
              color: MUTED,
            }}
          >
            {tag}
          </span>
        ))}
      </div>
      <div style={{ display: "flex", flexWrap: "wrap", gap: "0.75rem", fontSize: "0.8rem", marginTop: "0.25rem" }}>
        {repo.docsUrl ? (
          <a href={repo.docsUrl} style={{ color: ACCENT, fontWeight: 600 }}>
            Documentation
          </a>
        ) : null}
        <a href={repo.githubUrl} style={{ color: ACCENT, fontWeight: 600 }}>
          GitHub
        </a>
        {repo.containerUrl ? (
          <a href={repo.containerUrl} style={{ color: ACCENT, fontWeight: 600 }}>
            NGC container
          </a>
        ) : null}
      </div>
    </article>
  );
}

export default function RepoCatalog() {
  const [query, setQuery] = useState("");
  const [stage, setStage] = useState<RepoStage | "all">("all");

  const filtered = useMemo(() => {
    return NEMO_REPOS.filter((repo) => {
      if (stage !== "all" && repo.stage !== stage) return false;
      return matchesQuery(repo, query);
    }).sort((a, b) => a.name.localeCompare(b.name));
  }, [query, stage]);

  const counts = useMemo(() => {
    const byStage: Record<string, number> = { all: NEMO_REPOS.length };
    for (const repo of NEMO_REPOS) {
      byStage[repo.stage] = (byStage[repo.stage] ?? 0) + 1;
    }
    return byStage;
  }, []);

  return (
    <div className="nemo-repo-catalog" style={{ marginTop: "1rem" }}>
      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          gap: "0.75rem",
          alignItems: "center",
          marginBottom: "1rem",
        }}
      >
        <label style={{ flex: "1 1 220px", minWidth: "200px" }}>
          <span className="sr-only" style={{ position: "absolute", width: 1, height: 1, overflow: "hidden" }}>
            Search libraries
          </span>
          <input
            type="search"
            placeholder="Search by name, description, or tag…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            style={{
              width: "100%",
              padding: "0.5rem 0.75rem",
              fontSize: "0.95rem",
              border: `1px solid ${BORDER}`,
              borderRadius: "6px",
            }}
          />
        </label>
        <a
          href="https://github.com/orgs/NVIDIA-NeMo/repositories"
          style={{ fontSize: "0.875rem", color: ACCENT, fontWeight: 600, whiteSpace: "nowrap" }}
        >
          View on GitHub →
        </a>
      </div>

      <div
        role="tablist"
        aria-label="Filter by lifecycle stage"
        style={{
          display: "flex",
          flexWrap: "wrap",
          gap: "0.5rem",
          marginBottom: "1.25rem",
        }}
      >
        {REPO_STAGES.map(({ id, label }) => {
          const active = stage === id;
          const count = counts[id] ?? 0;
          return (
            <button
              key={id}
              type="button"
              role="tab"
              aria-selected={active}
              onClick={() => setStage(id)}
              style={{
                padding: "0.35rem 0.85rem",
                fontSize: "0.8rem",
                fontWeight: active ? 700 : 500,
                borderRadius: "999px",
                border: active ? `2px solid ${ACCENT}` : `1px solid ${BORDER}`,
                background: active ? "rgba(118, 185, 0, 0.1)" : "transparent",
                cursor: "pointer",
                color: "inherit",
              }}
            >
              {label} ({count})
            </button>
          );
        })}
      </div>

      <p style={{ fontSize: "0.875rem", color: MUTED, margin: "0 0 1rem" }}>
        Showing {filtered.length} of {NEMO_REPOS.length} libraries in{" "}
        <a href="https://github.com/orgs/NVIDIA-NeMo/repositories?type=all">NVIDIA-NeMo</a>. Stages match the{" "}
        <a href="https://github.com/NVIDIA-NeMo">org README</a>; tags add cross-cutting search facets.
      </p>

      {filtered.length === 0 ? (
        <p style={{ color: MUTED }}>No libraries match your search. Try another stage or clear the search box.</p>
      ) : (
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))",
            gap: "1rem",
          }}
        >
          {filtered.map((repo) => (
            <RepoCard key={repo.name} repo={repo} />
          ))}
        </div>
      )}
    </div>
  );
}
