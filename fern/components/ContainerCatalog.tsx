"use client";

/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
import { useMemo, useState } from "react";
import {
  CONTAINER_KINDS,
  CONTAINER_STAGES,
  FRAMEWORK_RECENT_RELEASES,
  NEMO_CONTAINERS,
  NGC_NEMO_TEAM_URL,
  SOFTWARE_VERSIONS_URL,
  containerKindLabel,
  containerStageLabels,
  type ContainerKind,
  type NemoContainer,
} from "./containers";
import type { RepoStage } from "./repos";

const ACCENT = "#76B900";
const BORDER = "var(--border-default, #dddddd)";
const MUTED = "var(--grayscale-a11, #666666)";

function matchesQuery(container: NemoContainer, query: string): boolean {
  const q = query.trim().toLowerCase();
  if (!q) return true;
  const haystack = [
    container.name,
    container.description,
    container.image,
    container.kind,
    containerKindLabel(container.kind),
    containerStageLabels(container.stages),
    ...(container.bundledLibraries ?? []),
    ...(container.tags ?? []),
  ]
    .join(" ")
    .toLowerCase();
  return haystack.includes(q);
}

function ContainerCard({ container }: { container: NemoContainer }) {
  const pullTag = container.latestTag ?? "latest";
  const pullCommand = `docker pull ${container.image}:${pullTag}`;

  return (
    <article
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
        <h3 style={{ margin: 0, fontSize: "1.05rem", lineHeight: 1.3 }}>{container.name}</h3>
        <span
          style={{
            fontSize: "0.7rem",
            fontWeight: 600,
            padding: "2px 8px",
            borderRadius: "4px",
            background: container.kind === "multi-library" ? "rgba(118, 185, 0, 0.12)" : "#f5f5f5",
            color: container.kind === "multi-library" ? "#3d5c00" : MUTED,
            whiteSpace: "nowrap",
          }}
        >
          {containerKindLabel(container.kind)}
        </span>
      </div>

      <p style={{ margin: 0, fontSize: "0.875rem", lineHeight: 1.45, color: MUTED, flex: 1 }}>
        {container.description}
      </p>

      <code
        style={{
          fontSize: "0.75rem",
          padding: "0.5rem 0.65rem",
          borderRadius: "6px",
          background: "#f8f8f8",
          border: `1px solid ${BORDER}`,
          wordBreak: "break-all",
        }}
      >
        {pullCommand}
      </code>

      {container.bundledLibraries?.length ? (
        <p style={{ margin: 0, fontSize: "0.8rem", color: MUTED }}>
          Bundled: {container.bundledLibraries.join(", ")}
        </p>
      ) : null}

      <div style={{ display: "flex", flexWrap: "wrap", gap: "0.35rem" }}>
        <span
          style={{
            fontSize: "0.7rem",
            fontWeight: 600,
            padding: "2px 8px",
            borderRadius: "4px",
            border: `1px solid ${BORDER}`,
            color: MUTED,
          }}
        >
          {containerStageLabels(container.stages)}
        </span>
        {(container.tags ?? []).slice(0, 3).map((tag) => (
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
        <a href={container.ngcUrl} style={{ color: ACCENT, fontWeight: 600 }}>
          NGC
        </a>
        {container.docsUrl ? (
          <a href={container.docsUrl} style={{ color: ACCENT, fontWeight: 600 }}>
            Documentation
          </a>
        ) : null}
        {container.kind === "multi-library" ? (
          <>
            <a href={SOFTWARE_VERSIONS_URL} style={{ color: ACCENT, fontWeight: 600 }}>
              Component versions
            </a>
            <a href="/about/release-notes/known-issues" style={{ color: ACCENT, fontWeight: 600 }}>
              Known issues
            </a>
          </>
        ) : null}
      </div>
    </article>
  );
}

export default function ContainerCatalog() {
  const [query, setQuery] = useState("");
  const [kind, setKind] = useState<ContainerKind | "all">("all");
  const [stage, setStage] = useState<RepoStage | "all">("all");

  const filtered = useMemo(() => {
    return NEMO_CONTAINERS.filter((container) => {
      if (kind !== "all" && container.kind !== kind) return false;
      if (stage !== "all" && !container.stages.includes(stage)) return false;
      return matchesQuery(container, query);
    }).sort((a, b) => {
      if (a.kind !== b.kind) return a.kind === "multi-library" ? -1 : 1;
      return a.name.localeCompare(b.name);
    });
  }, [query, kind, stage]);

  const counts = useMemo(() => {
    const byKind: Record<string, number> = { all: NEMO_CONTAINERS.length };
    for (const c of NEMO_CONTAINERS) {
      byKind[c.kind] = (byKind[c.kind] ?? 0) + 1;
    }
    return byKind;
  }, []);

  const showFrameworkReleases =
    kind === "all" || kind === "multi-library" || filtered.some((c) => c.kind === "multi-library");

  return (
    <div className="nemo-container-catalog" style={{ marginTop: "1rem" }}>
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
            Search containers
          </span>
          <input
            type="search"
            placeholder="Search by name, image, bundled library, or tag…"
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
          href={NGC_NEMO_TEAM_URL}
          style={{ fontSize: "0.875rem", color: ACCENT, fontWeight: 600, whiteSpace: "nowrap" }}
        >
          All on NGC →
        </a>
      </div>

      <div
        role="tablist"
        aria-label="Filter by container type"
        style={{ display: "flex", flexWrap: "wrap", gap: "0.5rem", marginBottom: "0.75rem" }}
      >
        {CONTAINER_KINDS.map(({ id, label }) => {
          const active = kind === id;
          const count = counts[id] ?? 0;
          return (
            <button
              key={id}
              type="button"
              role="tab"
              aria-selected={active}
              onClick={() => setKind(id)}
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

      <div
        role="tablist"
        aria-label="Filter by lifecycle stage"
        style={{ display: "flex", flexWrap: "wrap", gap: "0.5rem", marginBottom: "1.25rem" }}
      >
        {CONTAINER_STAGES.map(({ id, label }) => {
          const active = stage === id;
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
              {label}
            </button>
          );
        })}
      </div>

      <p style={{ fontSize: "0.875rem", color: MUTED, margin: "0 0 1rem" }}>
        Showing {filtered.length} of {NEMO_CONTAINERS.length} NeMo containers on NGC. The{" "}
        <strong>NeMo Framework</strong> image bundles multiple libraries; standalone images target a single workload.
      </p>

      {filtered.length === 0 ? (
        <p style={{ color: MUTED }}>No containers match your filters. Try another type, stage, or clear the search box.</p>
      ) : (
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fill, minmax(300px, 1fr))",
            gap: "1rem",
            marginBottom: "1.5rem",
          }}
        >
          {filtered.map((container) => (
            <ContainerCard key={container.image} container={container} />
          ))}
        </div>
      )}

      {showFrameworkReleases ? (
        <section>
          <h2 style={{ fontSize: "1.15rem", margin: "0 0 0.75rem" }}>Recent NeMo Framework releases</h2>
          <p style={{ fontSize: "0.875rem", color: MUTED, margin: "0 0 1rem" }}>
            Update <code>FRAMEWORK_RECENT_RELEASES</code> in <code>components/containers.ts</code> when a new{" "}
            <code>nvcr.io/nvidia/nemo</code> tag ships. Keep the last three entries.
          </p>
          <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
            {FRAMEWORK_RECENT_RELEASES.map((release) => (
              <div
                key={release.tag}
                style={{
                  border: `1px solid ${BORDER}`,
                  borderRadius: "8px",
                  padding: "1rem 1.125rem",
                }}
              >
                <h3 style={{ margin: "0 0 0.35rem", fontSize: "1rem" }}>
                  {release.tag}{" "}
                  <span style={{ fontWeight: 400, color: MUTED, fontSize: "0.85rem" }}>
                    — <code>nvcr.io/nvidia/nemo:{release.tag}</code>
                  </span>
                </h3>
                <p style={{ margin: 0, fontSize: "0.875rem", lineHeight: 1.45, color: MUTED }}>{release.summary}</p>
              </div>
            ))}
          </div>
        </section>
      ) : null}
    </div>
  );
}
