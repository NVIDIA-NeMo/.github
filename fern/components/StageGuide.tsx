"use client";

/**
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
import { NEMO_REPOS, type RepoStage } from "./repos";

const ACCENT = "#76B900";
const BORDER = "var(--border-default, #dddddd)";
const MUTED = "var(--grayscale-a11, #666666)";

export default function StageGuide({ stage }: { stage: RepoStage }) {
  const repos = NEMO_REPOS.filter((repo) => repo.stage === stage && repo.status !== "archived").sort((a, b) =>
    a.name.localeCompare(b.name),
  );

  if (repos.length === 0) {
    return <p style={{ color: MUTED }}>No active libraries in this stage.</p>;
  }

  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))",
        gap: "1rem",
        marginTop: "1rem",
      }}
    >
      {repos.map((repo) => {
        const primaryHref = repo.docsUrl ?? repo.githubUrl;
        return (
          <article
            key={repo.name}
            style={{
              border: `1px solid ${BORDER}`,
              borderRadius: "8px",
              padding: "1rem 1.125rem",
              display: "flex",
              flexDirection: "column",
              gap: "0.5rem",
              background: "var(--background-default, #fff)",
            }}
          >
            <h3 style={{ margin: 0, fontSize: "1rem" }}>
              <a href={primaryHref} style={{ color: "inherit", textDecoration: "none" }}>
                {repo.name}
              </a>
            </h3>
            <p style={{ margin: 0, fontSize: "0.875rem", lineHeight: 1.45, color: MUTED, flex: 1 }}>
              {repo.description}
            </p>
            <div style={{ display: "flex", flexWrap: "wrap", gap: "0.75rem", fontSize: "0.8rem" }}>
              {repo.docsUrl ? (
                <a href={repo.docsUrl} style={{ color: ACCENT, fontWeight: 600 }}>
                  Get started in docs
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
      })}
    </div>
  );
}
