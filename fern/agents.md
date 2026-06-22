# agents.md

This file is for agents **developing** the NeMo Open Source Software documentation hub — the Fern site in this repository.

If you are helping a user **use** a NeMo library (Curator, AutoModel, RL, and so on), send them to that library's docs or the published hub at [docs.nvidia.com/nemo/oss](https://docs.nvidia.com/nemo/oss). Do not treat this repo as the place to document library APIs, install commands, or version-specific behavior.

**NeMo Open Source Software hub** is the orientation layer for the [NVIDIA-NeMo](https://github.com/NVIDIA-NeMo) GitHub org. It helps readers choose a stack, stage, library, runtime path, or container — then routes them to the owning repo or docs surface.

## What This Hub Owns

Keep content here when it is stable and helps a reader choose:

- Framework vs Platform positioning and lifecycle stages.
- Task-to-library routing and runtime-path decisions.
- Catalog metadata from `components/repos.ts` and `components/containers.ts`.
- Framework container release metadata and cross-component known issues.
- Terminology, concepts, glossary entries, and curated external learning links.

## What Belongs Downstream

Send content to per-library Fern docs or repo READMEs when it changes quickly:

- Install commands, API usage, tutorials, and examples.
- Model-specific recipes, benchmarks, and support matrices.
- Version pins and library-only workarounds.

Rule of thumb: add one orienting paragraph and a link instead of copying a library team's material.

## Key Files

| Need | Path |
| --- | --- |
| Site config, nav, redirects | [docs.yml](./docs.yml) |
| Repo catalog data | [components/repos.ts](./components/repos.ts) |
| Container catalog data | [components/containers.ts](./components/containers.ts) |
| Catalog React components | [components/RepoCatalog.tsx](./components/RepoCatalog.tsx), [components/ContainerCatalog.tsx](./components/ContainerCatalog.tsx) |
| Taxonomy vocabulary | [TAXONOMY.md](./TAXONOMY.md) |
| GitHub topic guidance | [GH-TOPICS.MD](../GH-TOPICS.MD) |
| Org README (lifecycle columns) | [profile/README.md](../profile/README.md) |
| Maintainer README | [README.md](./README.md) |

When copy disagrees on taxonomy, [TAXONOMY.md](./TAXONOMY.md) wins.

## Common Edits

### Add or update a repo

Edit [components/repos.ts](./components/repos.ts):

- Set `stage` to the org README lifecycle column: `data`, `pretraining`, `rl`, `inference`, or `e2e`.
- Set `kind` to `library`, `integration`, `reference`, or `infrastructure`.
- Add durable `tags` for search facets.
- Add `docsUrl` and `containerUrl` only when stable public targets exist.

### Add or update a container

Edit [components/containers.ts](./components/containers.ts) and, when needed, [docs/pages/about/release-notes/known-issues.mdx](./docs/pages/about/release-notes/known-issues.mdx).

### Add a concept page

Add under [docs/pages/about/concepts/](./docs/pages/about/concepts/). Concept pages explain relationships and tradeoffs. Put short term definitions in [docs/pages/resources/glossary.mdx](./docs/pages/resources/glossary.mdx).

### Add navigation or redirects

Update [docs.yml](./docs.yml). Register new MDX pages there; do not rely on filesystem discovery alone.

## Content Invariants

- Spell out **NeMo Open Source Software** on first use in a page; **NeMo OSS** is acceptable afterward.
- Use **NeMo Speech** (not bare "NeMo") for the speech AI repo in user-facing copy.
- Custom React components must be imported in MDX (`import RepoCatalog from "@/components/RepoCatalog";`).
- Prefer linking to library docs over duplicating their commands or APIs.
- Keep the glossary lookup-oriented; avoid turning it into narrative documentation.

## Local Development

Prerequisites: Node.js 22+, `jq` (for the pinned Fern CLI version in the Makefile).

```bash
make docs-login    # first time only
make docs-check    # validate before opening a PR
make docs          # http://localhost:3000
```

Publishing uses the org secret `DOCS_FERN_TOKEN`. Maintainers can run `make docs-preview` locally or tag `docs/v*` and push to trigger [publish-fern-docs.yml](../.github/workflows/publish-fern-docs.yml).

## Published Targets

Configured in [docs.yml](./docs.yml):

- Preview: `nemo.docs.buildwithfern.com/oss`
- Production: `docs.nvidia.com/nemo/oss`

Theme assets (`global-theme: nvidia`) are owned by [fern-components](https://github.com/NVIDIA/fern-components). Keep logos, global CSS, and footer changes there.
