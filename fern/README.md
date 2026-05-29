# NeMo OSS Hub

Fern source for the **NeMo OSS** documentation hub: the lightweight entry point for open source repositories in [NVIDIA-NeMo](https://github.com/NVIDIA-NeMo).

The hub helps readers answer three questions:

- **What should I use?** Task map, lifecycle stages, Framework vs Platform, and the library catalog.
- **How should I run it?** Runtime chooser, install overview, container catalog, and release notes.
- **Where do I go next?** Per-library docs, GitHub repos, community links, glossary, and external learning resources.

Per-library docs own commands, APIs, tutorials, model support, and version-specific behavior. This hub owns orientation, routing, stable concepts, and cross-component release metadata.

## Quick Links

| Need | Source | Published Route |
| --- | --- | --- |
| Home | [docs/pages/index.mdx](./docs/pages/index.mdx) | `/` |
| Task-first routing | [docs/pages/get-started/task-map.mdx](./docs/pages/get-started/task-map.mdx) | `/get-started/task-map` |
| Runtime choice | [docs/pages/get-started/runtime-chooser.mdx](./docs/pages/get-started/runtime-chooser.mdx) | `/get-started/runtime-chooser` |
| Library catalog page | [docs/pages/about/libraries.mdx](./docs/pages/about/libraries.mdx) | `/about/libraries` |
| Library catalog data | [components/repos.ts](./components/repos.ts) | Catalog component source |
| Container catalog page | [docs/pages/about/release-notes/containers.mdx](./docs/pages/about/release-notes/containers.mdx) | `/about/release-notes/containers` |
| Container catalog data | [components/containers.ts](./components/containers.ts) | Catalog component source |
| Concepts | [docs/pages/about/concepts/index.mdx](./docs/pages/about/concepts/index.mdx) | `/about/concepts` |
| Glossary | [docs/pages/resources/glossary.mdx](./docs/pages/resources/glossary.mdx) | `/resources/glossary` |
| External learning | [docs/pages/resources/external-learning.mdx](./docs/pages/resources/external-learning.mdx) | `/resources/external-learning` |
| Taxonomy | [TAXONOMY.md](./TAXONOMY.md) | Maintainer reference |
| GitHub topics guidance | [GH-TOPICS.MD](./GH-TOPICS.MD) | Maintainer reference |
| Navigation and redirects | [docs.yml](./docs.yml) | Site config |

Published targets are configured in [docs.yml](./docs.yml):

- Preview: `nemo-framework.docs.buildwithfern.com/nemo`
- Production: `docs.nvidia.com/nemo`

## Site Shape

The hub follows the NVIDIA template-library IA, adapted for an ecosystem catalog:

- **About**: overview, ecosystem, architecture, concepts, libraries, release notes.
- **Get Started**: task map, quickstart, installation, runtime chooser, and stage guides.
- **Resources**: glossary, external learning, and community.

Concept pages explain durable relationships. The glossary is lookup-oriented. The task map routes user intent to the owning library. The runtime chooser routes setup decisions to containers, installs, source checkout, or Platform setup.

## Content Rules

Keep content here when it is stable and helps a reader choose:

- Framework vs Platform, lifecycle stages, and repo roles.
- Task-to-library routing.
- Runtime path decisions.
- Catalog metadata from `repos.ts` and `containers.ts`.
- Framework container release metadata and cross-component known issues.
- Terminology and curated external learning links.

Send content downstream when it changes quickly:

- Install commands, API usage, tutorials, and examples.
- Model-specific recipes, benchmarks, and support matrices.
- Version pins and library-only workarounds.

Rule of thumb: add one orienting paragraph and a link instead of copying a library team's material.

## Common Updates

### Add or Update a Repo

Update [components/repos.ts](./components/repos.ts).

- Set `stage` to the org README lifecycle column: `data`, `pretraining`, `rl`, `inference`, or `e2e`.
- Set `kind` to `library`, `integration`, `reference`, or `infrastructure`.
- Add durable `tags` for search facets such as modality, technique, or role.
- Add `docsUrl` and `containerUrl` only when stable public targets exist.

See [TAXONOMY.md](./TAXONOMY.md) for the canonical vocabulary.

### Add or Update a Container

Update [components/containers.ts](./components/containers.ts).

- Keep `latestTag` current.
- Keep `FRAMEWORK_RECENT_RELEASES` to a small recent set.
- Add standalone images to `NEMO_CONTAINERS` when they publish.
- Add cross-component release notes in [known-issues.mdx](./docs/pages/about/release-notes/known-issues.mdx).

### Add a Concept

Add a page under [docs/pages/about/concepts](./docs/pages/about/concepts).

Concept pages should explain relationships, tradeoffs, or decision models. Put short term definitions in [glossary.mdx](./docs/pages/resources/glossary.mdx).

### Add External Learning

Add durable third-party resources to [external-learning.mdx](./docs/pages/resources/external-learning.mdx).

Prefer sources that reveal how external users frame tasks or confusion. Avoid copying commands or version-specific steps.

## Local Development

Prerequisites:

- Node.js 22+
- Fern CLI: `npm install -g fern-api`

Run checks and preview:

```bash
cd fern
fern login
fern check
fern docs dev
```

Open [http://localhost:3000](http://localhost:3000).

Custom React components must be imported in MDX:

```mdx
import RepoCatalog from "@/components/RepoCatalog";

<RepoCatalog />
```

## Publish

Publishing uses the NVIDIA Fern organization token: `DOCS_FERN_TOKEN`.

```bash
git tag docs/v0.1.0 && git push origin docs/v0.1.0
```

You can also run the **Publish Fern Docs** workflow from GitHub Actions.

## Theme

This site uses `global-theme: nvidia`. Theme assets are owned by [fern-components](https://github.com/NVIDIA/fern-components); keep logos, global CSS, and footer changes there.

The GitHub org link is configured in `navbar-links` and `footer-links` in [docs.yml](./docs.yml). After publishing, verify that the NVIDIA global theme renders the header link.
